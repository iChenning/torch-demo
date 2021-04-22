import torch
import torchvision as tv
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as ddp
import os
import time
import random
import argparse
import numpy as np


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)

cudnn.benchmark = True

def main(args):
    # dist init
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    print(torch.cuda.device_count(), args.local_rank)

    # data
    train_transform = tv.transforms.Compose([])
    if args.data_augmentation:
        train_transform.transforms.append(tv.transforms.RandomCrop(32, padding=4))
        train_transform.transforms.append(tv.transforms.RandomHorizontalFlip())
    train_transform.transforms.append(tv.transforms.ToTensor())
    normalize = tv.transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                        std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    train_transform.transforms.append(normalize)

    test_transform = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        normalize])

    train_dataset = tv.datasets.CIFAR10(root='data/',
                                     train=True,
                                     transform=train_transform,
                                     download=True)

    test_dataset = tv.datasets.CIFAR10(root='data/',
                                    train=False,
                                    transform=test_transform,
                                    download=True)
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.bs,
                                               shuffle=False,
                                               pin_memory=True,
                                               num_workers=4,
                                               sampler=sampler)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.bs,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=4)

    # net
    net = tv.models.resnet18(num_classes=10)
    net = net.cuda()

    # optimizer and loss
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [50, 80], 0.1)
    criterion = torch.nn.CrossEntropyLoss().cuda()

    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = ddp(net, device_ids=[args.local_rank], find_unused_parameters=True)

    # train
    for i_epoch in range(100):
        net.train()
        time_s = time.time()
        train_loader.sampler.set_epoch(i_epoch)
        for i_iter, data in enumerate(train_loader):
            img, label = data
            img, label = img.cuda(), label.cuda()

            optimizer.zero_grad()
            feat = net(img)
            loss = criterion(feat, label)
            loss.backward()
            optimizer.step()
            time_e = time.time()

            if args.local_rank == 1:
                print('Epoch:{:3}/100 || Iter: {:4}/{} || '
                        'Loss: {:2.4f} '
                        'ETA: {:.2f}min'.format(
                i_epoch + 1, i_iter + 1, len(train_loader),
                loss.item(),
                (time_e - time_s) * (100 - i_epoch) * len(train_loader) / (i_iter + 1) / 60))
        scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--data_augmentation', type=bool, default=True)
    parser.add_argument("--local_rank", type=int, default=0, help='master rank')

    args = parser.parse_args()
    main(args)