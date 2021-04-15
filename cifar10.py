import torch
import torchvision as tv
import torch.backends.cudnn as cudnn
import os
import time
import random
import argparse
import numpy as np


# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)

cudnn.benchmark = True

def main(args):
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

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.bs,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=4)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.bs,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=4)

    # net
    net = tv.models.resnet18(num_classes=10)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)
    if args.multi_gpu and torch.cuda.device_count() > 1:
        print(torch.cuda.device_count())
        net = torch.nn.DataParallel(net)

    # optimizer and loss
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [50, 80], 0.1)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # train
    net.train()
    for i_epoch in range(100):
        time_s = time.time()
        for i_iter, data in enumerate(train_loader):
            img, label = data
            img, label = img.to(device), label.to(device)

            optimizer.zero_grad()
            feat = net(img)
            loss = criterion(feat, label)
            loss.backward()
            optimizer.step()
            time_e = time.time()

            print('Epoch:{:3}/100 || Iter: {:4}/{} || '
                        'Loss: {:2.4f} '
                        'ETA: {:.2f}min'.format(
                i_epoch + 1, i_iter + 1, len(train_loader),
                loss.item(),
                (time_e - time_s) * (100 - i_epoch) * len(train_loader) / (i_iter + 1) / 60))
        scheduler.step()

    # valid
    net.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0.
    total = 0.
    for images, labels in test_loader:
        images = images.cuda()
        labels = labels.cuda()

        with torch.no_grad():
            pred = net(images)

        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    val_acc = correct / total
    print(i_epoch, val_acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--data_augmentation', type=bool, default=True)
    parser.add_argument('--multi_gpu', type=bool, default=False)

    args = parser.parse_args()

    main(args)