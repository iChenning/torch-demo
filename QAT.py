import torch
import torchvision as tv
import torch.backends.cudnn as cudnn
from torch.quantization.quantize_fx import prepare_qat_fx, convert_fx
import os
import time
import random
import argparse
import numpy as np
import copy
from tqdm import  tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)

cudnn.benchmark = True


def calibrate(model, data_loader):
    model.eval()
    with torch.no_grad():
        for image, target in tqdm(data_loader):
            model(image.cuda())


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
    net = tv.models.mobilenet_v2(num_classes=10)
    net.load_state_dict(torch.load('mobilenet_v2.pth', map_location='cpu'))
    net.dropout = torch.nn.Sequential()

    # quantization
    model_to_quantize = copy.deepcopy(net).to(device)
    qconfig_dict = {"": torch.quantization.get_default_qat_qconfig('fbgemm')}
    model_to_quantize.train()
    model_prepared = prepare_qat_fx(model_to_quantize, qconfig_dict)
    # optimizer and loss
    optimizer = torch.optim.SGD(model_prepared.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 16], 0.1)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    # train
    model_prepared.train()
    for i_epoch in range(20):
        time_s = time.time()
        for i_iter, data in enumerate(train_loader):
            img, label = data
            img, label = img.to(device), label.to(device)

            optimizer.zero_grad()
            feat = model_prepared(img)
            loss = criterion(feat, label)
            loss.backward()
            optimizer.step()
            time_e = time.time()

            print('Epoch:{:3}/20 || Iter: {:4}/{} || '
                  'Loss: {:2.4f} '
                  'ETA: {:.2f}min'.format(
                i_epoch + 1, i_iter + 1, len(train_loader),
                loss.item(),
                (time_e - time_s) * (20 - i_epoch) * len(train_loader) / (i_iter + 1) / 60))
        scheduler.step()

    # to int8
    model_int8 = convert_fx(model_prepared)
    torch.jit.save(torch.jit.script(model_int8), 'int8-qat.pth')

    # valid
    loaded_quantized_model = torch.jit.load('int8-qat.pth')
    correct = 0.
    total = 0.
    with torch.no_grad():
        loaded_quantized_model.eval()
        for images, labels in tqdm(test_loader):
            images = images
            labels = labels

            pred = loaded_quantized_model(images)

            pred = torch.max(pred.data, 1)[1]
            total += labels.size(0)
            correct += (pred == labels).sum().item()

        val_acc = correct / total
        print(val_acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=256)
    parser.add_argument('--data_augmentation', type=bool, default=True)
    parser.add_argument('--multi_gpu', type=bool, default=False)

    args = parser.parse_args()

    main(args)