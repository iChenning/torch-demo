import torch
import torchvision as tv
import torch.backends.cudnn as cudnn
from torch.quantization import get_default_qconfig, quantize_jit
from torch.quantization.quantize_fx import prepare_fx, convert_fx
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
    model = copy.deepcopy(net).cuda()
    del net
    model.eval()
    graph_module = torch.fx.symbolic_trace(model)
    qconfig = get_default_qconfig("fbgemm")
    qconfig_dict = {"": qconfig}
    model_prepared = prepare_fx(graph_module, qconfig_dict)
    calibrate(model_prepared, test_loader)  # 这一步是做后训练量化
    model_int8 = convert_fx(model_prepared)
    torch.jit.save(torch.jit.script(model_int8), 'int8-ptq.pth')

    # valid
    loaded_quantized_model = torch.jit.load('int8-ptq.pth')
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