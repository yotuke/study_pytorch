"""
    torch.nn.Sequential是 PyTorch 中的一个模块，用于组织多个神经网络层，以形成一个顺序的网络架构。
使用 torch.nn.Sequential 可以方便地定义一个顺序的网络架构，只需将需要的神经网络层按顺序添加到 torch.nn.Sequential
"""
import torch
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

CIFAR10_test = datasets.CIFAR10("CIFAR10", False, transforms.ToTensor(), download=True)
CIFAR10_loader = DataLoader(CIFAR10_test, 64, shuffle=True, drop_last=True)


class CIFAR10(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Sequential(
            Conv2d(3, 32, 5, 1, 2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, 1, 2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, 1 ,2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        return self.model(x)

    pass


if __name__ == "__main__":
    cifar10 = CIFAR10()
    for batch_data in CIFAR10_loader:
        list_argmax = []
        imgs, targets = batch_data
        outputs = cifar10(imgs)
        for img in outputs:
            list_argmax.append(torch.argmax(img).item())
        print(list_argmax)
