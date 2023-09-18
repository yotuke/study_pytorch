"""
    在深度学习中，卷积被广泛应用于卷积神经网络（Convolutional Neural Network, CNN）中。
    在CNN中，卷积层通过使用卷积核来提取图像的特征，并将这些特征传递给后续的层进行处理。
    卷积层是CNN的核心组成部分，通过不断地使用卷积核来提取图像的不同层次的特征，从而实现图像的分类、识别等任务。
"""
import torch.nn as nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tensorboardX import SummaryWriter

CIFAR10_test = datasets.CIFAR10("./CIFAR10", False, transforms.ToTensor(), download=True)
CIFAR10_loader = DataLoader(CIFAR10_test, 64, True, drop_last=True)


class ConvDemo(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = Conv2d(3, 3, 4, 1, 0)

    def forward(self, x):
        return self.conv_1(x)

    pass


conv_demo = ConvDemo()
writer = SummaryWriter("log_img")

step = 1
for batch_data in CIFAR10_loader:
    imgs, targets = batch_data
    imgs_conv = conv_demo(imgs)
    writer.add_images("before_conv", imgs, step)
    writer.add_images("after_conv", imgs_conv, step)
    step = step + 1

writer.close()


