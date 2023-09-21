"""
    torch.nn.Linear 层可以被用于各种神经网络结构中，如全连接层（Fully Connected Layer）、卷积层（Convolutional Layer）等。
    它是构建深度学习模型的重要组成部分，用于对输入数据进行线性变换，为后续的非线性处理提供基础。
"""
import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CIFAR10_test = datasets.CIFAR10("CIFAR10", False, transforms.ToTensor(), download=True)
CIFAR10_loader = DataLoader(CIFAR10_test, 32, drop_last=True)


class Linear_Demo(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = Linear(98304, 10)

    def forward(self, x):
        return self.linear(x)

    pass


linear_demo = Linear_Demo()

for batch_data in CIFAR10_loader:
    imgs, targets = batch_data
    imgs_linear = linear_demo(torch.flatten(imgs))
    print(imgs_linear.shape)  # torch.Size([10])


