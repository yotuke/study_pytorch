"""
    在机器学习和统计学中，损失函数是一个非常重要的概念。它的主要作用是衡量模型预测结果与真实结果之间的差距，也就是模型的误差。
    损失函数的值越小，说明模型的预测误差越小，模型的性能越好。
"""
from tensorboardX import SummaryWriter
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear, CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CIFAR10_test = datasets.CIFAR10("CIFAR10", False, transforms.ToTensor(), download=True)
CIFAR10_loader = DataLoader(CIFAR10_test, 32, drop_last=True)


class CIFAR10_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Sequential(
            Conv2d(3, 32, 5, 1, 2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, 1, 2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, 1, 2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10))

    def forward(self, x):
        return self.model(x)

    pass


if __name__ == "__main__":

    cifar10 = CIFAR10_Model()
    loss_cross_entropy = CrossEntropyLoss()
    writer = SummaryWriter("log_loss")
    step = 1
    for batch_data in CIFAR10_loader:
        imgs, targets = batch_data
        output = cifar10(imgs)
        loss_result = loss_cross_entropy(output, targets)
        print(loss_result)
        writer.add_scalar("CrossEntropyLoss", loss_result, step)
        step = step + 1

    writer.close()
