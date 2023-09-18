"""
激活函数（Activation Function）通常作为神经网络中的一个组件，用于将输入的数值映射到输出的数值。激活函数的作用主要有以下几点：
     1.非线性变换：传统的线性模型难以处理复杂的非线性关系，而激活函数的使用可以使得神经网络具有非线性的表达能力，从而能够更好地拟合复杂的数据分布。
     2.控制网络的复杂度：通过选择不同的激活函数，可以控制神经网络在不同的区域具有不同的敏感度，从而避免过度拟合或欠拟合的问题。
     3.引入信息损失：通过使用激活函数，可以在一定程度上引入信息损失，使得神经网络更加稳定，不易受到噪声的影响。
     4.优化求解：一些激活函数具有可导性，这使得在使用梯度下降等优化算法时，可以更加方便地计算梯度，从而加速网络的训练过程。
"""
from torch import nn
from torch.nn import Sigmoid
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from tensorboardX import SummaryWriter

CIFAR10_test = datasets.CIFAR10("CIFAR10", False, transforms.ToTensor(), download=True)
CIFAR10_loader = DataLoader(CIFAR10_test, 64, drop_last=True)


class SigMoid_Demo(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = Sigmoid()

    def forward(self, x):
        return self.sigmoid(x)
    pass


sigmoid_demo = SigMoid_Demo()
writer = SummaryWriter("log_img")
step = 1
for batch_data in CIFAR10_loader:
    imgs, targets = batch_data
    imgs_sigmoid = sigmoid_demo(imgs)
    writer.add_images("before_sigmoid", imgs, step)
    writer.add_images("after_sigmoid", imgs_sigmoid, step)
    step = step + 1

writer.close()
