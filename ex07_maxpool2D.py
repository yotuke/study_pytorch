"""
    MaxPool 2D 是一种深度学习中的常用操作，用于对二维数据进行下采样。它通过在二维空间中选择一个固定大小的窗口，并对窗口内的值取最大值，来实现数
据的下采样。这个操作可以减少数据的大小，同时保留关键信息，在图像处理、语音识别等领域有广泛的应用。
    MaxPool 2D 通常与卷积层一起使用，作为卷积层的后续操作。它可以帮助降低特征图的分辨率，同时保留重要的空间信息，有助于提高模型的准确性和效率。
"""
from tensorboardX import SummaryWriter
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CIFAR10_test = datasets.CIFAR10("CIFAR10", False, transforms.ToTensor(), download=True)
CIFAR10_loader = DataLoader(CIFAR10_test, 64, True, drop_last=True)


class Maxpool2D_Demo(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool2d = MaxPool2d(3)

    def forward(self, x):
        return self.maxpool2d(x)
    pass


maxpool2d_demo = Maxpool2D_Demo()
writer = SummaryWriter("log_img")

step = 1
for batch_data in CIFAR10_loader:
    imgs, targets = batch_data
    imgs_maxpool2d = maxpool2d_demo(imgs)
    writer.add_images("before_maxpool2d", imgs, step)
    writer.add_images("after_maxpool2d", imgs_maxpool2d, step)
    step = step + 1

writer.close()