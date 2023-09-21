"""
    Torch.optim 的作用是最小化损失函数，以提高模型的性能。它通过不断更新模型的参数，使损失函数的值逐渐减小，从而提高模型的准确性。
"""
from tensorboardX import SummaryWriter
from torch import nn, optim
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
    optim = optim.SGD(cifar10.parameters(), lr=0.01)

    for epoch in range(20):
        loss_epoch = 0
        print(f"第{epoch}轮训练开始......")
        for batch_data in CIFAR10_loader:
            optim.zero_grad()
            imgs, targets = batch_data
            output = cifar10(imgs)
            loss_batch = loss_cross_entropy(output, targets)
            loss_epoch = loss_epoch + loss_batch
            loss_batch.backward()
            optim.step()
        print("第%d轮的损失函数是: %f" % (epoch, loss_epoch))
        writer.add_scalar("log_loss", loss_epoch, epoch)

    writer.close()
