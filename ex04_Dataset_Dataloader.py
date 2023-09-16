"""
Dataset
在数据科学和机器学习中，"Dataset" 是指一组相关的数据样本。这些样本可以是任何形式的数据，例如图像、文本、音频、视频等。
一个 dataset 通常由以下几个部分组成：
    ~数据集名称：用于标识该数据集的名称。
    ~数据样本：数据集中的具体样本，每个样本都有一定的特征和标签。
    ~特征：描述样本的特征信息，例如图像的像素值、文本的单词等。
    ~标签：描述样本的分类或预测结果，例如图像的类别标签、文本的情感标签等。
    ~数据集的元数据：例如数据集的创建时间、作者、数据来源等信息。

Dataloader
    DataLoader 是 Python 中的一个工具，用于从数据集中加载数据，并将其提供给机器学习模型进行训练或预测。
    DataLoader 通常用于处理大规模数据集，它可以通过分批加载数据来提高数据处理的效率。它还可以实现随机洗牌、数据增强等功能，以提高模型的泛化能力。
"""
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tensorboardX import SummaryWriter

CIFAR10_train = datasets.CIFAR10("./CIFAR10", True, transforms.ToTensor(), download=True)
CIFAR10_test  = datasets.CIFAR10("./CIFAR10", False, transforms.ToTensor(), download=True)

print("CIFAR10_train.length = ", len(CIFAR10_train))
print("CIFAR10_test.length  = ", len(CIFAR10_test))

CIFAR10_train_loader = DataLoader(CIFAR10_train, 64, True, drop_last=True)
CIFAR10_test_loader  = DataLoader(CIFAR10_test,  64, True, drop_last=True)

print("CIFAR10_train.loader.length = ", len(CIFAR10_train_loader))
print("CIFAR10_test.loader.length  = ", len(CIFAR10_test_loader))

writer = SummaryWriter("log_img")

step_train = 1
for batch_data in CIFAR10_train_loader:
    imgs, targets = batch_data
    writer.add_images("CIFAR10_train", imgs, step_train)
    step_train = step_train + 1

step_test = 1
for batch_data in CIFAR10_test_loader:
    imgs, targets = batch_data
    writer.add_images("CIFAR10_test", imgs, step_test)
    step_test = step_test + 1

writer.close()
