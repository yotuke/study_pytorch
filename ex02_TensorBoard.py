"""
********************************************************************
TensorBoard 的使用
    Tensorboard 是一个用于监测和可视化深度学习模型训练过程的工具。它可以帮助您：
    1.观察模型在训练过程中的损失和准确率。
    2.查看模型参数的变化情况。
    3.分析模型在不同数据集上的性能。
    4.可视化模型在训练过程中的激活图和热力图。
********************************************************************
"""
from tensorboardX import SummaryWriter
from PIL import Image
import numpy

IMAGE_URL = "dataset_demo/train/ants/0013035.jpg"


# writer.add_scalar()的使用
def add_scalar():
    writer = SummaryWriter("log_scalar")
    for i in range(-50, 51):
        writer.add_scalar("y=x^2", i * i, i)
        pass
    writer.close()
    pass


# writer.add_image()的使用
def add_image():
    writer = SummaryWriter("log_image")
    img = Image.open(IMAGE_URL)
    img_ndarray = numpy.array(img)  # HWC
    writer.add_image("tensorboard_image", img_ndarray, 1, dataformats="HWC")
    writer.close()
    pass


if __name__ == "__main__":
    add_scalar()    # writer.add_scalar()的使用
    add_image()     # writer.add_image()的使用
    pass
