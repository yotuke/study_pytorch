"""
    TensorBoard 的使用
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
