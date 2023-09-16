"""
************************************************************************************************************************
transform的使用
    在深度学习中，”Transform” 是一种数据预处理技术，用于将原始数据转换为适合机器学习算法处理的形式。它的目的是将数据转化为更易于理解和处理的形
式，以便算法可以更好地学习和提取特征。常见的数据预处理技术包括：
    1.归一化：将数据的值映射到一个特定的范围内，通常是[0,1]或[-1,1]。
    2.标准化：将数据的每一列除以该列的标准偏差，使得每一列的均值为 0，标准偏差为 1。
    3.正则化：通过对数据进行变换，使得数据的分布更加接近高斯分布。
    4.特征选择：选择对目标任务有意义的特征，并去除不相关或冗余的特征。
    5.缺失值处理：处理数据中存在的缺失值，例如使用平均值、中位数或众数代替缺失值。
这些数据预处理技术可以帮助机器学习算法更好地理解和处理数据，从而提高模型的准确性和泛化能力。在深度学习中，Transform 通常是在训练模型之前进行的，
并且是模型训练过程中的重要组成部分。
    常用的 transforms 方法：
        1. transforms.ToTensor() 输入: PIL Image`` or ``numpy.ndarray输出 torch.Tensor
        2. transforms.Normalize(param1， param2)
            ~输入: 设定的n个通道的均值和标准差
            ~用给定的平均值和标准差对tensor图像进行归一化处理, 该变换将对输入的每个通道进行归一化处理
            ~归一化的目的是使图像中的像素值具有可比性，这样可以使图像在不同的处理步骤中更容易进行比较和计算。此外，归一化还可以减少图像中像素值
        的范围，使得图像更容易被计算机处理，并且可以减少数值溢出的风险。
        3.transforms.Resize((param1, param2)) 参数：给定的（H, W）
            ~用给定的（H，W）对 PIL Image进行尺寸变换，返回值仍然是 PIL Image
        4.transforms.Compose([transforms操作1, ... , transforms操作2])
            ~将多个transforms操作组合在一起，可以简化代码，使转换过程更加清晰和易于维护
            ~e.g.:transforms.Compose([transforms.Resize((1080, 1080)), transforms.ToTensor()])
************************************************************************************************************************
"""
from torchvision import transforms
from PIL import Image
from tensorboardX import SummaryWriter
IMAGE_URL = "animals/labrador.jpg"


img = Image.open(IMAGE_URL)
writer = SummaryWriter("log_img")

# transforms.ToTensor() 类的使用
img_tensor = transforms.ToTensor()(img)  # <class 'torch.Tensor'>

# transforms.Normalize()类的使用
img_normalize = transforms.Normalize([10, 2, 6], [2, 1, 8])(img_tensor)
writer.add_image("by transforms.Normalize", img_normalize, 2)


# transforms.Compose()类的使用
img_compose = transforms.Compose([transforms.ToTensor(), transforms.Normalize([1, 2, 4], [1, 5, 8])])(img)
writer.add_image("by transforms.Normalize", img_normalize, 3)
writer.close()
