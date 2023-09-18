"""
    nn.module 是 Python 中的一个内置模块，用于处理神经网络（Neural Network）相关的任务。
    它提供了一些函数和类，用于构建、训练和评估神经网络模型。
    在 nn.module 中，包含了许多神经网络层的实现，如全连接层（fully connected layer）、卷积层（convolutional layer）、循环层（Recurrent layer）
等。
    此外，它还提供了一些优化算法和损失函数的实现，如 SGD、Adagrad、Adam 等优化算法，以及 CrossEntropy、MeanSquaredError 等损失函数。
    自建的神经网络模型要继承nn.module类，并重写nn.module的 __init__() 和 __forward__()方法
    class ModelDemo(nn.module):
        def __init__(self):
            super().__init__()
                  ...
        pass

        def forward():
            ...
        pass
"""
import torch.nn as nn


class ModelDemo(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return

    pass