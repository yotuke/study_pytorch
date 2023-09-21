"""
    网络模型的保存 save 和加载 load
"""
import torch

from ex12_optimizer import CIFAR10_Model

cifar10 = CIFAR10_Model()

# 保存模型结构及参数
torch.save(cifar10, "cifar10_model_01.pkl")

model01 = torch.load("cifar10_model_01.pkl")
print(model01)

# 仅保存模型的状态字典
torch.save(cifar10.state_dict(), "cifar10_state_dict.pkl")
state_dict = torch.load("cifar10_state_dict.pkl")
cifar10.load_state_dict(state_dict)
