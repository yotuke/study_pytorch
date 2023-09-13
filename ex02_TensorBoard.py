"""
    TensorBoard 的使用
"""
# from tensorboardX import SummaryWriter
from tensorboardX import SummaryWriter
if __name__ == "__main__":
    writer = SummaryWriter("test_log")
    for i in range(100):
        writer.add_scalar("y=x^2", i*i, i)
        pass
    writer.close()
    pass
