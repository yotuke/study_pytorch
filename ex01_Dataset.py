"""
    Dataset类代码实战
    Dataset：获取数据集的数据及对应的标签
        所有表示key-value数据样本的数据集都需要继承它。
        并且所有子类都要重写__getitem__方法，返回指定的数据及其标签的
    Dataloader：向模型中装载数据
"""

from torch.utils.data import Dataset
from PIL import Image
import os


class ModelDemo(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.imgs_dir = os.path.join(self.root_dir, self.label_dir)
        self.imgs_path_list = os.listdir(self.imgs_dir)
        pass

    def __getitem__(self, idx):
        img_name = self.imgs_path_list[idx]
        img_path = os.path.join(self.imgs_dir, img_name)
        img = Image.open(img_path)
        label = self.label_dir
        return img, label
        pass

    def __len__(self):
        return len(self.imgs_path_list)
        pass

    pass


if __name__ == "__main__":
    ants_model = ModelDemo("dataset_demo/train", "ants")
    iter_ants = iter(ants_model)    # 将ants_model转为迭代器对象
    for img, label in iter_ants:
        img.show()
        pass

    bees_model = ModelDemo("dataset_demo/train", "bees")
    iter_bees = iter(bees_model)
    for img, label in iter_bees:
        img.show()
        pass
    pass

