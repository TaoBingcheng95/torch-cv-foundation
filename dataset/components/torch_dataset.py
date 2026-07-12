import os
import sys
import matplotlib.pyplot as plt

from sympy import root
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, FashionMNIST, MNIST



class MNISTDataLoader:
    def __init__(self, root='./data',
                 download=False,
                 val_split=0.1,      # 默认从训练集分 10% 做验证
                 batch_size=32,
                 use_normalize=True, # 是否归一化
                 seed=42,            # 固定随机种子
                 pin_memory=True,
                 num_workers=0,
                 device='cuda'
                 ):
        super().__init__()

        self.root = root
        self.pin_memory = pin_memory
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device


        # 定义数据的预处理变换
        transform_list = [
            transforms.Resize(32),  # LeNet-5 经典输入是 32x32，MNIST 原是 28x28
            transforms.ToTensor()
        ]
        if use_normalize:
            # MNIST 的均值和标准差
            transform_list.append(transforms.Normalize(mean=(0.1307,), std=(0.3081,)))
        self.transform = transforms.Compose(transform_list)

        self.full_train_ds = MNIST(
            root=root, 
            train=True, 
            download=download,
            transform=self.transform 
        )

        self.test_ds = MNIST(
            root=root,
            train=False,
            download=download,
            transform=self.transform
        )
        self.classes = self.full_train_ds.classes
        self.class_to_idx = self.full_train_ds.class_to_idx
        self.idx_to_class = {value: key for key, value in self.class_to_idx.items()}

        if val_split>0:
            train_count = len(self.full_train_ds)
            val_count = int(train_count * val_split)
            train_count = train_count - val_count
            #【重要】固定随机种子，保证每次运行划分一致
            generator = torch.Generator().manual_seed(seed)
            self.train_ds, self.val_ds = random_split(self.full_train_ds,
                                                      lengths=[train_count, val_count],
                                                      generator=generator)
        else:
            # 如果不划分验证集，通常用测试集充当验证集（不推荐，但为了代码健壮性保留）
            self.train_ds = self.full_train_ds
            self.val_ds = self.test_ds

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def train_dataloader(self) -> DataLoader:
        """对应 LightningDataModule 的 train_dataloader"""
        return DataLoader(
            dataset=self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,       # 训练集必须 shuffle
            num_workers=self.num_workers,      # 初学者建议设为 0，避免 Windows 多进程报错
            pin_memory=self.pin_memory,
            # multiprocessing_context='spawn' if sys.platform.startswith('win') else None
        )

    def val_dataloader(self) -> DataLoader:
        """对应 LightningDataModule 的 val_dataloader"""
        return DataLoader(
            dataset=self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,      # 验证集不能 shuffle
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            # multiprocessing_context='spawn' if sys.platform.startswith('win') else None
        )

    def test_dataloader(self) -> DataLoader:
        """对应 LightningDataModule 的 test_dataloader"""
        return DataLoader(
            dataset=self.test_ds, 
            batch_size=self.batch_size, 
            shuffle=False,      # 测试集不能 shuffle
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            # multiprocessing_context='spawn' if sys.platform.startswith('win') else None
        )
    
    @property
    def num_classes(self) -> int:
        """
        Get the number of classes.

        :return: The number of MNIST classes (10).
        """
        return 10

    def plot_sample(self, loader: DataLoader = None):
        """
        可视化一个 batch 中的数据
        """
        if loader is None:
            loader = self.train_dataloader()
            
        images, labels = next(iter(loader))
        
        # 创建网格图
        fig, axes = plt.subplots(1, 5, figsize=(10, 2))
        
        for i, ax in enumerate(axes):
            img = images[i]
            # 如果是 (1, 32, 32) 需要转为 (32, 32) 或 (32, 32, 1)
            img = img.squeeze() 
            ax.imshow(img, cmap='viridis') # gray
            ax.set_title(f"Label: {labels[i].item()}")
            ax.axis('off')
            
        plt.tight_layout()
        plt.show()
        plt.close()


class FashionMNISTDataLoader:
    def __init__(self, root='./data',
                 download=False,
                 val_split=0.1,      # 默认从训练集分 10% 做验证
                 batch_size=64,
                 use_normalize=True,  # 是否归一化
                 seed=42,            # 固定随机种子
                 pin_memory=True,
                 num_workers=0,
                 device='cuda'
                 ):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.device = device
        
        # 定义数据的预处理变换
        transform_list = [
            transforms.Resize(32),  # LeNet-5 经典输入是 32x32，MNIST 原是 28x28
            transforms.ToTensor()
        ]
        if use_normalize:
            # FashionMNIST 的均值和标准差
            transform_list.append(transforms.Normalize(mean=(0.2860), std=(0.3529)))
        self.transform = transforms.Compose(transform_list)

        self.full_train_ds = FashionMNIST(
            root=root, 
            train=True, 
            download=download,
            transform=self.transform
        )

        self.test_ds = FashionMNIST(
            root=root,
            train=False,
            download=download,
            transform=self.transform 
        )
        self.classes = self.full_train_ds.classes
        self.class_to_idx = self.full_train_ds.class_to_idx
        self.idx_to_class = {value: key for key, value in self.class_to_idx.items()}

        if val_split>0:
            train_count = len(self.full_train_ds)
            val_count = int(train_count * val_split)
            train_count = train_count - val_count
            #【重要】固定随机种子，保证每次运行划分一致
            generator = torch.Generator().manual_seed(seed)
            self.train_ds, self.val_ds = random_split(self.full_train_ds,
                                                      lengths=[train_count, val_count],
                                                      generator=generator)
        else:
            # 如果不划分验证集，通常用测试集充当验证集（不推荐，但为了代码健壮性保留）
            self.train_ds = self.full_train_ds
            self.val_ds = self.test_ds

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def train_dataloader(self) -> DataLoader:
        """对应 LightningDataModule 的 train_dataloader"""
        return DataLoader(
            dataset=self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,       # 训练集必须 shuffle
            num_workers=0,       # 初学者建议设为 0，避免 Windows 多进程报错
            pin_memory=self.pin_memory,
            # multiprocessing_context='spawn' if sys.platform.startswith('win') else None
        )

    def val_dataloader(self) -> DataLoader:
        """对应 LightningDataModule 的 val_dataloader"""
        return DataLoader(
            dataset=self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,      # 验证集不能 shuffle
            num_workers=0,
            pin_memory=self.pin_memory,
            # multiprocessing_context='spawn' if sys.platform.startswith('win') else None
        )

    def test_dataloader(self) -> DataLoader:
        """对应 LightningDataModule 的 test_dataloader"""
        return DataLoader(
            dataset=self.test_ds, 
            batch_size=self.batch_size, 
            shuffle=False,      # 测试集不能 shuffle
            num_workers=0,
            pin_memory=self.pin_memory,
            # multiprocessing_context='spawn' if sys.platform.startswith('win') else None
        )

    @property
    def num_classes(self) -> int:
        """
        Get the number of classes.

        :return: The number of MNIST classes (10).
        """
        return 10

    def plot_sample(self, loader: DataLoader = None):
        """
        可视化一个 batch 中的数据
        """
        if loader is None:
            loader = self.train_dataloader()
            
        images, labels = next(iter(loader))
        
        # 创建网格图
        fig, axes = plt.subplots(1, 5, figsize=(10, 2))
        
        for i, ax in enumerate(axes):
            img = images[i]
            # 如果是 (1, 32, 32) 需要转为 (32, 32) 或 (32, 32, 1)
            img = img.squeeze() 
            ax.imshow(img, cmap='viridis') # gray
            ax.set_title(f"Label: {labels[i].item()}")
            ax.axis('off')
            
        plt.tight_layout()
        plt.show()
        plt.close()


class CIFAR10DataLoader:
    def __init__(self, root='./data',
                 download=False,
                 val_split=0.1,      # 默认从训练集分 10% 做验证
                 batch_size=64,
                 seed=42,            # 固定随机种子
                 ):
        super().__init__()

        self.root = root
        self.batch_size = batch_size
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        # CIFAR-10 标准归一化参数
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2470, 0.2435, 0.2616)

        # 定义 Transform
        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),  # 水平翻转
            # transforms.RandomVerticalFlip(),  # 垂直翻转
            transforms.RandomCrop(32, padding=4), # 随机裁剪，CIFAR 常用增强
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

        self.val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        
        # 加载原始数据集 (先不加 transform，方便分割)
        # 技巧：先加载 transform=None 的完整数据集，分割后再分配 transform
        full_train_ds = CIFAR10(root=self.root, train=True, download=download, transform=None)
        self.test_ds = CIFAR10(root=self.root, train=False, download=download, transform=self.val_transform)
        
        self.class_to_idx = full_train_ds.class_to_idx
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        # 划分训练集和验证集
        if val_split > 0:
            train_count = len(full_train_ds)
            val_count = int(train_count * val_split)
            train_count = train_count - val_count
            #【重要】固定随机种子，保证每次运行划分一致
            generator = torch.Generator().manual_seed(seed)
            train_subset, val_subset = random_split(full_train_ds, [train_count, val_count], 
                                                    generator=generator)
            # 手动为子集分配 transform
            # Subset 数据集内部持有 dataset 对象，我们可以修改其 transform 属性
            train_subset.dataset.transform = self.train_transform
            val_subset.dataset.transform = self.val_transform
            self.train_ds = train_subset
            self.val_ds = val_subset
        else:
            full_train_ds.transform = self.train_transform
            self.train_ds = full_train_ds
            self.val_ds = self.test_ds

    def train_dataloader(self, num_workers=0):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

    def val_dataloader(self, num_workers=0):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

    def test_dataloader(self, num_workers=0):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

    def plot_sample(self, loader=None):
        """
        可视化样本，并正确处理反归一化
        """
        if loader is None:
            loader = self.train_dataloader()
            
        images, labels = next(iter(loader))
        
        fig, axes = plt.subplots(1, 5, figsize=(10, 2))
        for i, ax in enumerate(axes):
            img = images[i]
            # 反归一化公式：img * std + mean
            img = img.permute(1, 2, 0) * torch.tensor(self.std) + torch.tensor(self.mean)
            img = img.clip(0, 1) # 防止数值超出 [0,1] 范围
            ax.imshow(img)
            ax.set_title(self.classes[labels[i]])
            ax.axis('off')
        plt.tight_layout()
        plt.show()
        plt.close()



if __name__ == '__main__':

    data_module = CIFAR10DataLoader(root='data',
                              batch_size=16, 
                              val_split=0.1)
    
    print(f"Train size: {len(data_module.train_ds)}")
    print(f"Val size: {len(data_module.val_ds)}")
    print(f"Test size: {len(data_module.test_ds)}")
    
    # 查看一个 batch 的形状
    train_loader = data_module.train_dataloader()
    imgs, labels = next(iter(train_loader))
    print(f"Batch shape: {imgs.shape}, Labels shape: {labels.shape}")
    
    data_module.plot_sample()
