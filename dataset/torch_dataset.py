import os
import sys
import matplotlib.pyplot as plt
from typing import Any, Dict, Optional, Tuple

# from sympy import root
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, FashionMNIST, MNIST



class MNISTDataLoader:
    _DEFAULT_RESIZE_SIZE = 32
    # MNIST 的均值和标准差
    MINIST_MEAN = 0.1307
    MINIST_STD = 0.3081
    def __init__(self, 
                 root: str='./data',
                 download: bool = False,
                #  val_split: float = 0.1,      # 默认从训练集分 10% 做验证
                 train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
                 train_val_split: Tuple[int, int] = (55_000, 5_000), # 只分割训练集, 默认从训练集分 10% 做验证
                 batch_size: int = 32,
                 use_normalize: bool = True, # 是否归一化
                 seed: int = 42,            # 固定随机种子
                 pin_memory: bool = True,
                 num_workers: int = 0,
                 device: str ='cuda',
                 world_size: int = 1,
                 ) -> None:
        super().__init__()

        self.root = root
        self.pin_memory = pin_memory
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.world_size = world_size
        self.device = device
        self.generator=torch.Generator()
        self.train_val_split = train_val_split
        self.train_val_test_split = train_val_test_split

        #【重要】固定随机种子，保证每次运行划分一致
        self.generator.manual_seed(seed)

        # 定义数据的预处理变换
        transform_list = [
            transforms.Resize(self._DEFAULT_RESIZE_SIZE),  # LeNet-5 经典输入是 32x32，MNIST 原是 28x28
            transforms.ToTensor()
        ]
        if use_normalize:
            transform_list.append(transforms.Normalize(mean=(self.MINIST_MEAN,), std=(self.MINIST_STD,)))
        self.transform = transforms.Compose(transform_list)

        self.data_test = MNIST(
            root=root,
            train=False,
            download=download,
            transform=self.transform
        ) # 10000 items

        self.full_data_train = MNIST(
            root=root, 
            train=True, 
            download=download,
            transform=self.transform 
        ) # 60000 items

        # random_split 返回的是 Subset 对象，它们会自动继承 full_train_set 的 transform
        self.data_train, self.data_val = random_split(
            dataset=self.full_data_train,
            lengths=self.train_val_split,
            generator=self.generator)

        # if val_split>0:
        #     train_count = len(self.full_train_ds)
        #     val_count = int(train_count * val_split)
        #     train_count = train_count - val_count
            
        #     self.train_ds, self.val_ds = random_split(self.full_train_ds,
        #                                             #   lengths=[train_count, val_count],
        #                                               lengths=self.train_val_split,
        #                                               generator=self.generator)
        # else:
        #     # 如果不划分验证集，通常用测试集充当验证集
        #     self.train_ds = self.full_train_ds
        #     self.val_ds = self.test_ds


        if self.batch_size % self.world_size != 0:
            raise RuntimeError(
                f"Batch size ({self.batch_size}) is not divisible by the number of devices ({self.world_size})."
            )
        self.batch_size_per_device = self.batch_size // self.world_size

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            shuffle=True,                      # 训练集必须 shuffle
            num_workers=self.num_workers,      # 初学者建议设为 0，避免 Windows 多进程报错
            pin_memory=self.pin_memory,
            # 跨 epoch 复用 worker，避免 spawn 平台（macOS/Windows）每轮重建开销；
            # num_workers=0 时必须为 False，否则 DataLoader 报错
            persistent_workers=self.num_workers > 0,
            # multiprocessing_context='spawn' if sys.platform.startswith('win') else None
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            shuffle=False,      # 验证集不能 shuffle
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            # 跨 epoch 复用 worker，避免 spawn 平台（macOS/Windows）每轮重建开销；
            # num_workers=0 时必须为 False，否则 DataLoader 报错
            persistent_workers=self.num_workers > 0,
            # multiprocessing_context='spawn' if sys.platform.startswith('win') else None
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test, 
            batch_size=self.batch_size_per_device, 
            shuffle=False,      # 测试集不能 shuffle
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            # 跨 epoch 复用 worker，避免 spawn 平台（macOS/Windows）每轮重建开销；
            # num_workers=0 时必须为 False，否则 DataLoader 报错
            persistent_workers=self.num_workers > 0,
            # multiprocessing_context='spawn' if sys.platform.startswith('win') else None
        )
    
    @property
    def num_classes(self) -> int:
        """
        Get the number of classes.

        :return: The number of MNIST classes (10).
        """
        return 10

    @property
    def classes(self) -> list[str]:
        return self.full_data_train.classes  # type: ignore
    
    @property
    def class_to_idx(self) -> dict[str, int]:
        return self.full_data_train.class_to_idx

    @property
    def idx_to_class(self) -> dict[int, str]:
        return {value: key for key, value in self.class_to_idx.items()}


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
    def __init__(self, root: str='./data',
                 download: bool = False,
                 val_split: float = 0.1,      # 默认从训练集分 10% 做验证
                 batch_size: int = 64,
                 use_normalize: bool = True,  # 是否归一化
                 seed: int = 42,            # 固定随机种子
                 pin_memory: bool = True,
                 num_workers: int = 0,
                 device: str ='cuda'
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
            # 如果不划分验证集，通常用测试集充当验证集
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
            num_workers=self.num_workers,       # 初学者建议设为 0，避免 Windows 多进程报错
            pin_memory=self.pin_memory,
            # 跨 epoch 复用 worker，避免 spawn 平台（macOS/Windows）每轮重建开销；
            # num_workers=0 时必须为 False，否则 DataLoader 报错
            persistent_workers=self.num_workers > 0,
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
            # 跨 epoch 复用 worker，避免 spawn 平台（macOS/Windows）每轮重建开销；
            # num_workers=0 时必须为 False，否则 DataLoader 报错
            persistent_workers=self.num_workers > 0,
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
            # 跨 epoch 复用 worker，避免 spawn 平台（macOS/Windows）每轮重建开销；
            # num_workers=0 时必须为 False，否则 DataLoader 报错
            persistent_workers=self.num_workers > 0,
            # multiprocessing_context='spawn' if sys.platform.startswith('win') else None
        )

    @property
    def num_classes(self) -> int:
        """
        Get the number of classes.

        :return: The number of MNIST classes (10).
        """
        return 10

    @property
    def classes(self) -> list[str]:
        return self.full_train_ds.classes  # type: ignore
    
    @property
    def class_to_idx(self) -> dict[str, int]:
        return self.full_train_ds.class_to_idx
    
    @property
    def idx_to_class(self) -> dict[int, str]:
        return {value: key for key, value in self.class_to_idx.items()}

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
    def __init__(self, root: str='./data',
                 download:bool =False,
                 val_split: float=0.1,      # 默认从训练集分 10% 做验证
                 batch_size: int=64,
                 seed: int=42,            # 固定随机种子
                 pin_memory: bool = True,
                 num_workers: int = 0,
                 device: str ='cuda'):
        super().__init__()

        self.root = root
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.device = device
        
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
        self.full_train_ds = CIFAR10(root=self.root, train=True, download=download, transform=None)
        self.test_ds = CIFAR10(root=self.root, train=False, download=download, transform=self.val_transform)
        
        # 划分训练集和验证集
        if val_split > 0:
            train_count = len(self.full_train_ds)
            val_count = int(train_count * val_split)
            train_count = train_count - val_count
            #【重要】固定随机种子，保证每次运行划分一致
            generator = torch.Generator().manual_seed(seed)
            train_subset, val_subset = random_split(self.full_train_ds, [train_count, val_count], 
                                                    generator=generator)
            # 手动为子集分配 transform
            # Subset 数据集内部持有 dataset 对象，我们可以修改其 transform 属性
            train_subset.dataset.transform = self.train_transform
            val_subset.dataset.transform = self.val_transform
            self.train_ds = train_subset
            self.val_ds = val_subset
        else:
            self.full_train_ds.transform = self.train_transform
            self.train_ds = self.full_train_ds
            self.val_ds = self.test_ds

    def train_dataloader(self):
        return DataLoader(self.train_ds, 
                          batch_size=self.batch_size, 
                          shuffle=True, 
                          num_workers=self.num_workers,
                          # 跨 epoch 复用 worker，避免 spawn 平台（macOS/Windows）每轮重建开销；
                          # num_workers=0 时必须为 False，否则 DataLoader 报错
                          persistent_workers=self.num_workers > 0,
                          )

    def val_dataloader(self):
        return DataLoader(self.val_ds, 
                          batch_size=self.batch_size, 
                          shuffle=False, 
                          num_workers=self.num_workers,
                          persistent_workers=self.num_workers > 0,)

    def test_dataloader(self, num_workers=0):
        return DataLoader(self.test_ds, 
                          batch_size=self.batch_size, 
                          shuffle=False, 
                          num_workers=self.num_workers,
                          persistent_workers=self.num_workers > 0,)

    @property
    def num_classes(self) -> int:
        return 10

    @property
    def classes(self) -> list[str]:
        # self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        return self.full_train_ds.classes  # type: ignore
    
    @property
    def class_to_idx(self) -> dict[str, int]:
        return self.full_train_ds.class_to_idx

    @property
    def idx_to_class(self) -> dict[int, str]:
        return {value: key for key, value in self.class_to_idx.items()}
    
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
