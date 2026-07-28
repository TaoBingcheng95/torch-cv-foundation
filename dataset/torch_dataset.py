import matplotlib.pyplot as plt
from typing import Optional, Union

import torch
from torch.utils.data import DataLoader, Subset, random_split
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, FashionMNIST, MNIST



class MNISTDataLoader:
    # MNIST 的均值和标准差
    MNIST_MEAN = 0.1307
    MNIST_STD = 0.3081
    DEFAULT_SIZE = 28
    RESIZE_SIZE = 32
    DEFAULT_TRAIN_LENGTH = 60000
    DEFAULT_TEST_LENGTH = 10000
    def __init__(self, 
                 root: str='./data',
                 download: bool = False,
                 # 验证集划分：float ∈ (0,1) 按比例从训练集切分，int ≥ 1 按绝对数量切分；
                 # 官方 test 集保持不动，保证指标与外部基准可比
                 val_split: Union[int, float] = 0.1,
                 batch_size: int = 32,
                 use_normalize: bool = True, # 是否归一化
                 # 仅 CUDA 设备受益（锁页内存加速 Host→GPU 拷贝），CPU/MPS 无效且会告警；
                 # 默认关闭，由调用方按设备显式开启（参考 auto_pin_memory）
                 pin_memory: bool = False,
                 num_workers: int = 0,
                 seed: int = 42,             # 固定随机种子
                 ) -> None:
        super().__init__()

        self.root = root
        self.pin_memory = pin_memory
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_normalize = use_normalize   # plot_sample 反归一化时需要
        # self.device = device
        #【重要】固定随机种子，保证每次运行划分一致
        self.generator=torch.Generator()
        self.generator.manual_seed(seed)

        # 定义数据的预处理变换
        transform_list = [
            transforms.Resize(self.RESIZE_SIZE),  # LeNet-5 经典输入是 32x32，MNIST 原是 28x28
            transforms.ToTensor()
        ]
        if use_normalize:
            transform_list.append(transforms.Normalize(mean=self.MNIST_MEAN, 
                                                       std=self.MNIST_STD))
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
        if val_split > 0:
            total_count = len(self.full_data_train)
            # float 按比例、int 按绝对数量
            val_count = int(total_count * val_split) if isinstance(val_split, float) else val_split
            train_count = total_count - val_count
            self.data_train, self.data_val = random_split(
                dataset=self.full_data_train,
                lengths=[train_count, val_count],
                generator=self.generator)
        else:
            # 不划分验证集时用测试集充当；注意若据此做模型选择/早停，
            # 测试指标会虚高（信息泄漏），仅适合快速实验
            self.data_train = self.full_data_train
            self.data_val = self.data_test

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
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
            batch_size=self.batch_size,
            shuffle=False,      # 验证集不能 shuffle
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            # multiprocessing_context='spawn' if sys.platform.startswith('win') else None
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test, 
            batch_size=self.batch_size, 
            shuffle=False,      # 测试集不能 shuffle
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
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


    def plot_sample(self, loader: Optional[DataLoader] = None):
        """
        可视化一个 batch 中的数据
        """
        if loader is None:
            loader = self.test_dataloader()
        
        images, labels = next(iter(loader))
        
        # 创建网格图：按实际 batch 大小取列数；squeeze=False 保证 axes 恒为二维数组，
        # 避免 batch_size=1 时返回单个 Axes 导致遍历报错
        ncols = min(images.shape[0], 5)
        fig, axes = plt.subplots(1, ncols, figsize=(10, 2), squeeze=False)
        
        for i, ax in enumerate(axes[0]):
            img = images[i]
            # 如果是 (1, 32, 32) 需要转为 (32, 32) 或 (32, 32, 1)
            img = img.squeeze()
            if self.use_normalize:
                # 反归一化公式：img * std + mean，并截断到 [0,1] 便于显示
                img = (img * self.MNIST_STD + self.MNIST_MEAN).clip(0, 1)
            ax.imshow(img, cmap='viridis') # gray
            ax.set_title(f"Label: {labels[i].item()}")
            ax.axis('off')
            
        plt.tight_layout()
        plt.show()
        plt.close()



class FashionMNISTDataLoader:
    # FashionMNIST 的均值和标准差
    FASHIONMNIST_MEAN = 0.2860
    FASHIONMNIST_STD = 0.3529
    DEFAULT_SIZE = 28
    RESIZE_SIZE = 32
    DEFAULT_TRAIN_LENGTH = 60000
    DEFAULT_TEST_LENGTH = 10000
    def __init__(self, root: str='./data',
                 download: bool = False,
                 # 验证集划分：float ∈ (0,1) 按比例从训练集切分，int ≥ 1 按绝对数量切分；
                 # 官方 test 集保持不动，保证指标与外部基准可比
                 val_split: Union[int, float] = 0.1,
                 batch_size: int = 64,
                 use_normalize: bool = True,  # 是否归一化
                 # 仅 CUDA 设备受益（锁页内存加速 Host→GPU 拷贝），CPU/MPS 无效且会告警；
                 # 默认关闭，由调用方按设备显式开启（参考 auto_pin_memory）
                 pin_memory: bool = False,
                 num_workers: int = 0,
                 seed: int = 42,            # 固定随机种子
                 ):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.use_normalize = use_normalize   # plot_sample 反归一化时需要
        #【重要】固定随机种子，保证每次运行划分一致
        self.generator=torch.Generator()
        self.generator.manual_seed(seed)
        
        # 定义数据的预处理变换
        transform_list = [
            transforms.Resize(self.RESIZE_SIZE),  # LeNet-5 经典输入是 32x32，FashionMNIST 原是 28x28
            transforms.ToTensor()
        ]
        if use_normalize:
            transform_list.append(transforms.Normalize(mean=self.FASHIONMNIST_MEAN, 
                                                       std=self.FASHIONMNIST_STD))
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

        if val_split > 0:
            total_count = len(self.full_train_ds)
            # float 按比例、int 按绝对数量
            val_count = int(total_count * val_split) if isinstance(val_split, float) else val_split
            train_count = total_count - val_count
            self.train_ds, self.val_ds = random_split(self.full_train_ds,
                                                      lengths=[train_count, val_count],
                                                      generator=self.generator)
        else:
            # 不划分验证集时用测试集充当；注意若据此做模型选择/早停，
            # 测试指标会虚高（信息泄漏），仅适合快速实验
            self.train_ds = self.full_train_ds
            self.val_ds = self.test_ds

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,                       # 训练集必须 shuffle
            num_workers=self.num_workers,       # 初学者建议设为 0，避免 Windows 多进程报错
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            # multiprocessing_context='spawn' if sys.platform.startswith('win') else None
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,      # 验证集不能 shuffle
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            # multiprocessing_context='spawn' if sys.platform.startswith('win') else None
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_ds, 
            batch_size=self.batch_size, 
            shuffle=False,      # 测试集不能 shuffle
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            # multiprocessing_context='spawn' if sys.platform.startswith('win') else None
        )

    @property
    def num_classes(self) -> int:
        """
        Get the number of classes.

        :return: The number of FashionMNIST classes (10).
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

    def plot_sample(self, loader: Optional[DataLoader] = None):
        """
        可视化一个 batch 中的数据
        """
        if loader is None:
            loader = self.test_dataloader() 
        images, labels = next(iter(loader))
        
        # 创建网格图：按实际 batch 大小取列数；squeeze=False 保证 axes 恒为二维数组，
        # 避免 batch_size=1 时返回单个 Axes 导致遍历报错
        ncols = min(images.shape[0], 5)
        fig, axes = plt.subplots(1, ncols, figsize=(10, 2), squeeze=False)
        
        for i, ax in enumerate(axes[0]):
            img = images[i]
            # 如果是 (1, 32, 32) 需要转为 (32, 32) 或 (32, 32, 1)
            img = img.squeeze() 
            if self.use_normalize:
                # 反归一化公式：img * std + mean，并截断到 [0,1] 便于显示
                img = (img * self.FASHIONMNIST_STD + self.FASHIONMNIST_MEAN).clip(0, 1)
            ax.imshow(img, cmap='viridis') # gray
            ax.set_title(f"Label: {labels[i].item()}")
            ax.axis('off')

        plt.tight_layout()
        plt.show()
        plt.close()



class CIFAR10DataLoader:
    # CIFAR-10 标准归一化参数
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD = (0.2470, 0.2435, 0.2616)
    DEFAULT_SIZE = 28
    RESIZE_SIZE = 32
    DEFAULT_TRAIN_LENGTH = 50000
    DEFAULT_TEST_LENGTH = 10000
    def __init__(self, root: str='./data',
                 download: bool = False,
                 # 验证集划分：float ∈ (0,1) 按比例从训练集切分，int ≥ 1 按绝对数量切分；
                 # 官方 test 集保持不动，保证指标与外部基准可比
                 val_split: Union[int, float] = 0.1,
                 batch_size: int = 64,
                 use_normalize: bool = True,  # 是否归一化
                 # 仅 CUDA 设备受益（锁页内存加速 Host→GPU 拷贝），CPU/MPS 无效且会告警；
                 # 默认关闭，由调用方按设备显式开启（参考 auto_pin_memory）
                 pin_memory: bool = False,
                 num_workers: int = 0,
                 seed: int = 42,            # 固定随机种子
                 ):
        super().__init__()

        self.root = root
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.use_normalize = use_normalize   # plot_sample 反归一化时需要
        #【重要】固定随机种子，保证每次运行划分一致
        self.generator=torch.Generator()
        self.generator.manual_seed(seed)
        
        # 定义 Transform
        train_transform_list = [
            transforms.RandomHorizontalFlip(),  # 水平翻转
            # transforms.RandomVerticalFlip(),  # 垂直翻转
            transforms.RandomCrop(self.RESIZE_SIZE, padding=4), # 随机裁剪，CIFAR 常用增强
            transforms.ToTensor(),
        ]
        val_transform_list = [
            transforms.ToTensor(),
        ]
        if use_normalize:
            normalize = transforms.Normalize(self.CIFAR10_MEAN, self.CIFAR10_STD)
            train_transform_list.append(normalize)
            val_transform_list.append(normalize)
        self.train_transform = transforms.Compose(train_transform_list)
        self.val_transform = transforms.Compose(val_transform_list)
        
        # 加载原始数据集：train/val 各建一份实例，分别绑定各自的 transform
        # 注意：random_split 返回的两个 Subset 持有同一个底层 dataset，
        # 若在其上互相赋值 transform 会彼此覆盖，导致训练集增强失效
        self.full_train_ds = CIFAR10(root=self.root, train=True, download=download, transform=self.train_transform)
        full_val_ds = CIFAR10(root=self.root, train=True, download=False, transform=self.val_transform)
        self.test_ds = CIFAR10(root=self.root, train=False, download=download, transform=self.val_transform)
        
        # 划分训练集和验证集
        if val_split > 0:
            total_count = len(self.full_train_ds)
            # float 按比例、int 按绝对数量
            val_count = int(total_count * val_split) if isinstance(val_split, float) else val_split
            train_count = total_count - val_count
            #【重要】只生成一次随机索引，两份实例共享同一划分，避免 train/val 样本重叠
            indices = torch.randperm(total_count, generator=self.generator).tolist()
            self.train_ds = Subset(self.full_train_ds, indices[:train_count])
            self.val_ds = Subset(full_val_ds, indices[train_count:])
        else:
            # 不划分验证集时用测试集充当；注意若据此做模型选择/早停，
            # 测试指标会虚高（信息泄漏），仅适合快速实验
            self.train_ds = self.full_train_ds
            self.val_ds = self.test_ds

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, 
                          batch_size=self.batch_size, 
                          shuffle=True, 
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          persistent_workers=self.num_workers > 0,
                          )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_ds, 
                          batch_size=self.batch_size, 
                          shuffle=False, 
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          persistent_workers=self.num_workers > 0,)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_ds, 
                          batch_size=self.batch_size, 
                          shuffle=False, 
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
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
    
    def plot_sample(self, loader: Optional[DataLoader] = None):
        """
        可视化样本，并正确处理反归一化
        """
        if loader is None:
            loader = self.test_dataloader()
        images, labels = next(iter(loader))

        # 创建网格图：按实际 batch 大小取列数；squeeze=False 保证 axes 恒为二维数组，
        # 避免 batch_size=1 时返回单个 Axes 导致遍历报错
        ncols = min(images.shape[0], 5)
        fig, axes = plt.subplots(1, ncols, figsize=(10, 2), squeeze=False)

        for i, ax in enumerate(axes[0]):
            img = images[i].permute(1, 2, 0)
            if self.use_normalize:
                # 反归一化公式：img * std + mean
                img = img * torch.tensor(self.CIFAR10_STD) + torch.tensor(self.CIFAR10_MEAN)
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
