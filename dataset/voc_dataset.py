import matplotlib.pyplot as plt
from typing import Optional, Union
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split
import torchvision.transforms as transforms
from torchvision.datasets import VOCSegmentation, VOCDetection


VOC_COLORMAP = np.array([
    [0, 0, 0],       [128, 0, 0],   [0, 128, 0],   [128, 128, 0],
    [0, 0, 128],     [128, 0, 128], [0, 128, 128], [128, 128, 128],
    [64, 0, 0],      [192, 0, 0],   [64, 128, 0],  [192, 128, 0],
    [64, 0, 128],    [192, 0, 128], [64, 128, 128],[192, 128, 128],
    [0, 64, 0],      [128, 64, 0],  [0, 192, 0],   [128, 192, 0],
    [0, 64, 128] ], dtype=np.uint8)  # (21, 3)


VOC_CLASSES = (
    'background',
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor')


# 20 个前景类在 ImageSets/Main 下的文件名（无空格、无斜杠）
VOC_CLASS_NAMES = (
    'background',
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')



def label_to_color(label, colormap=VOC_COLORMAP, ignore_value=255):
    """
    将单通道标签图转为彩色RGB图（用于可视化）。
    
    Args:
        label (np.ndarray or torch.Tensor): 形状 (H, W)，值为 0~20 或 255。
        colormap (np.ndarray): (21, 3) 的 uint8 颜色表。
        ignore_value (int): 忽略区域的像素值（通常为255），会显示为黑色。
    
    Returns:
        np.ndarray: (H, W, 3) uint8 RGB 彩色图。
    """
    if hasattr(label, 'numpy'):   # 处理 torch.Tensor
        label = label.numpy()
    label = np.asarray(label, dtype=np.int64)
    
    h, w = label.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 对每个类别做映射（循环仅 21 次，开销很小）
    for cls_id in range(len(colormap)):
        color[label == cls_id] = colormap[cls_id]
    
    # 处理忽略区域（255）——这里设为黑色，你也可以设为白色或半透明
    color[label == ignore_value] = [0, 0, 0]
    return color



def label_to_color_vectorized(label, ignore_color=(0, 0, 0)):
    """
    使用广播/索引将标签图映射为彩色图（完全向量化，无循环）。
    
    Args:
        label (np.ndarray or torch.Tensor): 形状 (H, W)，值为 0~20 或 255。
        ignore_color (tuple): 忽略区域(255)渲染的颜色，默认为黑色。
    
    Returns:
        np.ndarray: (H, W, 3) uint8 RGB 彩色图。
    """
    # 统一转为 numpy 整数数组
    if hasattr(label, 'numpy'):
        label = label.numpy()
    label = np.asarray(label, dtype=np.int64)
    
    # 构建 256 行的完整查找表 (LUT)
    # 第 0~20 行存 VOC 颜色，第 255 行存忽略色，其余行默认为黑色 (0,0,0)
    lut = np.zeros((256, 3), dtype=np.uint8)
    lut[:21] = VOC_COLORMAP          # 映射 0~20 类别
    lut[255] = ignore_color          # 映射 255 忽略区域
    
    # 🎯 核心：一步索引（NumPy 高级广播），无 Python 循环！
    return lut[label]



class VOCSegmentationDataLoader:
    # VOCSegmentation 的均值和标准差
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    RESIZE_SIZE = 500
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
        # 图像：用 ToTensor (除以255，转为FloatTensor)
        # 标签：用 PILToTensor (不除以255，转为Int64/LongTensor，保留原始索引)
        self.transform_img = transforms.Compose([
            transforms.Resize((self.RESIZE_SIZE,self.RESIZE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(self.IMAGENET_MEAN, self.IMAGENET_STD)
        ])

        # 注意：PILToTensor 返回的是 (H, W) 的 LongTensor，值域保持 0~20 和 255
        self.transform_target = transforms.Compose([
            transforms.Resize((self.RESIZE_SIZE,self.RESIZE_SIZE)),
            transforms.PILToTensor()
        ])

        self.data_test = VOCSegmentation(
            root=root,
            year = '2012',
            image_set='val',
            download=download,
            transform=self.transform_img,
            target_transform=self.transform_target
        )
        self.full_data_train = VOCSegmentation(
            root=root, 
            year = '2012',
            image_set='train',
            download=download,
            transform=self.transform_img,
            target_transform=self.transform_target
        )

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
        return 21

    @property
    def classes(self) -> list[str]:
        return VOC_CLASS_NAMES  # type: ignore


    def plot_sample(self, loader: Optional[DataLoader] = None):
        """
        可视化一个 batch 中的数据
        """
        if loader is None:
            loader = self.test_dataloader()
        
        images, labels = next(iter(loader))
        
        # 创建网格图：按实际 batch 大小取列数；squeeze=False 保证 axes 恒为二维数组，
        # 避免 batch_size=1 时返回单个 Axes 导致遍历报错
        ncols = min(images.shape[0], 2)
        fig, axes = plt.subplots(2, ncols, figsize=(8, 8), squeeze=False) # 
        std = torch.Tensor(self.IMAGENET_STD)
        mean = torch.Tensor(self.IMAGENET_MEAN)
        
        for idx in range(ncols):
            img = images[idx]
            label = labels[idx]
            img = img.squeeze()
            img = img.permute((1,2,0))
            label = label.squeeze()
            label_color = label_to_color_vectorized(label)
            if self.use_normalize:
                # 反归一化公式：img * std + mean，并截断到 [0,1] 便于显示
                img = (img * std + mean)
                img = img.clip(0, 1) # 防止数值超出 [0,1] 范围
            axes[0,idx].imshow(img) 
            axes[1,idx].imshow(label_color)
            axes[0,idx].axis('off')
            axes[1,idx].axis('off')
            
        plt.tight_layout()
        plt.show()
        plt.close()



if __name__ == '__main__':
    dm = VOCSegmentationDataLoader(batch_size=4)
    dm.plot_sample()
