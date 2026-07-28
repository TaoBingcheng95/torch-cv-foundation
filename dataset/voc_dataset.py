"""
VOC2012 数据集读取脚本（分割 / 检测 / 分类）

Pascal VOC (Visual Object Classes) 是经典的视觉识别基准数据集。
本脚本实现 VOC2012 三大视觉任务的 Dataset：
  1. 语义分割 (Segmentation) — 像素级类别标注
  2. 目标检测 (Detection)   — 边界框 + 类别标注
  3. 图像分类 (Classification) — 图像级多标签分类

数据目录结构::

    root/
    └── VOCdevkit/
        └── VOC2012/
            ├── JPEGImages/          # 原始 RGB 图像 (.jpg)
            ├── Annotations/         # XML 标注 (检测 / 分类)
            ├── SegmentationClass/   # 语义分割标注 (.png, 调色板模式)
            └── ImageSets/
                ├── Main/            # 检测 & 分类划分文件
                └── Segmentation/    # 分割划分文件
"""

import logging
import os
import xml.etree.ElementTree as ET
from typing import Optional, Union
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, Dataset, Subset


logger = logging.getLogger(__name__)


# ======================================================================
#  VOC 共享常量
# ======================================================================

VOC_CLASSES = (
    'background',
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor')

# 20 个前景类在 ImageSets/Main 下的文件名（无空格、无斜杠）
VOC_CLASS_NAMES = (
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')


VOC_COLORMAP = np.array([
    [0, 0, 0],       [128, 0, 0],   [0, 128, 0],   [128, 128, 0],
    [0, 0, 128],     [128, 0, 128], [0, 128, 128], [128, 128, 128],
    [64, 0, 0],      [192, 0, 0],   [64, 128, 0],  [192, 128, 0],
    [64, 0, 128],    [192, 0, 128], [64, 128, 128],[192, 128, 128],
    [0, 64, 0],      [128, 64, 0],  [0, 192, 0],   [128, 192, 0],
    [0, 64, 128] ], dtype=np.uint8)  # (21, 3)

# ImageNet RGB 均值/标准差（用于内置变换，与 torchvision 预训练权重的预处理一致）
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _read_id_list(filepath):
    """读取文本文件中的样本 ID 列表（每行一个 ID）"""
    ids = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(line)
    return ids


def detection_collate_fn(batch):
    """检测任务专用 collate_fn（torchvision 检测参考实现同款写法）

    检测样本每张图的框数 N 不同，default_collate 无法堆叠成规则张量，
    因此保持样本独立，将 batch 转置为 ``(images, targets)`` 两个元组：

        DataLoader(ds, batch_size=4, collate_fn=detection_collate_fn)
        images, targets = next(iter(loader))
        # images:  元组，长度 batch_size，各元素尺寸可不同
        # targets: 元组，元素为 target dict
    """
    return tuple(zip(*batch))


# ======================================================================
#  基类
# ======================================================================

class _VOCBase(Dataset):
    """
    VOC 数据集公共基类

    负责数据集目录校验、图像 ID 列表读取、图像加载等公共逻辑。
    子类需实现 ``__getitem__``。

    Args:
        root (str): 数据集根目录，需包含 ``VOCdevkit/VOC2012`` 子目录
        split (str): 数据划分，可选 ``'train'`` / ``'val'`` / ``'trainval'``
        task (str): 子任务名，用于定位 ImageSets 子目录
            （``'Segmentation'`` / ``'Main'``）
    """

    NUM_CLASSES = 21
    # 与 torchvision / 本仓库 loss、metrics、trainer 统一使用 255 作为忽略标签
    IGNORE_INDEX = 255
    VOC_CLASSES = VOC_CLASSES
    VOC_CLASS_NAMES = VOC_CLASS_NAMES
    VOC_COLORMAP = VOC_COLORMAP
    IMAGENET_MEAN = IMAGENET_MEAN
    IMAGENET_STD = IMAGENET_STD

    def __init__(self, root, split='train', task='Main'):
        super().__init__()
        assert split in ('train', 'val', 'trainval'), \
            f"split 必须为 'train'/'val'/'trainval'，收到: '{split}'"
        self.root = root
        self.split = split

        self.dataset_dir = os.path.join(root, 'VOCdevkit', 'VOC2012')
        if not os.path.isdir(self.dataset_dir):
            raise FileNotFoundError(
                f"数据集目录不存在: {self.dataset_dir}\n"
                f"请下载 VOC2012 并解压到 '{root}' 目录下")

        imgsets_file = os.path.join(
            self.dataset_dir, 'ImageSets', task, f'{split}.txt')
        if not os.path.isfile(imgsets_file):
            raise FileNotFoundError(f"划分文件不存在: {imgsets_file}")

        self.ids = _read_id_list(imgsets_file)
        # ids/images/annotations 三个列表始终保持下标一一对应，
        # 过滤样本时必须通过 _filter_samples 同步进行
        ids, images, annotations = [], [], []
        for sid in self.ids:
            img_path = os.path.join(self.dataset_dir, 'JPEGImages', f'{sid}.jpg')
            ann_path = os.path.join(self.dataset_dir, 'Annotations', f'{sid}.xml')
            if not os.path.isfile(img_path):
                logger.warning("图像缺失，已跳过: %s", img_path)
                continue
            ids.append(sid)
            images.append(img_path)
            annotations.append(ann_path)
        self.ids = ids
        self.images = images
        self.annotations = annotations

        logger.info("%s [%s] 加载完成: %d 个样本",
                    self.__class__.__name__, split, len(self.images))

    def _filter_samples(self, keep_indices):
        """按下标同步过滤 ids/images/annotations，保证三个列表始终一一对应

        子类需要剔除样本时（如分割标注缺失、XML 缺失）必须调用本方法，
        禁止只替换其中某一个列表，否则列表间下标错位会造成图像与标注错配。
        """
        keep = set(keep_indices)
        self.ids = [x for i, x in enumerate(self.ids) if i in keep]
        self.images = [x for i, x in enumerate(self.images) if i in keep]
        self.annotations = [x for i, x in enumerate(self.annotations) if i in keep]

    def __len__(self):
        return len(self.images)

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"root='{self.root}', split='{self.split}', "
                f"samples={len(self)})")


# ======================================================================
#  语义分割
# ======================================================================

class VOC2012ClassSeg(_VOCBase):
    """
    Pascal VOC 2012 语义分割数据集

    继承 :class:`_VOCBase`，固定使用 VOC2012 版本。
    官方数据: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/

    训练集 1464 张，验证集 1449 张，共 21 个类别（含背景）。

    读取 ``ImageSets/Segmentation/{split}.txt`` 中列出的样本，
    返回 (image, label) 对。标注中像素值含义：
      - 0~20 : 21 个语义类别（含 background）
      - 255  : 边界 / 忽略区域，保留原值（配合 ``ignore_index=255`` 使用，
               与 torchvision 及本仓库 loss/metrics 的约定一致）

    Args:
        root (str): 数据集根目录，需包含 ``VOCdevkit/VOC2012`` 子目录
        split (str): 数据划分，可选 ``'train'`` / ``'val'`` / ``'trainval'``
        transform (callable | bool): 数据变换。
            - ``False``（默认）: 返回原始 numpy 数组 (H,W,3) uint8 + (H,W) int32
            - ``True``: 使用内置 ImageNet 归一化变换，返回 Tensor
            - callable: 自定义变换函数，接收 (img_np, lbl_np) 返回 (img, lbl)
    """
    url = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'

    def __init__(self, root, split='train', transform=False):
        super().__init__(root, split=split, task='Segmentation')
        self._transform = transform
        # 校验分割标注文件是否存在；通过 _filter_samples 同步剔除，
        # 保证 ids/images/annotations/labels 四个列表下标对应同一样本
        keep_indices, valid_labels = [], []
        for i, sid in enumerate(self.ids):
            lbl_path = os.path.join(
                self.dataset_dir, 'SegmentationClass', f'{sid}.png')
            if not os.path.isfile(lbl_path):
                logger.warning("分割标注缺失，已跳过: %s", lbl_path)
                continue
            keep_indices.append(i)
            valid_labels.append(lbl_path)
        self._filter_samples(keep_indices)
        self.labels = valid_labels

    def __getitem__(self, index):
        """读取第 index 个样本

        Returns:
            当 transform=False 时:
                img (np.ndarray): (H, W, 3) uint8, RGB 顺序
                lbl (np.ndarray): (H, W) int32, 值域 [0, 20] ∪ {255}
            当 transform=True 时:
                img (Tensor): (3, H, W) float32, ImageNet 归一化后
                lbl (Tensor): (H, W) int64
            当 transform=callable 时:
                由自定义变换决定返回格式
        """
        # 读取图像 (RGB, uint8)
        img = np.array(Image.open(self.images[index]).convert('RGB'), dtype=np.uint8)
        # 读取标注 (调色板 PNG → 单通道类别索引)；
        # 255 即边界/忽略像素，与 IGNORE_INDEX 一致，无需重映射
        lbl = np.array(Image.open(self.labels[index]), dtype=np.int32)

        if callable(self._transform):
            return self._transform(img, lbl)
        elif self._transform:
            return self._default_transform(img, lbl)
        else:
            return img, lbl

    # ------------------------------------------------------------------
    #  内置变换 & 反变换
    # ------------------------------------------------------------------
    def _default_transform(self, img, lbl):
        """内置变换: uint8→float32/255, ImageNet 归一化, HWC→CHW, 转 Tensor"""
        img = img.astype(np.float32) / 255.0
        img = (img - self.IMAGENET_MEAN) / self.IMAGENET_STD
        img = img.transpose(2, 0, 1)               # HWC -> CHW
        img = torch.from_numpy(img.copy()).float()
        lbl = torch.from_numpy(lbl.copy()).long()
        return img, lbl

    def untransform(self, img, lbl):
        """
        反变换: 将 Tensor 还原为可显示的 RGB numpy 图像

        Args:
            img (Tensor): (3, H, W) 经 _default_transform 后的图像
            lbl (Tensor): (H, W) 标签

        Returns:
            img (np.ndarray): (H, W, 3) uint8, RGB 顺序
            lbl (np.ndarray): (H, W) int64
        """
        img = img.numpy().transpose(1, 2, 0)  # CHW -> HWC
        # 反归一化公式：img * std + mean，再还原到 [0, 255]
        img = (img * self.IMAGENET_STD + self.IMAGENET_MEAN) * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
        lbl = lbl.numpy()
        return img, lbl

    # ------------------------------------------------------------------
    #  可视化工具
    # ------------------------------------------------------------------
    def label2color(self, label):
        """将类别索引图转换为 VOC 彩色可视化图

        Args:
            label (np.ndarray): (H, W) 整数标签，值域 [0, 20]，255（忽略区）显示为黑色

        Returns:
            color (np.ndarray): (H, W, 3) uint8, RGB 彩色图
        """
        label = np.asarray(label, dtype=np.int64)
        h, w = label.shape
        color = np.zeros((h, w, 3), dtype=np.uint8)
        for cls_id in range(self.NUM_CLASSES):
            color[label == cls_id] = self.VOC_COLORMAP[cls_id]
        return color

    def plot_sample(self, index=0):
        """可视化第 index 个样本：原图 + 彩色分割掩码

        直接从磁盘读取原始图像与标注（不经过 transform）；
        绘制完成后立即关闭 figure，避免多窗口堆积。
        """
        img = np.array(Image.open(self.images[index]).convert('RGB'), dtype=np.uint8)
        lbl = np.array(Image.open(self.labels[index]), dtype=np.int32)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(img)
        axes[0].set_title('Image')
        axes[0].axis('off')
        axes[1].imshow(self.label2color(lbl))
        axes[1].set_title('Segmentation Mask')
        axes[1].axis('off')
        plt.tight_layout()
        plt.show()
        plt.close()


# ======================================================================
#  语义分割 DataLoader
# ======================================================================

class SegJointTransform:
    """分割任务联合变换：图像与掩码同步 Resize / 翻转，再归一化转 Tensor

    定义为顶层类（而非闭包/lambda），保证在 macOS/Windows 的 spawn
    多进程模式下可被 pickle，num_workers > 0 时不报错。

    Args:
        img_size (int): 输出的正方形边长。VOC 图像尺寸不一，
            必须统一尺寸后 default_collate 才能堆叠成规则 batch
        hflip (bool): 是否启用 0.5 概率随机水平翻转（训练集增强）
    """

    def __init__(self, img_size: int = 320, hflip: bool = False):
        self.img_size = img_size
        self.hflip = hflip

    def __call__(self, img, lbl):
        """接收 VOC2012ClassSeg 输出的 numpy 数组，返回 Tensor 对

        Args:
            img (np.ndarray): (H, W, 3) uint8, RGB
            lbl (np.ndarray): (H, W) int32, 值域 [0, 20] ∪ {255}

        Returns:
            img (Tensor): (3, img_size, img_size) float32, ImageNet 归一化
            lbl (Tensor): (img_size, img_size) int64
        """
        img = Image.fromarray(img)
        # 标签值域 [0,20]∪{255} 在 uint8 范围内，可安全转换
        lbl = Image.fromarray(lbl.astype(np.uint8))
        # 图像用双线性；掩码必须用最近邻，避免插值产生 3、127 等非法类别值
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        lbl = lbl.resize((self.img_size, self.img_size), Image.NEAREST)

        if self.hflip and torch.rand(1).item() < 0.5:
            # 几何变换必须图像与掩码同步，否则像素级标注错位
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            lbl = lbl.transpose(Image.FLIP_LEFT_RIGHT)

        img = np.asarray(img, dtype=np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = torch.from_numpy(img.transpose(2, 0, 1).copy()).float()  # HWC -> CHW
        lbl = torch.from_numpy(np.asarray(lbl, dtype=np.int64))
        return img, lbl


class VOC2012ClassSegLoader:
    """VOC2012 语义分割 DataLoader 封装（设计理念同 MNISTDataLoader）

    划分策略：VOC2012 官方 test 集不公开标注，因此
      - 官方 ``val`` 集（1449 张）保持不动充当 test 集，保证指标与外部基准可比
      - 从官方 ``train`` 集（1464 张）按 val_split 切分出验证集

    参考 CIFAR10DataLoader 的做法：train/val 各建一份 dataset 实例，
    分别绑定增强/评估 transform，并共享同一份随机索引避免样本重叠。
    """
    NUM_CLASSES = 21
    IGNORE_INDEX = 255
    DEFAULT_TRAIN_LENGTH = 1464   # 官方 train 划分
    DEFAULT_TEST_LENGTH = 1449    # 官方 val 划分（此处用作 test）

    def __init__(self,
                 root: str = './data',
                 # 验证集划分：float ∈ (0,1) 按比例从训练集切分，int ≥ 1 按绝对数量切分；
                 # 官方 val 集保持不动（充当 test），保证指标与外部基准可比
                 val_split: Union[int, float] = 0.1,
                 batch_size: int = 8,
                 img_size: int = 320,        # 统一尺寸，否则变尺寸图像无法堆叠成 batch
                 use_augment: bool = True,   # 训练集是否启用随机水平翻转增强
                 # 仅 CUDA 设备受益（锁页内存加速 Host→GPU 拷贝），CPU/MPS 无效且会告警；
                 # 默认关闭，由调用方按设备显式开启（参考 auto_pin_memory）
                 pin_memory: bool = False,
                 num_workers: int = 0,
                 seed: int = 42,             # 固定随机种子
                 ) -> None:
        super().__init__()

        self.root = root
        self.batch_size = batch_size
        self.img_size = img_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        #【重要】固定随机种子，保证每次运行划分一致
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

        self.train_transform = SegJointTransform(img_size, hflip=use_augment)
        self.eval_transform = SegJointTransform(img_size, hflip=False)

        # train/val 各建一份实例，分别绑定各自的 transform；
        # 若共用一份实例，Subset 上互相赋值 transform 会彼此覆盖，导致训练集增强失效
        self.full_train_ds = VOC2012ClassSeg(
            root, split='train', transform=self.train_transform)
        full_val_ds = VOC2012ClassSeg(
            root, split='train', transform=self.eval_transform)
        self.test_ds = VOC2012ClassSeg(
            root, split='val', transform=self.eval_transform)

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
        return DataLoader(
            dataset=self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,                      # 训练集必须 shuffle
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            # 跨 epoch 复用 worker，避免 spawn 平台（macOS/Windows）每轮重建开销；
            # num_workers=0 时必须为 False，否则 DataLoader 报错
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,      # 验证集不能 shuffle
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,      # 测试集不能 shuffle
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    @property
    def num_classes(self) -> int:
        """
        Get the number of classes.

        :return: The number of VOC segmentation classes (21, 含 background).
        """
        return self.NUM_CLASSES

    @property
    def classes(self) -> list[str]:
        return list(VOC_CLASSES)

    @property
    def class_to_idx(self) -> dict[str, int]:
        return {name: idx for idx, name in enumerate(VOC_CLASSES)}

    @property
    def idx_to_class(self) -> dict[int, str]:
        return {value: key for key, value in self.class_to_idx.items()}

    def plot_sample(self, loader: Optional[DataLoader] = None):
        """
        可视化一个 batch 中的数据：上排原图（反归一化），下排彩色分割掩码
        """
        if loader is None:
            loader = self.test_dataloader()

        images, labels = next(iter(loader))

        # 创建网格图：按实际 batch 大小取列数；squeeze=False 保证 axes 恒为二维数组，
        # 避免 batch_size=1 时返回单个 Axes 导致遍历报错
        ncols = min(images.shape[0], 5)
        fig, axes = plt.subplots(2, ncols, figsize=(2.4 * ncols, 5), squeeze=False)

        for i in range(ncols):
            # 反归一化公式：img * std + mean，并截断到 [0,1] 便于显示
            img = images[i].numpy().transpose(1, 2, 0)  # CHW -> HWC
            img = np.clip(img * IMAGENET_STD + IMAGENET_MEAN, 0, 1)
            axes[0][i].imshow(img)
            axes[0][i].set_title('Image')
            axes[0][i].axis('off')
            axes[1][i].imshow(self.test_ds.label2color(labels[i].numpy()))
            axes[1][i].set_title('Mask')
            axes[1][i].axis('off')

        plt.tight_layout()
        plt.show()
        plt.close()


# ======================================================================
#  目标检测
# ======================================================================

class VOC2012Detection(_VOCBase):
    """
    Pascal VOC 2012 目标检测数据集

    读取 ``ImageSets/Main/{split}.txt`` 中的样本列表，
    从 ``Annotations/{id}.xml`` 解析边界框与类别。

    返回格式对齐 torchvision 检测模型约定，每个样本返回 ``(image, target)``：
      - image  : (H, W, 3) uint8 RGB numpy 数组（transform 可将其转为 Tensor）
      - target : dict，字段与 torchvision 检测参考实现一致：
          * boxes    (Tensor[N, 4] float32): [xmin, ymin, xmax, ymax]
          * labels   (Tensor[N] int64): 类别索引 1~20（0=background 不出现）
          * image_id (Tensor[1] int64): 样本在数据集中的下标
          * area     (Tensor[N] float32): 框面积，COCO 评测时用于尺寸分档
          * iscrowd  (Tensor[N] int64): 由 difficult 标记映射，评测时被忽略

    检测样本的框数 N 不定，默认的 default_collate 无法堆叠，
    搭配 DataLoader 时需传入本模块的 :func:`detection_collate_fn`。

    Args:
        root (str): 数据集根目录
        split (str): ``'train'`` / ``'val'`` / ``'trainval'``
        keep_difficult (bool): 是否保留 ``difficult=1`` 的目标（默认 False）
        transform (callable | None): 变换函数，接收 ``(image, target)``
            返回变换后的 ``(image, target)``（torchvision 检测变换约定，
            几何变换需同步更新 boxes）
    """

    def __init__(self, root, split='train', keep_difficult=False, transform=None):
        super().__init__(root, split=split, task='Main')
        self.keep_difficult = keep_difficult
        self.transform = transform

        # 剔除 XML 缺失的样本（同步过滤，避免预解析时直接报错）
        keep_indices = []
        for i, ann_path in enumerate(self.annotations):
            if not os.path.isfile(ann_path):
                logger.warning("XML 标注缺失，已跳过: %s", ann_path)
                continue
            keep_indices.append(i)
        if len(keep_indices) < len(self.annotations):
            self._filter_samples(keep_indices)

        # 预解析所有 XML 标注
        self._targets = []
        for ann_path in self.annotations:
            target = self._parse_annotation(ann_path)
            if not self.keep_difficult and target['difficult'].size > 0:
                keep = ~target['difficult'].astype(bool)
                target['boxes'] = target['boxes'][keep]
                target['labels'] = target['labels'][keep]
                target['difficult'] = target['difficult'][keep]
            self._targets.append(target)

        n_objects = sum(t['labels'].size for t in self._targets)
        logger.info("  共 %d 个目标框（keep_difficult=%s）", n_objects, keep_difficult)

    def _parse_annotation(self, ann_path):
        """
        解析单个 XML 标注文件

        Returns:
            dict: {
                'size'      : (width, height),
                'boxes'     : (N, 4) float32, [xmin, ymin, xmax, ymax],
                'labels'    : (N,) int64, 类别索引 1~20,
                'difficult' : (N,) bool,
                'truncated' : (N,) bool,
            }
        """
        tree = ET.parse(ann_path)
        root = tree.getroot()

        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        boxes, labels, difficults, truncateds = [], [], [], []
        for obj in root.findall('object'):
            name = obj.findtext('name', default='')
            # XML 中的类名是无空格/斜杠形式（如 'pottedplant'、'tvmonitor'），
            # 必须用 VOC_CLASS_NAMES 匹配；VOC_CLASSES 是展示名（'potted plant' 等），
            # 若误用会静默丢弃这两个类的全部标注框
            if name not in self.VOC_CLASS_NAMES:
                logger.warning("未知类别 '%s'，已跳过: %s", name, ann_path)
                continue
            cls_id = self.VOC_CLASS_NAMES.index(name) + 1  # 1~20（0 为 background）
            # difficult/truncated 节点在部分增广数据（如 SBD）中可能缺失，默认按 0 处理
            difficult = int(obj.findtext('difficult', default='0'))
            truncated = int(obj.findtext('truncated', default='0'))
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(cls_id)
            difficults.append(difficult)
            truncateds.append(truncated)

        return {
            'size': (width, height),
            'boxes': np.array(boxes, dtype=np.float32).reshape(-1, 4),
            'labels': np.array(labels, dtype=np.int64),
            'difficult': np.array(difficults, dtype=bool),
            'truncated': np.array(truncateds, dtype=bool),
        }

    def __getitem__(self, index):
        """读取第 index 个样本（torchvision 检测约定格式）

        Returns:
            img (np.ndarray): (H, W, 3) uint8, RGB（经 transform 后由其决定）
            target (dict): boxes / labels / image_id / area / iscrowd，
                均为 Tensor，字段含义见类 docstring
        """
        img = np.array(
            Image.open(self.images[index]).convert('RGB'), dtype=np.uint8)
        t = self._targets[index]
        boxes = torch.from_numpy(t['boxes'].copy())            # (N, 4) float32
        labels = torch.from_numpy(t['labels'].copy())          # (N,) int64
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([index], dtype=torch.int64),
            'area': (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            'iscrowd': torch.from_numpy(t['difficult'].copy().astype(np.int64)),
        }

        if self.transform is not None:
            img, target = self.transform(img, target)
        return img, target

    # ------------------------------------------------------------------
    #  可视化工具
    # ------------------------------------------------------------------
    def draw_boxes(self, img, boxes, labels):
        """在图像上绘制边界框与类别标签

        Args:
            img (np.ndarray): (H, W, 3) uint8, RGB
            boxes (np.ndarray): (N, 4), [xmin, ymin, xmax, ymax]
            labels (np.ndarray): (N,), 类别索引

        Returns:
            fig (matplotlib.figure.Figure): 绘制了检测框的图像
        """
        fig, ax = plt.subplots(1, figsize=(10, 8))
        ax.imshow(img)
        for box, lbl in zip(boxes, labels):
            xmin, ymin, xmax, ymax = box
            color = self.VOC_COLORMAP[lbl] / 255.0
            rect = plt.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin,
                linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            ax.text(xmin, ymin - 4, self.VOC_CLASSES[lbl],
                    color='white', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', fc=color, alpha=0.8))
        ax.set_title(f'Detection ({len(boxes)} objects)')
        ax.axis('off')
        return fig

    def plot_sample(self, index=0):
        """可视化第 index 个样本的检测框

        直接从磁盘读取原始图像（不经过 transform），配合预解析的标注绘制；
        绘制完成后立即关闭 figure，避免多窗口堆积。
        """
        img = np.array(Image.open(self.images[index]).convert('RGB'), dtype=np.uint8)
        target = self._targets[index]
        fig = self.draw_boxes(img, target['boxes'], target['labels'])
        plt.tight_layout()
        plt.show()
        plt.close()


# ======================================================================
#  图像分类（多标签）
# ======================================================================

class VOC2012Classification(_VOCBase):
    """
    Pascal VOC 2012 图像分类数据集（多标签）

    读取 ``ImageSets/Main/{split}.txt`` 中的样本列表，
    从各类别文件 ``ImageSets/Main/{class}_{split}.txt`` 解析图像级标签。

    每个类别文件每行格式::

        image_id  label    # +1=存在  -1=不存在  0=困难样本

    每个样本返回 ``(image, labels, difficult)``：
      - image     : (H, W, 3) uint8, RGB
      - labels    : (20,) int8, 每个前景类的标签 (+1 / -1)
      - difficult : (20,) bool, 标记哪些类别为困难样本 (原始值=0)

    Args:
        root (str): 数据集根目录
        split (str): ``'train'`` / ``'val'`` / ``'trainval'``
        transform (callable | None): 图像变换，接收 np.ndarray 返回变换后的图像
    """

    NUM_CLASSES = 20  # 分类任务仅关注 20 个前景类

    def __init__(self, root, split='train', transform=None):
        super().__init__(root, split=split, task='Main')
        self.transform = transform

        # 建立 image_id → 数组下标 的映射
        id2idx = {}
        for idx, img_path in enumerate(self.images):
            sid = os.path.splitext(os.path.basename(img_path))[0]
            id2idx[sid] = idx

        n = len(self.images)
        self.labels = -np.ones((n, 20), dtype=np.int8)
        self.difficult = np.zeros((n, 20), dtype=bool)

        # 逐类别读取标签文件
        main_dir = os.path.join(self.dataset_dir, 'ImageSets', 'Main')
        for cls_idx, cls_name in enumerate(VOC_CLASS_NAMES):
            cls_file = os.path.join(main_dir, f'{cls_name}_{split}.txt')
            if not os.path.isfile(cls_file):
                logger.warning("类别文件缺失: %s", cls_file)
                continue
            with open(cls_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 2:
                        continue
                    sid, flag = parts[0], int(parts[1])
                    if sid not in id2idx:
                        continue
                    idx = id2idx[sid]
                    if flag == 0:
                        self.difficult[idx, cls_idx] = True
                    else:
                        self.labels[idx, cls_idx] = flag

        n_pos = (self.labels == 1).sum()
        n_diff = self.difficult.sum()
        logger.info("  正标签: %d, 困难标签: %d", n_pos, n_diff)

    def __getitem__(self, index):
        """读取第 index 个样本

        Returns:
            img (np.ndarray): (H, W, 3) uint8, RGB
            labels (np.ndarray): (20,) int8, +1=存在 / -1=不存在
            difficult (np.ndarray): (20,) bool, 困难样本标记
        """
        img = np.array(
            Image.open(self.images[index]).convert('RGB'), dtype=np.uint8)
        labels = self.labels[index].copy()
        difficult = self.difficult[index].copy()

        if self.transform is not None:
            img = self.transform(img)
        return img, labels, difficult

    def get_positive_classes(self, index):
        """返回第 index 个样本中存在的类别名称列表"""
        labels = self.labels[index]
        return [VOC_CLASSES[i + 1] for i in range(20) if labels[i] == 1]

    def plot_sample(self, index=0):
        """可视化第 index 个样本及其正类别标签

        直接从磁盘读取原始图像（不经过 transform）；
        绘制完成后立即关闭 figure，避免多窗口堆积。
        """
        img = np.array(Image.open(self.images[index]).convert('RGB'), dtype=np.uint8)
        pos_classes = self.get_positive_classes(index)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(img)
        ax.set_title(f"Classification: {', '.join(pos_classes) or 'none'}")
        ax.axis('off')
        plt.tight_layout()
        plt.show()
        plt.close()



if __name__ == '__main__':
    # 库模块本身不配置 handler，仅在直接运行时打开 INFO 日志
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')

    # ---- 1. 语义分割 ----
    print("=" * 60)
    print("语义分割 (Segmentation)")
    print("=" * 60)
    seg_dataset = VOC2012ClassSeg(root='data', split='train')
    print(seg_dataset)
    print(f"类别数: {seg_dataset.NUM_CLASSES}")

    image, label = seg_dataset[0]
    print(f"image: {image.shape}, dtype={image.dtype}")
    print(f"label: {label.shape}, dtype={label.dtype}")
    print(f"label 唯一值: {np.unique(label)}")
    seg_dataset.plot_sample(0)

    # ---- 2. 目标检测 ----
    print("\n" + "=" * 60)
    print("目标检测 (Detection)")
    print("=" * 60)
    det_dataset = VOC2012Detection(root='data', split='train')
    print(det_dataset)

    image, target = det_dataset[0]
    print(f"image: {image.shape}")
    print(f"target keys: {sorted(target.keys())}")
    print(f"boxes: {tuple(target['boxes'].shape)}, labels: {tuple(target['labels'].shape)}")
    for box, lbl in zip(target['boxes'], target['labels']):
        print(f"  {VOC_CLASSES[lbl]:>15s}  "
              f"[{box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f}]")

    # collate_fn 演示：框数不定的样本保持独立，不强行堆叠
    from torch.utils.data import DataLoader
    loader = DataLoader(det_dataset, batch_size=4, shuffle=False,
                        collate_fn=detection_collate_fn)
    images, targets = next(iter(loader))
    print(f"\nDataLoader batch: {len(images)} 张图，"
          f"各图框数 = {[len(t['boxes']) for t in targets]}")
    det_dataset.plot_sample(0)

    # ---- 3. 图像分类 ----
    print("\n" + "=" * 60)
    print("图像分类 (Classification)")
    print("=" * 60)
    cls_dataset = VOC2012Classification(root='data', split='train')
    print(cls_dataset)

    image, labels, difficult = cls_dataset[0]
    print(f"image: {image.shape}")
    print(f"labels: {labels.shape}, dtype={labels.dtype}")
    pos_classes = cls_dataset.get_positive_classes(0)
    print(f"正类别: {pos_classes}")
    cls_dataset.plot_sample(0)

    # ---- 4. 分割 DataLoader ----
    print("\n" + "=" * 60)
    print("分割 DataLoader (VOC2012ClassSegLoader)")
    print("=" * 60)
    data_module = VOC2012ClassSegLoader(root='data',
                                        batch_size=4,
                                        val_split=0.1,
                                        img_size=320)
    print(f"Train size: {len(data_module.train_ds)}")
    print(f"Val size: {len(data_module.val_ds)}")
    print(f"Test size: {len(data_module.test_ds)}")

    # 查看一个 batch 的形状
    train_loader = data_module.train_dataloader()
    imgs, lbls = next(iter(train_loader))
    print(f"Batch shape: {imgs.shape}, Labels shape: {lbls.shape}")
    print(f"label 唯一值: {torch.unique(lbls).tolist()}")

    data_module.plot_sample()
