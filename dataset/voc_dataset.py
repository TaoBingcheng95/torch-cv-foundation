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

import os
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import torch
from torch.utils.data import Dataset


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

# ImageNet BGR 通道均值（用于内置变换）
MEAN_BGR = np.array([104.00698793, 116.66876762, 122.67891434])


def _read_id_list(filepath):
    """读取文本文件中的样本 ID 列表（每行一个 ID）"""
    ids = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(line)
    return ids


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
    IGNORE_INDEX = -1
    VOC_CLASSES = VOC_CLASSES
    VOC_COLORMAP = VOC_COLORMAP
    MEAN_BGR = MEAN_BGR

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
        self.images = []
        self.annotations = []
        for sid in self.ids:
            img_path = os.path.join(self.dataset_dir, 'JPEGImages', f'{sid}.jpg')
            ann_path = os.path.join(self.dataset_dir, 'Annotations', f'{sid}.xml')
            if not os.path.isfile(img_path):
                print(f"[WARNING] 图像缺失，已跳过: {img_path}")
                continue
            self.images.append(img_path)
            self.annotations.append(ann_path)

        print(f"{self.__class__.__name__} [{split}] 加载完成: {len(self.images)} 个样本")

    def __len__(self):
        return len(self.images)

    def _load_image_with_xml(self, index):
        """加载图像 (RGB uint8) 并解析对应的 XML 标注文件"""
        img = np.array(
            Image.open(self.images[index]).convert('RGB'), dtype=np.uint8)
        tree = ET.parse(self.annotations[index])
        return img, tree

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"root='{self.root}', split='{self.split}', "
                f"samples={len(self)})")


# ======================================================================
#  语义分割
# ======================================================================

class VOCClassSegBase(_VOCBase):
    """
    Pascal VOC 语义分割数据集基类

    读取 ``ImageSets/Segmentation/{split}.txt`` 中列出的样本，
    返回 (image, label) 对。标注中像素值含义：
      - 0~20 : 21 个语义类别（含 background）
      - 255  : 边界 / 忽略区域，映射为 -1（配合 ``ignore_index=-1`` 使用）

    Args:
        root (str): 数据集根目录，需包含 ``VOCdevkit/VOC2012`` 子目录
        split (str): 数据划分，可选 ``'train'`` / ``'val'`` / ``'trainval'``
        transform (callable | bool): 数据变换。
            - ``False``（默认）: 返回原始 numpy 数组 (H,W,3) uint8 + (H,W) int32
            - ``True``: 使用内置 BGR 均值减除变换，返回 Tensor
            - callable: 自定义变换函数，接收 (img_np, lbl_np) 返回 (img, lbl)
    """

    def __init__(self, root, split='train', transform=False):
        super().__init__(root, split=split, task='Segmentation')
        self._transform = transform
        # 校验分割标注文件是否存在
        valid_images, valid_labels = [], []
        for img_path in self.images:
            sid = os.path.splitext(os.path.basename(img_path))[0]
            lbl_path = os.path.join(
                self.dataset_dir, 'SegmentationClass', f'{sid}.png')
            if not os.path.isfile(lbl_path):
                print(f"[WARNING] 分割标注缺失，已跳过: {lbl_path}")
                continue
            valid_images.append(img_path)
            valid_labels.append(lbl_path)
        self.images = valid_images
        self.labels = valid_labels

    def __getitem__(self, index):
        """读取第 index 个样本

        Returns:
            当 transform=False 时:
                img (np.ndarray): (H, W, 3) uint8, RGB 顺序
                lbl (np.ndarray): (H, W) int32, 值域 [-1, 20]
            当 transform=True 时:
                img (Tensor): (3, H, W) float32, BGR 均值减除后
                lbl (Tensor): (H, W) int64
            当 transform=callable 时:
                由自定义变换决定返回格式
        """
        # 读取图像 (RGB, uint8)
        img = np.array(Image.open(self.images[index]).convert('RGB'), dtype=np.uint8)
        # 读取标注 (调色板 PNG → 单通道类别索引)
        lbl = np.array(Image.open(self.labels[index]), dtype=np.int32)
        # 255 为边界像素，映射为 IGNORE_INDEX
        lbl[lbl == 255] = self.IGNORE_INDEX

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
        """内置变换: RGB→BGR, 减均值, HWC→CHW, 转 Tensor"""
        img = img[:, :, ::-1].astype(np.float64)  # RGB -> BGR
        img -= self.MEAN_BGR
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
            lbl (np.ndarray): (H, W) int32
        """
        img = img.numpy().transpose(1, 2, 0)  # CHW -> HWC
        img = img + self.MEAN_BGR
        img = img[:, :, ::-1]                  # BGR -> RGB
        img = np.clip(img, 0, 255).astype(np.uint8)
        lbl = lbl.numpy()
        return img, lbl

    # ------------------------------------------------------------------
    #  可视化工具
    # ------------------------------------------------------------------
    def label2color(self, label):
        """将类别索引图转换为 VOC 彩色可视化图

        Args:
            label (np.ndarray): (H, W) 整数标签，值域 [0, 20]，-1 显示为黑色

        Returns:
            color (np.ndarray): (H, W, 3) uint8, RGB 彩色图
        """
        label = np.asarray(label, dtype=np.int64)
        h, w = label.shape
        color = np.zeros((h, w, 3), dtype=np.uint8)
        for cls_id in range(self.NUM_CLASSES):
            color[label == cls_id] = self.VOC_COLORMAP[cls_id]
        return color


class VOC2012ClassSeg(VOCClassSegBase):
    """Pascal VOC 2012 语义分割数据集

    继承 :class:`VOCClassSegBase`，固定使用 VOC2012 版本。
    官方数据: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/

    训练集 1464 张，验证集 1449 张，共 21 个类别（含背景）。
    """

    url = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'

    def __init__(self, root, split='train', transform=False):
        super().__init__(root, split=split, transform=transform)


# ======================================================================
#  目标检测
# ======================================================================

class VOC2012Detection(_VOCBase):
    """
    Pascal VOC 2012 目标检测数据集

    读取 ``ImageSets/Main/{split}.txt`` 中的样本列表，
    从 ``Annotations/{id}.xml`` 解析边界框与类别。

    每个样本返回 ``(image, boxes, labels)``：
      - image  : (H, W, 3) uint8, RGB
      - boxes  : (N, 4) float32, 格式 [xmin, ymin, xmax, ymax]
      - labels : (N,) int64, 类别索引 1~20（0=background 不会出现在框标注中）

    Args:
        root (str): 数据集根目录
        split (str): ``'train'`` / ``'val'`` / ``'trainval'``
        keep_difficult (bool): 是否保留 ``difficult=1`` 的目标（默认 False）
        transform (callable | None): 图像变换，接收 np.ndarray 返回变换后的图像
    """

    def __init__(self, root, split='train', keep_difficult=False, transform=None):
        super().__init__(root, split=split, task='Main')
        self.keep_difficult = keep_difficult
        self.transform = transform

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
        print(f"  共 {n_objects} 个目标框（keep_difficult={keep_difficult}）")

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
            name = obj.find('name').text
            if name not in self.VOC_CLASSES:
                continue
            cls_id = self.VOC_CLASSES.index(name)  # 1~20
            difficult = int(obj.find('difficult').text)
            truncated = int(obj.find('truncated').text)
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
        """读取第 index 个样本

        Returns:
            img (np.ndarray): (H, W, 3) uint8, RGB
            boxes (np.ndarray): (N, 4) float32, [xmin, ymin, xmax, ymax]
            labels (np.ndarray): (N,) int64, 类别索引 1~20
        """
        img = np.array(
            Image.open(self.images[index]).convert('RGB'), dtype=np.uint8)
        target = self._targets[index]
        boxes = target['boxes'].copy()
        labels = target['labels'].copy()

        if self.transform is not None:
            img = self.transform(img)
        return img, boxes, labels

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
                print(f"[WARNING] 类别文件缺失: {cls_file}")
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
        print(f"  正标签: {n_pos}, 困难标签: {n_diff}")

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


# ======================================================================
#  演示
# ======================================================================

if __name__ == '__main__':

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

    color_label = seg_dataset.label2color(label)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(image)
    axes[0].set_title('Image')
    axes[0].axis('off')
    axes[1].imshow(color_label)
    axes[1].set_title('Segmentation Mask')
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()

    # ---- 2. 目标检测 ----
    print("\n" + "=" * 60)
    print("目标检测 (Detection)")
    print("=" * 60)
    det_dataset = VOC2012Detection(root='data', split='train')
    print(det_dataset)

    image, boxes, labels = det_dataset[0]
    print(f"image: {image.shape}")
    print(f"boxes: {boxes.shape}, labels: {labels.shape}")
    for box, lbl in zip(boxes, labels):
        print(f"  {VOC_CLASSES[lbl]:>15s}  "
              f"[{box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f}]")

    det_dataset.draw_boxes(image, boxes, labels)
    plt.tight_layout()
    plt.show()

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

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(image)
    ax.set_title(f"Classification: {', '.join(pos_classes)}")
    ax.axis('off')
    plt.tight_layout()
    plt.show()
