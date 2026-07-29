import os
import sys
import numpy as np

import torch



def get_smart_num_workers():
    """获取推荐 num_workers，感知容器限制并适配平台。"""
    # 获取可用 CPU 核心数（容器友好）
    if hasattr(os, 'sched_getaffinity'):
        available = len(os.sched_getaffinity(0))
    else:
        available = os.cpu_count() or 1  # Windows/macOS 直接使用

    if sys.platform != 'linux':
        # spawn 平台：保守取值，最多 4 个 worker
        return max(1, min(available // 2, 4))
    else:
        # Linux (fork)：保留一个核心给主进程，无硬性上限（但可设 cap）
        # 如果你希望限制上限，可以用 min(available, 16) 等，这里建议不设限
        return max(1, available - 1)


def auto_pin_memory(device, num_workers=0):
    """智能判断是否使用 pin_memory，考虑平台和 num_workers。"""
    if not torch.cuda.is_available():
        return False

    dev = torch.device(device) if device is not None else torch.device('cuda')
    if dev.type != 'cuda':
        return False

    # Windows 上，若 num_workers > 0，pin_memory 易导致死锁或错误，强制禁用
    if sys.platform == 'win32' and num_workers > 0:
        return False

    # macOS 上，spawn 机制同样有风险，保守禁用（除非 num_workers=0）
    if sys.platform == 'darwin' and num_workers > 0:
        return False

    # Linux 且 num_workers >= 0 时均可启用（但 num_workers=0 时其实无加速效果，但仍可开）
    return True



################################# 数据处理函数 #################################


def decode_seg_map_sequence(label_masks, dataset='pascal'):
    rgb_masks = []
    for label_mask in label_masks:
        rgb_mask = decode_segmap(label_mask, dataset)
        rgb_masks.append(rgb_mask)
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
    return rgb_masks


def decode_segmap(label_mask, dataset):
    """
    Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        dataset (str): name of dataset
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    if dataset == 'pascal' or dataset == 'coco':
        n_classes = 21
        label_colours = get_pascal_labels()
    elif dataset == 'cityscapes':
        n_classes = 19
        label_colours = get_cityscapes_labels()
    else:
        raise NotImplementedError

    r = np.zeros_like(label_mask)
    g = np.zeros_like(label_mask)
    b = np.zeros_like(label_mask)
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    
    return rgb


def encode_segmap(mask):
    """
    Encode segmentation label images as pascal classes
    Args:
        mask (np.ndarray): raw segmentation label image of dimension
          (M, N, 3), in which the Pascal classes are encoded as colours.
    Returns:
        (np.ndarray): class map with dimensions (M,N), where the value at
        a given location is the integer denoting the class index.
    """
    mask = mask.astype(np.uint8)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    for ii, label in enumerate(get_pascal_labels()):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(np.uint8)
    return label_mask


def get_cityscapes_labels():
    return np.array([
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32]], dtype=np.uint8)


def get_pascal_labels():
    """Load the mapping that associates pascal classes with label colors
    Returns:
        np.ndarray with dimensions (21, 3)
    """
    return np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                       [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                       [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                       [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                       [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                       [0, 64, 128]], dtype=np.uint8)
