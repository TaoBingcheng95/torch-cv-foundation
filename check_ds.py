import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.datasets import CIFAR10, FashionMNIST, MNIST
from torchvision.datasets import VOCSegmentation

from  torchvision import transforms
# from dataset import VOC2012ClassSegLoader
from dataset.torch_dataset import CIFAR10DataLoader, MNISTDataLoader

VOC_COLORMAP = np.array([
    [0, 0, 0],       [128, 0, 0],   [0, 128, 0],   [128, 128, 0],
    [0, 0, 128],     [128, 0, 128], [0, 128, 128], [128, 128, 128],
    [64, 0, 0],      [192, 0, 0],   [64, 128, 0],  [192, 128, 0],
    [64, 0, 128],    [192, 0, 128], [64, 128, 128],[192, 128, 128],
    [0, 64, 0],      [128, 64, 0],  [0, 192, 0],   [128, 192, 0],
    [0, 64, 128] ], dtype=np.uint8)  # (21, 3)



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


if __name__ == "__main__":
    # ds = CIFAR10DataLoader()
    # # print(ds.classes)
    # # print(ds.class_to_idx)
    # # print(ds.idx_to_class)
    # test_loader = ds.test_dataloader()
    # ds.plot_sample()

    # dm = VOC2012ClassSegLoader(root='data', batch_size=8, val_split=0.1, img_size=320)
    # train_loader = dm.train_dataloader()
    # x, y = next(iter(train_loader))
    # print(x.shape, y.shape)
    # dm.plot_sample()

    # 图像：用 ToTensor (除以255，转为FloatTensor)
    # 标签：用 PILToTensor (不除以255，转为Int64/LongTensor，保留原始索引)
    transform_img = transforms.Compose([
        transforms.Resize((500,500)),
        transforms.ToTensor()
    ])

    # 注意：PILToTensor 返回的是 (H, W) 的 LongTensor，值域保持 0~20 和 255
    transform_target = transforms.Compose([
        transforms.Resize((500,500)),
        transforms.PILToTensor()
    ])
    dm = VOCSegmentation(root='./data', year="2012",image_set='val', transform=transform_img, target_transform=transform_target)

    # print(len(dm))
    x, y = dm[10]
    print(x.dtype)
    print(torch.min(x), torch.max(x))
    print(y.dtype)
    print(torch.min(y), torch.max(y))
    # x = x.numpy()
    # y = y.numpy()[0,:,:]
    # y_color = label_to_color(y)
    # print(x.shape)
    # print(y.shape)
    # print(np.unique(y))
    # print(y_color.shape)
    # print(np.mean(x))

    # fig, axes = plt.subplots(1,2)
    # axes[0].imshow(x.transpose(1,2,0))
    # axes[1].imshow(y_color)
    # plt.show()

    # transform = transforms.Compose([
    #         # transforms.Resize(32),  # LeNet-5 经典输入是 32x32，MNIST 原是 28x28
    #         transforms.ToTensor()
    #     ])
    # mm = CIFAR10(root='./data',transform=transform, train=False)
    # print((len(mm))) # 50000
