
import torch
from torch import nn
import torch.nn.functional as F



class GRN(nn.Module):
    """
    全局响应归一化 (Global Response Normalization)
    
    ConvNeXt V2 的核心创新。解决的问题：
    - 纯卷积网络的通道间缺乏竞争机制（每个通道独立处理）
    - MAE 预训练在纯卷积网络上效果不如 ViT（特征坍塌）
    
    GRN 的做法：
    1. 计算每个通道在所有空间位置上的全局响应（L2 范数）
    2. 用全局响应归一化当前特征，使通道之间产生竞争
    3. 通过可学习参数控制归一化的强度
    
    Input/Output: (B, H, W, C)
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x



class LayerNorm(nn.Module):
    """
    LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        if data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x



class PatchMerging(nn.Module):
    #  Patch Merging (下采样层)
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm = LayerNorm(in_channels, eps=1e-6, data_format="channels_first")
        self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)
    
    def forward(self, x):
        x = self.norm(x)
        x = self.downsample(x)
        return x
