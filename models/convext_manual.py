"""
NCHW 与 NHWC 的冲突。
卷积层的默认格式 (NCHW)：PyTorch 的 nn.Conv2d 输出格式是 [Batch, Channels, Height, Width]。经过 Stem 的 4x4 卷积后，您的特征图形状是 [2, 96, 56, 56]，通道数 (96) 在第 2 维。
LayerNorm 的默认格式 (NHWC)：PyTorch 原生的 nn.LayerNorm 是为 NLP 任务设计的，它期望输入的最后一维是特征维度（通道数）。它期望的形状是 [Batch, Height, Width, Channels]。
冲突爆发：当 [2, 96, 56, 56] 送入 nn.LayerNorm(96) 时，LayerNorm 发现最后一维是 56，而不是它期望的 96，于是直接抛出异常。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F



class LayerNorm(nn.Module):
    """ 
    支持 channels_last (NHWC) 和 channels_first (NCHW) 两种格式的 LayerNorm。
    视觉任务中强烈建议使用 channels_first。
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # 针对 NCHW 格式手动计算 LayerNorm
            u = x.mean(1, keepdim=True)  # 在通道维度 (dim=1) 求均值
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            # 广播乘法：将 [C] 变成 [1, C, 1, 1] 以匹配 NCHW
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x



class GRN(nn.Module):
    #  GRN 模块 (适配 channels_first)
    def __init__(self, dim, data_format="channels_first"):
        super().__init__()
        # 根据数据格式调整 gamma 和 beta 的形状
        if data_format == "channels_first":
            self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))
        else:
            self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
            self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))
    
    def forward(self, x):
        # 计算空间维度的 L2 范数
        if x.dim() == 4: # NCHW
            Gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        else: # NHWC
            Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
            
        Nx = Gx / (Gx.mean(dim=1 if x.dim()==4 else -1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta



class ConvNeXtBlock(nn.Module):
    # ConvNeXt Block (全程 NCHW，无需 permute)
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        
        # 使用自定义的 channels_first LayerNorm
        self.norm1 = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        
        # 【优化】：用 Conv2d(1x1) 代替 nn.Linear，原生支持 NCHW，无需 permute！
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1) 
        self.act = nn.GELU()
        
        self.dwconv = nn.Conv2d(4 * dim, 4 * dim, kernel_size=7, 
                                padding=3, groups=4 * dim)  # 7x7 深度可分离卷积
        
        self.norm2 = LayerNorm(4 * dim, eps=1e-6, data_format="channels_first")
        self.grn = GRN(4 * dim, data_format="channels_first")
        
        # 【优化】：同样用 Conv2d(1x1) 代替 nn.Linear
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1) 
        
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()
        
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(1, dim, 1, 1), 
                                  requires_grad=True) if layer_scale_init_value > 0 else None
    
    def forward(self, x):
        input = x
        x = self.norm1(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.dwconv(x)
        x = self.norm2(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        
        if self.gamma is not None:
            x = self.gamma * x
            
        x = input + self.drop_path(x)
        return x



class PatchMerging(nn.Module):
    #  Patch Merging (下采样层)
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)
        # 下采样后接 LayerNorm
        self.norm = LayerNorm(out_channels, data_format="channels_first")
    
    def forward(self, x):
        x = self.downsample(x)
        x = self.norm(x)
        return x



class ConvNeXt(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], 
                 dims=[96, 192, 384, 768],
                 drop_path_rate=0., 
                 layer_scale_init_value=1e-6):
        super().__init__()
        
        # Stem: 4x4 卷积 + 自定义 LayerNorm
        self.downsample_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
                LayerNorm(dims[0], data_format="channels_first") # 修复点在这里！
            )
        ])
        
        for i in range(3):
            self.downsample_layers.append(PatchMerging(dims[i], dims[i+1]))
        
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        self.stages = nn.ModuleList()
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtBlock(dim=dims[i], 
                               drop_path=dp_rates[cur + j],
                               layer_scale_init_value=layer_scale_init_value)
                  for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]
        
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # 最后分类前转回 NHWC 用原生 LN 即可
        self.head = nn.Linear(dims[-1], num_classes)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        
        # 全局平均池化：将 NCHW 变成 NC，然后送入 LayerNorm 和 Linear
        x = x.mean([-2, -1]) 
        x = self.norm(x)
        return x
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


# 工厂函数
def convnext_tiny(num_classes=1000):
    return ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                    num_classes=num_classes)


def convnext_small(num_classes=1000):
    return ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], 
                    num_classes=num_classes)


def convnext_base(num_classes=1000):
    return ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], 
                    num_classes=num_classes)


def convnext_large(num_classes=1000):
    return ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], 
                    num_classes=num_classes)



if __name__ == "__main__":
    # 创建 ConvNeXt-Tiny
    model = convnext_tiny(num_classes=1000)
    model.train()
    
    # 模拟输入
    x = torch.randn(1, 3, 224, 224)
    
    # 前向传播
    output = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # 验证输出
    assert output.shape == (1, 1000), "输出形状错误！"
    print("✅ ConvNeXt 测试通过！")
