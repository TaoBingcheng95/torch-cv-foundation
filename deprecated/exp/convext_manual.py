"""
https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py

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
    
    LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
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



class ConvNeXtBlockPytochVision(nn.Module):
    """ConvNeXt V1 Block，与PytochVision实现一致。
    流程: DWConv → Permute → nn.LayerNorm → Linear(升维) → GELU → Linear(降维) → Permute → LayerScale → DropPath → Residual
    """
    def __init__(self, dim, 
                 stochastic_depth_prob=0., 
                 layer_scale=1e-6):
        super().__init__()
        
        # 1. 深度可分离卷积 (Depthwise Conv)
        # 模仿 ViT 的 Self-Attention：每个通道独立计算，并在大感受野上混合信息
        # 关键点：kernel_size=7 (大核卷积)，这就好比 Transformer 看得更远
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) 
        
        # 2. LayerNorm 代替 BatchNorm
        # ViT 用 LayerNorm
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        
        # 3. 倒瓶颈结构 (Inverted Bottleneck)
        # 模仿 Transformer 的 MLP Block：先升维 (4倍)，再降维
        # dim -> 4*dim -> dim
        self.pwconv1 = nn.Linear(dim, 4 * dim) 
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        
        # 4. Layer Scale
        # 可学习的逐通道缩放因子，初始值极小 (1e-6)，稳定深层网络训练
        self.layer_scale = nn.Parameter(
            layer_scale * torch.ones(dim), 
            requires_grad=True
        ) if layer_scale > 0 else None
        
        # 5. DropPath (随机深度)
        # 一种正则化手段，随机丢弃整个样本的残差分支
        # 简化演示，实际需调用 timm 的 DropPath，如 `DropPath(stochastic_depth_prob)`
        self.drop_path = nn.Dropout(stochastic_depth_prob) if stochastic_depth_prob > 0. else nn.Identity()

    def forward(self, x):
        input = x
        
        # 步骤 1: 空间混合 (Spatial Mixing)
        x = self.dwconv(x)
        
        # 步骤 2: 归一化 (需要调整维度以适应 LayerNorm)
        # [N, C, H, W] -> [N, H, W, C]
        x = x.permute(0, 2, 3, 1) 
        x = self.norm(x)
        
        # 步骤 3: 通道混合 (Channel Mixing)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        # 步骤 4: Layer Scale (逐通道缩放)
        if self.layer_scale is not None:
            x = self.layer_scale * x
        
        # 换回常规维度 [N, H, W, C] -> [N, C, H, W]
        x = x.permute(0, 3, 1, 2)

        # 步骤 5: 残差连接
        x = input + self.drop_path(x)
        return x



class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt V1 Block — 官方路线 (1)：全程 NCHW，零 permute。
    流程: DWConv → LayerNorm(channels_first) → Conv1x1(升维) → GELU → Conv1x1(降维) → LayerScale → Residual

    ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        stochastic_depth_prob (float): Stochastic depth rate. Default: 0.0
        layer_scale (float): Init value for Layer Scale. Default: 1e-6.

    """
    def __init__(self, dim, stochastic_depth_prob=0., layer_scale=1e-6):
        super().__init__()

        # 1. 深度可分离卷积 (Depthwise Conv)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        
        # 2. LayerNorm (channels_first，无需 permute)
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        
        # 3. 倒瓶颈结构：Conv2d(1x1) 代替 Linear，原生支持 NCHW
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)
        
        # 4. Layer Scale
        self.layer_scale = nn.Parameter(
            layer_scale * torch.ones(dim, 1, 1),
            requires_grad=True
        ) if layer_scale > 0 else None
        
        # 5. DropPath (随机深度)
        self.drop_path = nn.Dropout(stochastic_depth_prob) if stochastic_depth_prob > 0. else nn.Identity()
    
    def forward(self, x):
        input = x
        
        # 步骤 1: 空间混合 (Spatial Mixing)
        x = self.dwconv(x)
        
        # 步骤 2: 归一化
        # x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C) if channels_last
        x = self.norm(x)
        
        # 步骤 3: 通道混合 (Channel Mixing)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        # 步骤 4: Layer Scale (逐通道缩放)
        if self.layer_scale is not None:
            x = self.layer_scale * x
        # x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W) if channels_last
        # 步骤 5: 残差连接
        x = input + self.drop_path(x)
        return x



class PatchMerging(nn.Module):
    #  Patch Merging (下采样层)
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm = LayerNorm(in_channels, eps=1e-6, data_format="channels_first")
        # LayerNorm after downsample
        self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)
    
    def forward(self, x):
        x = self.norm(x)
        x = self.downsample(x)
        return x



class ConvNeXt(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], 
                 dims=[96, 192, 384, 768],
                 drop_path_rate=0., 
                 layer_scale=1e-6):
        super().__init__()
        
        # Stem: 4x4 卷积 + 自定义 LayerNorm
        self.downsample_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
                LayerNorm(dims[0], data_format="channels_first")
            )
        ])

        for i in range(3):
            self.downsample_layers.append(PatchMerging(dims[i], dims[i+1]))
        
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        self.stages = nn.ModuleList()
        # self.stages = nn.Sequential()
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtBlock(dim=dims[i], 
                               stochastic_depth_prob=dp_rates[cur + j],
                               layer_scale=layer_scale)
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
    from torchinfo import summary
    # 创建 ConvNeXt-Base
    model = convnext_base(num_classes=1000)
    model.train()
    
    # 模拟输入
    input_size = (1,3,224,224)
    dummy_input = torch.randn(input_size)
    
    # # 前向传播
    # output = model(x)
    
    # print(f"输入形状: {x.shape}")
    # print(f"输出形状: {output.shape}")
    # print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # # 验证输出
    # assert output.shape == (1, 1000), "输出形状错误！"
    # print("✅ ConvNeXt 测试通过！")

    summary(model, input_size=input_size)
