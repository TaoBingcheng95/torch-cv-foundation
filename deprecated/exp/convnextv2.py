"""
ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders (CVPR 2023)

核心思想：
1. 在 ConvNeXt V1 基础上加入 GRN (Global Response Normalization) 层
2. 使用 FCMAE (Fully Convolutional Masked Autoencoder) 进行自监督预训练
3. 证明纯卷积网络 + MAE 框架可以达到与 ViT-MAE 相当甚至更好的性能

与 MAE (ViT) 的对比：
- MAE 用 Transformer 做 Encoder → ConvNeXt V2 用纯卷积做 Encoder
- MAE 遮罩 75% → FCMAE 遮罩 60%
- MAE 的 Decoder 是 Transformer → FCMAE 的 Decoder 是转置卷积

ref: https://arxiv.org/abs/2301.00808
"""


import torch
from torch import nn

from .utils.pytorch_api import Permute
from .utils.utils import DropPath


# ======================== 1. GRN (Global Response Normalization) ========================

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
        # gamma: 可学习的缩放参数，控制归一化强度（初始化为 0，训练初期不影响）
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        # beta: 可学习的偏置参数
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        # x: (B, H, W, C)
        
        # Step 1: 计算全局响应 —— 每个通道在所有空间位置上的 L2 范数
        # 对 H, W 维度求均值再开方，得到每个通道的"全局响应强度"
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)  # (B, 1, 1, C)
        
        # Step 2: 归一化 —— 将全局响应缩放到均值为 1
        # 这样每个通道的特征会被其全局响应"调制"：响应大的通道被抑制，小的被增强
        # Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)  # (B, 1, 1, C)
        Nx = Gx / (Gx.mean(dim=(1,2,3), keepdim=True) + 1e-6)  # 对 H, W, C 三个维度取均值
        
        # Step 3: 应用归一化 + 可学习参数
        return self.gamma * (x * Nx) + self.beta + x


# ======================== 2. ConvNeXt V2 Block ========================

class ConvNeXtV2Block(nn.Module):
    """
    ConvNeXt V2 的基本构建块
    
    与 ConvNeXt V1 Block 的唯一区别：加入了 GRN 层。
    这一个小改动让纯卷积网络在 MAE 框架下也能有效预训练。
    
    数据流：
        x → DWConv 7x7 → Permute → LN → Linear(C→4C) → GELU → GRN → Linear(4C→C) → Permute → +残差
    
    Input/Output: (B, C, H, W)
    """
    def __init__(self, dim, drop_path: float=0.0):
        """
        :param dim: 通道数 C
        :param drop_path: 随机深度概率（Stochastic Depth）
        """
        super().__init__()
        
        # 深度可分离卷积 (Depthwise Convolution)：7x7 大感受野，每个通道独立卷积
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        
        # LayerNorm 对每个位置的特征向量归一化
        self.norm = nn.LayerNorm(dim)
        
        # 逐点卷积 (Pointwise)：用 Linear 实现 1x1 Conv，先扩展到 4C
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        
        # GELU 激活函数
        self.act = nn.GELU()
        
        # GRN：ConvNeXt V2 的核心创新
        self.grn = GRN(4 * dim)
        
        # 逐点卷积：从 4C 投影回 C
        self.pwconv2 = nn.Linear(4 * dim, dim)
        
        # 随机深度（训练时随机跳过整个 block）
        # self.drop_path = drop_path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, x):
        # x: (B, C, H, W)
        residual = x
        
        # Depthwise Conv 7x7：提取局部空间特征
        x = self.dwconv(x)  # (B, C, H, W)
        
        # 转换到 channels-last 格式以便使用 Linear 和 LayerNorm
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        
        x = self.norm(x)
        x = self.pwconv1(x)    # (B, H, W, 4C) - 通道扩展
        x = self.act(x)
        x = self.grn(x)        # GRN：通道间竞争机制
        x = self.pwconv2(x)    # (B, H, W, C) - 通道压缩
        
        # 转换回 channels-first
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        
        # 随机深度 + 残差连接
        x = self.drop_path(x)
        
        return residual + x


# ======================== 3. ConvNeXt V2 Backbone ========================

class ConvNeXtV2(nn.Module):
    """
    ConvNeXt V2 主干网络（用于监督分类）
    
    架构：Stem → [Stage + Downsample] x 3 → Stage → GAP → LN → Classifier
    
    与 ResNet 的关键区别：
    - 使用 7x7 DWConv 代替 3x3 Conv（更大感受野，更少参数）
    - 使用 GELU 代替 ReLU
    - 使用 4x 扩展比（类似 Transformer FFN 的 4x）
    - 更少的下采样层（4 个 stage vs ResNet 的 4 个 stage，但 stride 更小）
    """
    def __init__(
        self,
        in_chans=3,
        num_classes=1000,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        drop_path_rate=0.0,
    ):
        """
        :param in_chans: 输入通道数（RGB=3）
        :param num_classes: 分类类别数
        :param depths: 每个 stage 中的 block 数量
        :param dims: 每个 stage 的通道数
        :param drop_path_rate: 随机深度的最大概率（线性递增）
        """
        super().__init__()
        
        # ---- Stem: 将图像下采样 4 倍 ----
        # 4x4 Conv, stride 4: (B, 3, 224, 224) → (B, dims[0], 56, 56)
        # 修正 stem 中 LayerNorm 的输入格式
        # Conv2d 输出 (B,C,H,W)，LayerNorm 期望 last dim 为 C
        # 所以我们用一个 Permute 来处理
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            Permute([0, 2, 3, 1]), # Permute(0, 2, 3, 1),       # (B, C, H, W) → (B, H, W, C)
            nn.LayerNorm(dims[0]),
            Permute([0, 3, 1, 2]),  # Permute(0, 3, 1, 2),       # (B, H, W, C) → (B, C, H, W)
        )
        
        # ---- 4 个 Stage ----
        # 计算每个 block 的 drop_path 概率（线性递增：0 → drop_path_rate）
        total_depth = sum(depths)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]
        
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        
        cur = 0
        for i in range(4):
            # Stage i: 由 depths[i] 个 ConvNeXtV2Block 组成
            stage = nn.Sequential(*[
                ConvNeXtV2Block(dim=dims[i], drop_path=dp_rates[cur + j])
                for j in range(depths[i])
            ])
            self.stages.append(stage)
            cur += depths[i]
            
            # Downsample: 在 stage 之间下采样 2 倍（最后一个 stage 后不需要）
            if i < 3:
                downsample = nn.Sequential(
                    Permute([0, 2, 3, 1]),              # (B,C,H,W) → (B,H,W,C)
                    nn.LayerNorm(dims[i]),
                    Permute([0, 3, 1, 2]),              # (B,H,W,C) → (B,C,H,W)
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2)
                )
                self.downsamples.append(downsample)
            else:
                self.downsamples.append(nn.Identity())
        
        # ---- 输出头 ----
        self.head_norm = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                  # Global Average Pooling: (B,C,H,W) → (B,C,1,1)
            nn.Flatten(1),                             # (B, C, 1, 1) → (B, C, 1)
        )
        self.head_ln = nn.LayerNorm(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        
        # 权重初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
    
    def forward_features(self, x):
        """提取特征（不经过分类头），FCMAE 也会用到"""
        x = self.stem(x)  # (B, dims[0], H/4, W/4) = (B, 96, 56, 56)
        
        for i in range(4):
            x = self.stages[i](x)
            if i < 3:
                x = self.downsamples[i](x)
        
        return x  # (B, dims[3], H/32, W/32) = (B, 768, 7, 7)
    
    def forward(self, x):
        x = self.forward_features(x)       # (B, dims[3], H/32, W/32)
        x = self.head_norm(x)  # (B, dims[3])
        x = self.head_ln(x)
        x = self.head(x)                   # (B, num_classes)
        return x



# ======================== 4. FCMAE (Fully Convolutional Masked Autoencoder) ========================

class FCMAE_ConvNeXtV2(nn.Module):
    """
    全卷积遮罩自编码器 (Fully Convolutional Masked Autoencoder)
    
    将 MAE 的思想应用到纯卷积网络：
    1. 随机遮住 60% 的 patch（MAE 遮 75%，因为卷积需要更多上下文）
    2. Encoder (ConvNeXt V2) 处理所有 patch（教学简化版，不做稀疏处理）
    3. Decoder (转置卷积) 重建被遮住的 patch 像素
    4. Loss 仅在被遮住的 patch 上计算
    
    与 MAE 的对比：
    - MAE Decoder: Transformer blocks → FCMAE Decoder: 转置卷积上采样
    - MAE 预测 patch 像素 → FCMAE 预测 stem 输出的特征图
    - MAE mask_ratio=0.75 → FCMAE mask_ratio=0.60
    """
    def __init__(
        self,
        img_size=224,
        in_chans=3,
        patch_size=4,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        decoder_depth=1,
        decoder_embed_dim=512,
        drop_path_rate=0.0,
        mask_ratio=0.6,
    ):
        """
        :param img_size: 输入图像大小
        :param patch_size: stem 的下采样倍数（ConvNeXt 默认 4）
        :param depths: Encoder 每个 stage 的 block 数
        :param dims: Encoder 每个 stage 的通道数
        :param decoder_depth: Decoder 中 Block 的层数
        :param decoder_embed_dim: Decoder 的通道维度
        :param mask_ratio: 遮罩比例（默认 0.6，即遮住 60%）
        """
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        
        # 计算 stem 输出的特征图大小
        self.feat_h = img_size // patch_size  # 224/4 = 56
        self.feat_w = img_size // patch_size  # 224/4 = 56
        self.num_patches = self.feat_h * self.feat_w  # 56*56 = 3136
        
        # ---- Encoder: ConvNeXt V2 backbone（不含分类头）----
        self.encoder = ConvNeXtV2(
            in_chans=in_chans,
            num_classes=0,  # 不需要分类头
            depths=depths,
            dims=dims,
            drop_path_rate=drop_path_rate,
        )
        encoder_dim = dims[-1]  # Encoder 最终输出的通道数
        
        # ---- Decoder: 从 Encoder 特征重建 ----
        # 1. 投影到 decoder 维度
        self.decoder_embed = nn.Conv2d(encoder_dim, decoder_embed_dim, kernel_size=1)
        
        # 2. Decoder Transformer blocks（在特征图上做自注意力，帮助全局信息交互）
        self.decoder_blocks = nn.ModuleList([
            ConvNeXtV2Block(dim=decoder_embed_dim, drop_path=0.0)
            for _ in range(decoder_depth)
        ])
        
        # 3. 转置卷积上采样：从 H/32 恢复到 H/4（上采样 8 倍 = 2x2x2）
        self.decoder_upsample = nn.Sequential(
            # H/32 → H/16
            nn.ConvTranspose2d(decoder_embed_dim, decoder_embed_dim, kernel_size=2, stride=2),
            nn.GroupNorm(1, decoder_embed_dim),  # 等价于 InstanceNorm
            nn.GELU(),
            # H/16 → H/8
            nn.ConvTranspose2d(decoder_embed_dim, decoder_embed_dim, kernel_size=2, stride=2),
            nn.GroupNorm(1, decoder_embed_dim),
            nn.GELU(),
            # H/8 → H/4
            nn.ConvTranspose2d(decoder_embed_dim, decoder_embed_dim, kernel_size=2, stride=2),
            nn.GroupNorm(1, decoder_embed_dim),
            nn.GELU(),
        )
        
        # 4. 预测头：输出 stem 特征图的像素值
        # 目标维度 = stem 的输出通道数 = dims[0]
        self.decoder_pred = nn.Conv2d(decoder_embed_dim, dims[0], kernel_size=1)
        
        # 5. 可学习的 mask token（替代被遮住的 patch 位置）
        self.mask_token = nn.Parameter(torch.zeros(1, encoder_dim, 1, 1))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        
        # 6. mask token 投影层：将 encoder_dim 投影到 stem 输出维度 dims[0]
        #    因为 mask_token 在 encoder 维度空间，需要投影到 stem 特征空间才能替换
        self.mask_token_proj = nn.Conv2d(encoder_dim, dims[0], kernel_size=1)
        
        # 权重初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            if m.bias is not None:
                nn.init.zeros_(m.bias)
            if m.weight is not None:
                nn.init.ones_(m.weight)
    
    def generate_mask(self, batch_size, device):
        """
        生成随机遮罩，在 stem 特征图的空间维度（H/4 x W/4）上操作。
        返回：
        - mask_spatial: (B, 1, H/4, W/4) 空间遮罩，可直接与特征图相乘
        """
        h, w = self.feat_h, self.feat_w  # 56 x 56
        N = h * w
        len_keep = int(N * (1 - self.mask_ratio))
        
        noise = torch.rand(batch_size, N, device=device)  # (B, 3136)
        ids_shuffle = torch.argsort(noise, dim=1)
        
        # 生成空间遮罩：0=可见，1=被遮住
        mask = torch.ones(batch_size, N, device=device)  # (B, 3136)
        mask[:, :len_keep] = 0
        # 恢复原始顺序
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        # 重塑为空间维度以便与特征图操作
        mask_spatial = mask.view(batch_size, 1, h, w)  # (B, 1, 56, 56)
        
        return mask_spatial
    
    def forward_encoder(self, x, mask_spatial):
        """
        Encoder: 通过 ConvNeXt V2 backbone 提取特征。
        
        关键步骤：在 stem 输出上应用遮罩，将被遮位置替换为可学习的 mask_token。
        这样 Encoder 看不到被遮区域的真实特征，迫使网络通过上下文推理来重建。
        
        :param x: 输入图像 (B, 3, H, W)
        :param mask_spatial: 空间遮罩 (B, 1, H/4, W/4)，0=可见，1=被遮
        :return: encoder 输出特征 (B, dims[-1], H/32, W/32)
        """
        # Stem 提取特征
        x = self.encoder.stem(x)  # (B, dims[0], H/4, W/4)
        
        # 将 mask_token 上采样到 stem 特征图尺度，填充被遮位置
        # mask_token: (1, encoder_dim, 1, 1) → 广播到 (B, encoder_dim, H/4, W/4)
        # 注意：这里用 encoder_dim 投影到 dims[0] 维度，或直接初始化 dims[0] 的 mask_token
        # 简化实现：直接用 1x1 Conv 将 mask_token 投影到 stem 维度
        mask_tokens = self.mask_token_proj(self.mask_token)  # (1, dims[0], 1, 1)
        # 替换被遮住的区域：x * (1 - mask) + mask_token * mask
        x = x * (1 - mask_spatial) + mask_tokens * mask_spatial
        
        # 经过后续 stages + downsamples
        for i in range(4):
            x = self.encoder.stages[i](x)
            if i < 3:
                x = self.encoder.downsamples[i](x)
        
        return x  # (B, dims[-1], H/32, W/32)
    
    def forward_decoder(self, feat):
        """
        Decoder: 从 Encoder 特征重建 stem 输出的特征图。
        
        流程：
        1. 1x1 Conv 投影到 decoder 维度
        2. 通过 ConvNeXtV2 blocks 做特征交互（含 GRN 通道竞争）
        3. 三层转置卷积上采样 8 倍（H/32 → H/4）
        4. 1x1 Conv 预测头输出 stem 特征图
        
        :param feat: Encoder 输出 (B, dims[-1], H/32, W/32)
        :return: 重建的 stem 特征图 (B, dims[0], H/4, W/4)
        """
        B, C, h, w = feat.shape
        
        # 投影到 decoder 维度
        x = self.decoder_embed(feat)  # (B, decoder_dim, h, w)
        
        # Decoder blocks：在特征图上做卷积交互
        for blk in self.decoder_blocks:
            x = blk(x)
        
        # 转置卷积上采样：h x w → (h*8) x (w*8) = H/4 x W/4
        x = self.decoder_upsample(x)  # (B, decoder_dim, H/4, W/4)
        
        # 预测头：输出 stem 特征图
        x = self.decoder_pred(x)  # (B, dims[0], H/4, W/4)
        
        return x
    
    def forward_loss(self, target, pred, mask_spatial):
        """
        计算重建损失，仅在被遮住的 patch 上计算。
        
        FCMAE 的重建目标是 stem 输出的特征图（而非原图像素）。
        这样 decoder 只需上采样 8 倍（而非 32 倍），且目标在特征空间更平滑。
        
        target: (B, dims[0], H/4, W/4) stem 输出特征图
        pred:   (B, dims[0], H/4, W/4) decoder 预测的特征图
        mask_spatial: (B, 1, H/4, W/4) 空间遮罩，0=可见，1=被遮住
        """
        # 逐像素 MSE loss
        loss = (pred - target) ** 2          # (B, dims[0], H/4, W/4)
        loss = loss.mean(dim=1)              # (B, H/4, W/4) 对通道维取均值
        
        # 仅对被遮住的区域计算 loss
        mask = mask_spatial.squeeze(1)       # (B, H/4, W/4)
        loss = (loss * mask).sum() / mask.sum()
        
        return loss
    
    def forward(self, imgs):
        """
        完整前向传播：遮罩 → 编码 → 解码 → 计算损失
        
        FCMAE 的数据流：
        imgs → stem → [mask with mask_token] → encoder → decoder(pred) → MSE loss
                                                                    ↘ stem(target, detached)
        
        与 MAE 的对比：
        - MAE: 在 patch embedding 后替换被遮 token → FCMAE: 在 stem 特征图上替换被遮位置
        - MAE: Encoder 只处理可见 token（省 75% 计算）→ 这里仍处理完整特征图（教学简化）
        """
        B = imgs.shape[0]
        
        # Step 1: 获取重建目标 —— stem 输出的特征图（stop gradient）
        # detach 防止梯度回传到 stem，因为 target 是"答案"，不应参与编码器训练
        with torch.no_grad():
            target = self.encoder.stem(imgs)  # (B, dims[0], H/4, W/4)
        
        # Step 2: 生成空间遮罩（在 H/4 x W/4 的特征图尺度上）
        mask_spatial = self.generate_mask(B, imgs.device)  # (B, 1, H/4, W/4)
        
        # Step 3: Encoder 提取特征（在 stem 输出上应用遮罩）
        # 被遮位置替换为 mask_token，Encoder 无法看到被遮区域的真实特征
        feat = self.forward_encoder(imgs, mask_spatial)  # (B, dims[-1], H/32, W/32)
        
        # Step 4: Decoder 重建 stem 特征图
        pred = self.forward_decoder(feat)  # (B, dims[0], H/4, W/4)
        
        # Step 5: 计算损失（仅在被遮住的 patch 上）
        loss = self.forward_loss(target, pred, mask_spatial)
        
        return loss, pred, mask_spatial


# ======================== 5. 工厂函数 ========================

def convnextv2_atto(**kwargs):
    """ConvNeXt V2 Atto: 最小模型，~3.7M 参数"""
    return ConvNeXtV2(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)

def convnextv2_tiny(**kwargs):
    """ConvNeXt V2 Tiny: ~29M 参数"""
    return ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)

def convnextv2_base(**kwargs):
    """ConvNeXt V2 Base: ~89M 参数"""
    return ConvNeXtV2(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)

def fcmae_convnextv2_tiny(**kwargs):
    """FCMAE ConvNeXt V2 Tiny: 自监督预训练版本"""
    return FCMAE_ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)

def fcmae_convnextv2_base(**kwargs):
    """FCMAE ConvNeXt V2 Base: 自监督预训练版本"""
    return FCMAE_ConvNeXtV2(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)


# ======================== 6. 使用 Demo：FCMAE 预训练 → 微调 ========================

def _count_params(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters()) / 1e6


def demo_fcmae_pretrain():
    """
    FCMAE 自监督预训练 Demo
    
    演示 FCMAE_ConvNeXtV2 的完整训练流程：
    1. 构建 FCMAE 模型（Encoder + Decoder）
    2. 在随机数据上进行自监督预训练（重建被遮住的 stem 特征）
    3. 提取预训练好的 Encoder 权重
    """
    import torch.optim as optim
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("FCMAE 自监督预训练 Demo")
    print("=" * 60)
    
    # ---- 构建 FCMAE 模型 ----
    # 使用较小的配置加速演示（实际训练用 Tiny/Base 配置）
    model = FCMAE_ConvNeXtV2(
        img_size=64,          # 小图像加速演示
        patch_size=4,
        depths=[2, 2, 2, 2],   # 较浅的网络
        dims=[32, 64, 128, 256],
        decoder_depth=1,
        decoder_embed_dim=128,
        mask_ratio=0.6,
    ).to(device)
    
    enc_params = _count_params(model.encoder)
    dec_params = _count_params(model) - enc_params
    print(f"\n模型参数量:")
    print(f"  Encoder: {enc_params:.2f}M")
    print(f"  Decoder: {dec_params:.2f}M")
    print(f"  总计:    {_count_params(model):.2f}M")
    print(f"\n配置: img_size=64, mask_ratio=0.6, stem输出: 16x16")
    
    # ---- 优化器 ----
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    
    # ---- 模拟训练 ----
    print("\n--- 自监督预训练 ---")
    model.train()
    num_epochs = 5
    batch_size = 4
    
    for epoch in range(num_epochs):
        # 模拟无标签图像数据（自监督不需要标签）
        imgs = torch.randn(batch_size, 3, 64, 64, device=device)
        
        # 前向传播：遮罩 → 编码 → 解码 → 计算损失
        loss, pred, mask = model(imgs)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计遮罩比例
        mask_ratio_actual = mask.mean().item()
        print(f"  Epoch {epoch+1}/{num_epochs}  "
              f"loss={loss.item():.4f}  "
              f"mask_ratio={mask_ratio_actual:.2f}  "
              f"pred_shape={tuple(pred.shape)}")
    
    print("\n预训练完成！")
    return model


def demo_finetune(fcmae_model):
    """
    微调 Demo：从 FCMAE 预训练的 Encoder 提取权重，加载到分类模型
    
    流程：
    1. 从 FCMAE 提取 Encoder 权重
    2. 构建 ConvNeXtV2 分类模型
    3. 加载预训练权重
    4. 在模拟数据上微调
    """
    import torch.optim as optim
    
    device = next(fcmae_model.parameters()).device
    print("\n" + "=" * 60)
    print("FCMAE 预训练 → 微调 Demo")
    print("=" * 60)
    
    # ---- Step 1: 提取 Encoder 权重 ----
    encoder_state_dict = fcmae_model.encoder.state_dict()
    print(f"\n从 FCMAE 提取 Encoder 权重: {len(encoder_state_dict)} 个参数张量")
    
    # ---- Step 2: 构建分类模型（与 FCMAE Encoder 配置一致）----
    num_classes = 10  # 假设 10 类分类任务
    classifier = ConvNeXtV2(
        in_chans=3,
        num_classes=num_classes,
        depths=[2, 2, 2, 2],
        dims=[32, 64, 128, 256],
    ).to(device)
    
    # ---- Step 3: 加载预训练权重 ----
    # Encoder 权重包含 stem/stages/downsamples/head_norm/head_ln
    # 分类模型的 head 是新增的，不匹配，用 strict=False 跳过
    missing, unexpected = classifier.load_state_dict(encoder_state_dict, strict=False)
    print(f"\n权重加载结果:")
    print(f"  缺少（新增的分类头）: {[k for k in missing if 'head' in k]}")
    print(f"  多余（FCMAE独有）:   {unexpected}")
    
    # ---- Step 4: 微调 ----
    # 分类头用较大学习率，Encoder 用较小学习率（典型微调策略）
    head_params = [p for n, p in classifier.named_parameters() if 'head' in n]
    backbone_params = [p for n, p in classifier.named_parameters() if 'head' not in n]
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': 1e-5},  # Encoder: 小学习率，保留预训练知识
        {'params': head_params,     'lr': 1e-3},  # 分类头: 大学习率，快速适应
    ], weight_decay=0.05)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\n--- 监督微调 ({num_classes} 类) ---")
    classifier.train()
    num_epochs = 3
    batch_size = 4
    
    for epoch in range(num_epochs):
        # 模拟带标签的训练数据
        imgs = torch.randn(batch_size, 3, 64, 64, device=device)
        labels = torch.randint(0, num_classes, (batch_size,), device=device)
        
        # 前向传播
        logits = classifier(imgs)  # (B, num_classes)
        loss = criterion(logits, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 计算准确率
        acc = (logits.argmax(dim=1) == labels).float().mean().item()
        print(f"  Epoch {epoch+1}/{num_epochs}  "
              f"loss={loss.item():.4f}  "
              f"acc={acc:.2%}  "
              f"logits_shape={tuple(logits.shape)}")
    
    print("\n微调完成！")
    
    # ---- 推理演示 ----
    classifier.eval()
    with torch.no_grad():
        test_imgs = torch.randn(2, 3, 64, 64, device=device)
        test_logits = classifier(test_imgs)
        test_preds = test_logits.argmax(dim=1)
        print(f"\n推理示例:")
        print(f"  输入: {tuple(test_imgs.shape)}")
        print(f"  预测类别: {test_preds.tolist()}")
        print(f"  输出 logits: {test_logits.shape}")


def demo_full_pipeline():
    """
    完整流水线：FCMAE 预训练 → 提取权重 → 分类微调
    
    这是 ConvNeXt V2 论文的标准训练范式：
    1. 在大规模无标签数据上用 FCMAE 自监督预训练
    2. 丢弃 Decoder，保留 Encoder
    3. 在下游任务上微调 Encoder + 新分类头
    """
    print("\n" + "#" * 60)
    print("# ConvNeXt V2 完整训练流水线")
    print("# FCMAE 自监督预训练 → 监督微调")
    print("#" * 60)
    
    # Phase 1: 自监督预训练
    fcmae_model = demo_fcmae_pretrain()
    
    # Phase 2: 微调
    demo_finetune(fcmae_model)
    
    print("\n" + "=" * 60)
    print("流水线完成！")
    print("=" * 60)
    print("""
总结:
  1. FCMAE 预训练: 遮罩 60% stem 特征 → Encoder → Decoder 重建 → 仅遮罩区域计算 Loss
  2. 提取 Encoder: 丢弃 Decoder，保留预训练好的 ConvNeXt V2 backbone
  3. 监督微调:   加载 Encoder 权重 → 添加新分类头 → 在目标任务上微调
""")


if __name__ == "__main__":
    demo_full_pipeline()
