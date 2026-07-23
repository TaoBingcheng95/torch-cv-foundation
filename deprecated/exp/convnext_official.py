"""
ConvNeXt / ConvNeXt V2 统一实现
基于 facebookresearch 官方仓库复现：
- ConvNeXt V1: https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
- ConvNeXt V2: https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/convnextv2.py

核心区别：
- V1 Block: DWConv → LN → Linear(C→4C) → GELU → Linear(4C→C) → Layer Scale → 残差
- V2 Block: DWConv → LN → Linear(C→4C) → GELU → GRN → Linear(4C→C) → 残差
  V2 用 GRN (Global Response Normalization) 替代了 V1 的 Layer Scale，
  使纯卷积网络在 MAE 框架下也能有效预训练。

通过 use_grn 参数切换：
  use_grn=False → ConvNeXt V1 (Layer Scale)
  use_grn=True  → ConvNeXt V2 (GRN)

ConvNeXt V2 变体规格：
标签    SI 词头    含义（10的幂次）    depths / dims              参数量级
atto    a (阿托)   10⁻¹⁸             [2,2,6,2] / [40,80,160,320]    ~3.7M
femto   f (飞母托) 10⁻¹⁵             [2,2,6,2] / [48,96,192,384]    ~5.2M
pico    p (皮可)   10⁻¹²             [2,2,6,2] / [64,128,256,512]   ~9.0M
nano    n (纳诺)   10⁻⁹              [2,2,8,2] / [80,160,320,640]   ~15.6M
tiny    —          极小              [3,3,9,3] / [96,192,384,768]   ~28.6M
base    —          基准              [3,3,27,3] / [128,256,512,1024] ~88.7M
large   —          大                [3,3,27,3] / [192,384,768,1536] ~197.7M
huge    —          巨大              [3,3,27,3] / [352,704,1408,2816] ~659M

从 tiny 开始，模型 depths 变为完整的 [3,3,9,3]

ref: https://arxiv.org/abs/2201.03545 (V1)
ref: https://arxiv.org/abs/2301.00808 (V2)
"""

import torch
import torch.nn as nn

from .utils.utils import DropPath
from .utils.convnext_api import LayerNorm, GRN, PatchMerging


__all__ = [
    # 类
    'ConvNeXt', 'ConvNeXtV2', 'ConvNeXtBlock', 'FCMAE_ConvNeXtV2',
    # ConvNeXt V1
    'convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large', 'convnext_xlarge',
    # ConvNeXt V2
    'convnextv2_atto', 'convnextv2_femto', 'convnextv2_pico', 'convnextv2_nano',
    'convnextv2_tiny', 'convnextv2_base', 'convnextv2_large', 'convnextv2_huge',
    # FCMAE
    'fcmae_convnextv2_tiny', 'fcmae_convnextv2_base',
]



class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt Block（统一 V1 / V2）

    V1 数据流 (use_grn=False):
        x → DWConv 7x7 → Permute → LN → Linear(C→4C) → GELU → Linear(4C→C) → γ·x → +残差

    V2 数据流 (use_grn=True):
        x → DWConv 7x7 → Permute → LN → Linear(C→4C) → GELU → GRN → Linear(4C→C) → +残差

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale (V1 only). Default: 1e-6.
        use_grn (bool): If True, use GRN instead of Layer Scale (V2 mode). Default: False.

    Input/Output: (B, C, H, W)
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, use_grn=False):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        # V2: GRN 替代 Layer Scale
        self.grn = GRN(4 * dim) if use_grn else None
        self.pwconv2 = nn.Linear(4 * dim, dim)
        # V1: Layer Scale（V2 模式下不启用）
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if (layer_scale_init_value > 0 and not use_grn) else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        if self.grn is not None:
            x = self.grn(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x



class ConvNeXt(nn.Module):
    """
    ConvNeXt / ConvNeXt V2 统一模型

    架构：Stem → [Stage + Downsample] x 3 → Stage → GAP → LN → Classifier

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale (V1 only). Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
        use_grn (bool): If True, use GRN (V2 mode); otherwise use Layer Scale (V1 mode). Default: False.
    """
    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 use_grn=False,
                 ):
        super().__init__()
        self.depths = depths
        self.use_grn = use_grn

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtBlock(dim=dims[i], drop_path=dp_rates[cur + j],
                                layer_scale_init_value=layer_scale_init_value,
                                use_grn=use_grn) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


# 向后兼容别名：ConvNeXtV2 即 use_grn=True 的 ConvNeXt
ConvNeXtV2 = ConvNeXt


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
        self.encoder = ConvNeXt(
            in_chans=in_chans,
            num_classes=0,  # 不需要分类头
            depths=depths,
            dims=dims,
            drop_path_rate=drop_path_rate,
            use_grn=True,  # V2 模式
        )
        encoder_dim = dims[-1]  # Encoder 最终输出的通道数

        # ---- Decoder: 从 Encoder 特征重建 ----
        # 1. 投影到 decoder 维度
        self.decoder_embed = nn.Conv2d(encoder_dim, decoder_embed_dim, kernel_size=1)

        # 2. Decoder blocks（在特征图上做卷积交互，含 GRN 通道竞争）
        self.decoder_blocks = nn.ModuleList([
            ConvNeXtBlock(dim=decoder_embed_dim, drop_path=0.0, use_grn=True)
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
        # Stem 提取特征（downsample_layers[0] = stem）
        x = self.encoder.downsample_layers[0](x)  # (B, dims[0], H/4, W/4)

        # 将 mask_token 投影到 stem 维度，填充被遮位置
        mask_tokens = self.mask_token_proj(self.mask_token)  # (1, dims[0], 1, 1)
        # 替换被遮住的区域：x * (1 - mask) + mask_token * mask
        x = x * (1 - mask_spatial) + mask_tokens * mask_spatial

        # 经过后续 stages + downsample_layers
        for i in range(4):
            x = self.encoder.stages[i](x)
            if i < 3:
                x = self.encoder.downsample_layers[i + 1](x)

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
            target = self.encoder.downsample_layers[0](imgs)  # (B, dims[0], H/4, W/4)

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



#################################  ConvNeXt V1  #################################

def convnext_tiny(**kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model


def convnext_small(**kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    return model


def convnext_base(**kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model


def convnext_large(**kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model


def convnext_xlarge(**kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    return model


#################################  ConvNeXt V2  #################################

def convnextv2_atto(**kwargs):
    model = ConvNeXt(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], use_grn=True, **kwargs)
    return model


def convnextv2_femto(**kwargs):
    model = ConvNeXt(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], use_grn=True, **kwargs)
    return model


def convnextv2_pico(**kwargs):
    model = ConvNeXt(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], use_grn=True, **kwargs)
    return model


def convnextv2_nano(**kwargs):
    model = ConvNeXt(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], use_grn=True, **kwargs)
    return model


def convnextv2_tiny(**kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], use_grn=True, **kwargs)
    return model


def convnextv2_base(**kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], use_grn=True, **kwargs)
    return model


def convnextv2_large(**kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], use_grn=True, **kwargs)
    return model


def convnextv2_huge(**kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], use_grn=True, **kwargs)
    return model


#################################  FCMAE  #################################

def fcmae_convnextv2_tiny(**kwargs):
    """FCMAE ConvNeXt V2 Tiny: 自监督预训练版本"""
    return FCMAE_ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)


def fcmae_convnextv2_base(**kwargs):
    """FCMAE ConvNeXt V2 Base: 自监督预训练版本"""
    return FCMAE_ConvNeXtV2(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
