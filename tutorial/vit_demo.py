"""
ref: https://dmichel.github.io/machine-learning/minvit/
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchinfo import summary



# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def visualize_position_embedding_similarity():
    # 假设我们有一个训练好的 ViT，Patch Size=16, 图片 224x224
    # Grid 大小为 14x14 = 196 个 Patches
    N = 14 
    num_patches = N * N
    dim = 768 # Embedding 维度

    # 【模拟数据】：真实的训练权重无法在此生成。
    # 这里我们用数学方式构造一个理想化的“距离相关”矩阵，
    # 模拟训练好的模型应该呈现出的效果：物理距离越近，向量相似度越高。
    # 这是一个“以图为实”的教学演示。
    
    # 1. 生成网格坐标
    coords = np.array([(i, j) for i in range(N) for j in range(N)])
    
    # 2. 计算所有 Patch 之间的欧氏距离矩阵
    # shape: (196, 196)
    dist_matrix = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    
    # 3. 模拟相似度：距离越近，相似度越高 (高斯衰减)
    # 真实训练出的 Pos Embed 也会呈现出这种类似于 2D 拓扑的结构
    sim_matrix = np.exp(-dist_matrix ** 2 / (2 * 2.5 ** 2))

    # 4. 可视化
    # 我们选择中心点、左上角点等几个关键位置，看它们关注哪里
    target_indices = [
        (0, 0),   # 左上角 Patch
        (0, 7),   # 顶部中间 Patch
        (7, 0),   # 左侧中间 Patch
        (7, 7),   # 中心 Patch
    ]
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    for idx, (r, c) in enumerate(target_indices):
        # 找到该 Patch 在 1D 序列中的索引
        query_idx = r * N + c
        
        # 取出该 Patch 与所有其他 Patch 的相似度
        sim_map = sim_matrix[query_idx].reshape(N, N)
        
        ax = axes[idx]
        im = ax.imshow(sim_map, cmap='viridis', vmin=0, vmax=1)
        
        # 标记查询点的位置
        ax.scatter([c], [r], c='red', marker='x', s=100, linewidth=2)
        ax.set_title(f"查询 Patch: 行{r}, 列{c}", fontsize=12)
        ax.axis('off')

    # 添加颜色条
    cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="位置编码相似度")
    
    plt.suptitle(f"位置编码自相似度可视化 ({N}x{N} Grid)", fontsize=16)
    return fig



class SimpleViT(nn.Module):
    def __init__(self, image_size=224, 
                    patch_size=16, 
                    num_classes=1000, 
                    dim=768, 
                    depth=12, 
                    heads=12):
        super().__init__()
        
        # 检查图像尺寸是否能被 Patch 整除
        assert image_size % patch_size == 0, "图像尺寸必须能被 Patch Size 整除"
        
        # 计算 Patch 的总数量 N
        num_patches = (image_size // patch_size) ** 2
        
        # 1. Patch Embedding 层
        # 作用：将图像切块并投影到 dim 维度
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        
        # 2. 可学习的参数 (Learnable Parameters)
        # [CLS] Token: 聚合全局信息的特殊向量
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        # 位置编码: 加上它，模型才知道“左上角”和“右下角”的区别
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        
        # 3. Transformer Encoder (关键修改区域！)
        # dim_feedforward: PyTorch默认是2048，对于小模型过大，我们改为标准的 4*dim
        # norm_first=True: 开启 Pre-Norm。这是 ViT 能够快速收敛的秘诀！
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, 
            nhead=heads, 
            dim_feedforward=dim * 4, 
            dropout=0.1,
            activation='gelu',  # ViT 标配激活函数
            batch_first=True,
            norm_first=True     # <--- 重点：前置归一化
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # 4. MLP Head 分类头
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        # x: [Batch, 3, H, W]
        
        # Step 1: Patch Embedding
        # 卷积后形状: [Batch, dim, H/P, W/P] -> 展平 -> [Batch, N, dim]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        
        # Step 2: 拼接 [CLS] Token
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        # Step 3: 加上位置编码
        x = x + self.pos_embed
        
        # Step 4: Transformer 编码 (Pre-Norm 在内部自动处理)
        x = self.transformer(x)
        
        # Step 5: 取出 [CLS] 对应的输出
        cls_output = x[:, 0]
        
        return self.mlp_head(cls_output)



# ======================== ViT Model Definition ========================

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    # drop_prob是进行droppath的概率
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    # 在ViT中，shape是(B,1,1),B是batch size
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    # 按shape,产生0-1之间的随机向量,并加上keep_prob  
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    # 向下取整，二值化，这样random_tensor里1出现的概率的期望就是keep_prob
    random_tensor.floor_()  # binarize
    # 将一定图层变为0
    output = x.div(keep_prob) * random_tensor
    return output


class PatchEmbedding(nn.Module):
    """
    将图像切分为 patch 并投影到 embedding 空间。
    本模块只负责 "图像 -> patch tokens" 这一件事，不包含 CLS token 和位置编码。
    
    Input:  (B, C, H, W)
    Output: (B, N, D)  where N = (H/P) * (W/P), D = embed_dim
    
    设计说明：timm 等工业库将 CLS token 和位置编码放在 ViT 主类中管理，
    而不是组装在 PatchEmbedding 里。这样做的理由是：
    - 职责分离：PatchEmbedding 只做图像切分，位置信息是独立的概念
    - 可替换性：不同 ViT 变体可能使用不同的位置编码（RoPE、正弦等）或聚合方式（GAP 代替 CLS）
    - 分辨率灵活：可变输入分辨率时需要对 pos_embed 插值，放在外层更易操作
    """
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 in_ch=3, 
                 embed_dim=768):
        """
        :param img_size: 输入图像的大小
        :param patch_size: 一个 patch 的大小
        :param in_ch: 输入图像的通道数
        :param embed_dim: 输出的每个 patch token 的维度 D
        """
        super().__init__()
        # 校验：图像尺寸必须能被 patch_size 整除，否则 patch 数量计算不正确
        assert img_size % patch_size == 0, \
            f"图像尺寸({img_size})必须能被 patch_size({patch_size}) 整除"
        
        self.img_size = img_size
        self.patch_size = patch_size    # P
        self.in_chans   = in_ch         # C
        self.embed_dim  = embed_dim     # D
        
        # Calculate number of patches
        self.num_patches = (img_size // patch_size) ** 2     # N = (H/P) * (W/P)
        
        # Patch embedding: 等价于对每个 patch 做一次线性投影，用 Conv2d 实现更高效
        # 对每个 P×P 的 patch，通过 stride=patch_size 实现无重叠切分，输出 embed_dim 维向量
        self.proj = nn.Conv2d(
            in_channels=in_ch, 
            out_channels=embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size  # 无重叠切分
        )
        
    def forward(self, x):
        """
        (B, C, H, W) -> Conv2d -> flatten -> transpose -> (B, N, D)
        """
        B, C, H, W = x.shape
        
        # 校验输入图像尺寸是否与初始化时一致
        assert H == self.img_size and W == self.img_size, \
            f"期望输入尺寸({self.img_size}, {self.img_size})，但得到({H}, {W})"
        
        # Step 1: 卷积投影 (B, C, H, W) -> (B, D, H/P, W/P)
        x = self.proj(x)  # (B, 768, 14, 14) for img_size=224, patch_size=16
        
        # Step 2: 展平空间维度 (B, D, H/P, W/P) -> (B, D, N)
        x = x.flatten(2)  # (B, 768, 196)
        
        # Step 3: 转置为序列格式 (B, D, N) -> (B, N, D)
        x = x.transpose(1, 2)  # (B, 196, 768)
        
        return x



class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention (MHSA) module
    Input: (B, N, D)
    Output: (B, N, D)
    """
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.1, bias=False):
        """
        :param embed_dim: 嵌入维度 D（默认768，对应 ViT-Base）
        :param num_heads: 注意力头数（默认12，对应 ViT-Base；每个头的维度 head_dim = embed_dim // num_heads）
                          注意：原版 ViT-Base 使用 heads=12, head_dim=64；
                                若改为 heads=8，则 head_dim=96，与常见实现不同
        :param dropout: Dropout 概率
        """
        super().__init__()
        # Ensure embed_dim is divisible by num_heads
        assert embed_dim % num_heads == 0, "Embed dim must be divisible by num heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # Dimension per head
                
        # Linear layers for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Output linear layer
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # 预计算缩放因子，避免每次 forward 重复创建 tensor
        self.scale = math.sqrt(self.head_dim)
        
        # Dropout layer
        # 注意：原版 ViT 还使用 DropPath（随机深度/Stochastic Depth），
        # 即在残差连接路径上随机丢弃整个子层输出，有助于训练稳定性。
        # 此处简化为仅使用普通 Dropout，作为教学实现。
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, D = x.shape  # (B, 197, 768) for img_size=224, patch_size=16
        
        # Step 1: Compute Q, K, V (B, N, D)
        q = self.q_proj(x)  # (B, 197, 768)
        k = self.k_proj(x)  # (B, 197, 768)
        v = self.v_proj(x)  # (B, 197, 768)
        
        # Step 2: Split into multiple heads (B, num_heads, N, head_dim)
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, 12, 197, 64)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, 12, 197, 64)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, 12, 197, 64)
        
        # Step 3: Compute attention scores (B, num_heads, N, N)
        scores = torch.matmul(q, k.transpose(-2, -1))  # (B, 12, 197, 197)
        scores = scores / self.scale  # Scale by sqrt(head_dim)
        
        # Step 4: Softmax to get attention weights (B, num_heads, N, N)
        attn_weights = torch.softmax(scores, dim=-1)  # (B, 12, 197, 197)
        attn_weights = self.dropout(attn_weights)
        
        # Step 5: Compute weighted sum of V (B, num_heads, N, head_dim)
        attn_output = torch.matmul(attn_weights, v)  # (B, 12, 197, 64)
        
        # Step 6: Concatenate heads (B, N, D)
        attn_output = attn_output.transpose(1, 2).contiguous()  # (B, 197, 12, 64)
        attn_output = attn_output.view(B, N, D)  # (B, 197, 768)
        
        # Step 7: Linear projection
        output = self.out_proj(attn_output)  # (B, 197, 768)
        # output = self.dropout(output)
        
        return output



class MLP(nn.Module):
    """
    Multi-Layer Perceptron for Transformer encoder
    Input: (B, N, D)
    Output: (B, N, D)
    """
    def __init__(self, embed_dim=768, mlp_dim=3072, dropout=0.1, bias=False):
        """
        :param embed_dim: 输入/输出维度 D（默认768）
        :param mlp_dim: 隐藏层维度（默认 4*embed_dim = 3072，符合 ViT 论文设计）
        :param dropout: Dropout 概率
        """
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, mlp_dim, bias=bias)  # Expand to 4x dimension
        self.fc2 = nn.Linear(mlp_dim, embed_dim, bias=bias)  # Project back
        self.dropout = nn.Dropout(dropout)
        # 原版 ViT 使用 tanh 近似的 GELU：nn.GELU(approximate='tanh')
        # 此处使用精确 GELU，两者差异很小，但教学中值得了解
        self.gelu = nn.GELU()  # Activation function
        
    def forward(self, x):
        x = self.fc1(x)  # (B, 197, 768) -> (B, 197, 3072)
        x = self.gelu(x)
        # x = self.dropout(x)
        x = self.fc2(x)  # (B, 197, 3072) -> (B, 197, 768)
        x = self.dropout(x)
        return x



class TransformerEncoderLayer(nn.Module):
    """
    Single layer of Transformer encoder
    Input: (B, N, D)
    Output: (B, N, D)
    """
    def __init__(self, embed_dim=768, num_heads=12, mlp_dim=3072, dropout=0.1, bias=False):
        """
        单个 Transformer Encoder 层，采用 Pre-Norm 架构（LayerNorm 在子层之前）。
        这是 ViT 论文的关键设计选择，相比 Post-Norm 能提供更稳定的梯度，加速收敛。
        
        数据流：
            x -> LN -> MHSA -> + (残差) -> LN -> MLP -> + (残差) -> output
        
        注意：原版 ViT 在每个残差连接路径上还使用了 DropPath（随机深度），
        即在 MHSA/MLP 输出上加 DropPath 后再与输入相加。此处简化未实现。
        """
        super().__init__()
        # Layer normalization before MHSA (Pre-Norm)
        # 注意：LayerNorm 始终保留 bias（β），不随外部 bias 参数变化
        # 去掉 β 会导致 LN 输出强制零均值，严重限制模型的表达能力
        self.ln1 = nn.LayerNorm(embed_dim)
        # Multi-Head Self-Attention
        self.mhsa = MultiHeadSelfAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, bias=bias)
        # Layer normalization before MLP (Pre-Norm)
        self.ln2 = nn.LayerNorm(embed_dim)
        # MLP block
        self.mlp = MLP(embed_dim=embed_dim, mlp_dim=mlp_dim, dropout=dropout, bias=bias)
        
    def forward(self, x):
        # Pre-Norm + 残差连接：先归一化，再过 MHSA，最后加回原始输入
        x = x + self.mhsa(self.ln1(x))  # (B, 197, 768)
        # Pre-Norm + 残差连接：先归一化，再过 MLP，最后加回原始输入
        x = x + self.mlp(self.ln2(x))  # (B, 197, 768)
        return x



class VisionTransformer(nn.Module):
    """
    Full Vision Transformer model for image classification
    Input: (B, C, H, W)
    Output: (B, num_classes)
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_ch=3,
        embed_dim=768,
        num_heads=12,
        num_layers=12,
        mlp_dim=3072,
        num_classes=1000,
        dropout=0.1, 
        bias=False):
        """
        完整的 Vision Transformer 模型，对应 ViT-Base 配置。
        
        :param img_size: 输入图像大小（默认224，ViT 论文标准尺寸）
        :param patch_size: 每个 patch 的大小（默认16，ViT-Base 配置）
        :param in_ch: 输入通道数（RGB=3）
        :param embed_dim: Token 嵌入维度 D（默认768，ViT-Base）
        :param num_heads: 多头注意力头数（默认12，ViT-Base；head_dim = 768/12 = 64）
        :param num_layers: Transformer Encoder 层数（默认12，ViT-Base）
        :param mlp_dim: MLP 隐藏层维度（默认 4*D = 3072）
        :param num_classes: 分类类别数（默认1000，对应 ImageNet）
        :param dropout: Dropout 概率
        """
        super().__init__()
        # ---- Step 1: Patch Embedding（图像 -> patch tokens）----
        # PatchEmbedding 只负责将图像切分为 patch 并投影到 embed_dim 维度
        # 不包含 CLS token 和位置编码，这两个是独立的概念
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_ch, embed_dim)
        num_patches = self.patch_embed.num_patches  # N = (H/P) * (W/P) = 196
        
        # ---- Step 2: CLS token（全局聚合向量）----
        # 一个可学习的特殊向量，拼接在 patch tokens 序列的最前面
        # 经过 Transformer 编码后，它汇聚了全局信息，用于最终分类
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # ---- Step 3: 位置编码（让模型知道每个 patch 在图像中的位置）----
        # 没有位置编码，模型无法区分"左上角的猫"和"右下角的猫"
        # 维度为 (1, N+1, D)，+1 是为 CLS token 预留的位置
        # 原版 ViT 使用可学习的位置编码（std=0.02 初始化）；
        # 也可用固定的正弦/余弦编码（原始 Transformer），或 MAE 中的 2D 正弦编码
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim) * 0.02)
        
        self.pos_drop = nn.Dropout(dropout)
        
        # ---- Step 4: Transformer Encoder（专心负责序列编码）----
        # 数据在传入 Encoder 之前已完成所有预处理（patch切分 + CLS拼接 + 位置编码）
        # Encoder 只需专注于 token 之间的注意力交互和特征提取
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim, dropout=dropout, bias=bias)
            for _ in range(num_layers)
        ])
        
        # ---- Step 5: 输出归一化 + 分类头 ----
        # Pre-Norm 架构的必要组件：所有 Encoder 层结束后，对最终输出做一次全局 LN
        self.ln = nn.LayerNorm(embed_dim)
        
        # 教学简化：原版 ViT 论文的分类头是两层 MLP：
        #   Linear(D, D) -> GELU -> Linear(D, num_classes)
        # 这里简化为单层线性层，对于大多数任务效果相近
        self.classifier = nn.Linear(embed_dim, num_classes, bias=bias)
        
        # Initialize weights
        # 注意：self.apply() 只处理 nn.Module（Linear/Conv2d/LayerNorm），
        # 不会影响 nn.Parameter 类型的 cls_token 和 pos_embed，
        # 因此它们保留上面 randn*0.02 的初始化
        self.apply(self._init_weights)
        # 分类头零初始化：原版 ViT 对分类头使用零初始化或更小的 std，
        # 避免训练初期分类头输出过大导致梯度不稳定
        nn.init.zeros_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)
        
    def _init_weights(self, m):
        """
        权重初始化策略。
        原版 ViT 论文使用截断正态分布（std=0.02）初始化所有 Linear 和 Conv 层。
        此处使用 xavier_uniform_ 作为教学简化。
        """
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        B = x.shape[0]
        
        # Step 1: Patch Embedding —— 图像切分为 patch tokens (B, C, H, W) -> (B, N, D)
        x = self.patch_embed(x)  # (B, 196, 768)
        
        # Step 2: 拼接 CLS token —— 在序列头部添加一个可学习的全局聚合向量
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, 768)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 197, 768)
        
        # Step 3: 加上位置编码 —— 让模型知道每个 token 对应的空间位置
        x = x + self.pos_embed  # (B, 197, 768)
        x = self.pos_drop(x)
        
        # Step 4: Transformer Encoder —— 12 层 Pre-Norm Encoder（注意力 + FFN + 残差）
        for layer in self.encoder_layers:
            x = layer(x)  # (B, 197, 768)
        
        # Step 5: 输出归一化（Pre-Norm 架构的最后一步）
        x = self.ln(x)  # (B, 197, 768)
        
        # Step 6: 提取 CLS token 特征 -> 分类头 (B, num_classes)
        cls_output = x[:, 0, :]  # (B, 768)
        logits = self.classifier(cls_output)  # (B, 1000)
        
        return logits




if __name__ == "__main__":

    # fig = visualize_position_embedding_similarity()
    # fig.savefig("fig3_2_pos_embed.png", dpi=300)

    # Configuration (adjust based on server resources)
    IMG_SIZE = 224
    PATCH_SIZE = 16
    EMBED_DIM = 768
    NUM_HEADS = 12
    NUM_LAYERS = 12
    MLP_DIM = 3072

    BATCH_SIZE = 64
    EPOCHS = 20
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5  # Regularization to prevent overfitting

    # model = SimpleViT(image_size=IMG_SIZE, patch_size=PATCH_SIZE)
    model = VisionTransformer(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        mlp_dim=MLP_DIM,
        num_classes=1000,
        dropout=0.1
    )

    input_size=(1, 3, IMG_SIZE, IMG_SIZE)
    dummy_img = torch.randn(input_size)
    # print(f"输入尺寸: {dummy_img.shape}") # torch.Size([1, 3, 224, 224])
    # print(f"输出尺寸: {model(dummy_img).shape}") # torch.Size([1, 1000])
    
    summary(model, input_size=input_size)
