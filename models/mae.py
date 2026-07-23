
from functools import partial
import math

import numpy as np
import torch
import torch.nn as nn



def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb



def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed



class PatchEmbed_bak(nn.Module):
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



class Block(nn.Module):
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
        # LayerNorm 始终保留 bias（β），不随外部 bias 参数变化
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



class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, bias=False, dropout=0.1):
        super().__init__()
        
        # 根据 mlp_ratio 自动计算 encoder 和 decoder 的 MLP 隐藏层维度
        encoder_mlp_dim = int(embed_dim * mlp_ratio)          # e.g. 768*4=3072, 1024*4=4096
        decoder_mlp_dim = int(decoder_embed_dim * mlp_ratio)  # e.g. 512*4=2048

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed_bak(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

        self.blocks = nn.ModuleList([
            Block(embed_dim=embed_dim, num_heads=num_heads, mlp_dim=encoder_mlp_dim, dropout=dropout, bias=bias)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_dim=decoder_mlp_dim, dropout=dropout, bias=bias)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        # 注意：PatchEmbed_bak 中卷积层的属性名是 proj
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        # self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        # PatchEmbed_bak 中 patch_size 是 int 类型，直接取值
        p = self.patch_embed.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x) 

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


def mae_vit_base_patch16(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model



if __name__ == '__main__':
    from torchinfo import summary
    input_size = (1, 3, 224, 224)
    dummy_input = torch.rand(input_size)
    model = mae_vit_base_patch16()
    # model(dummy_input)
    summary(model, input_data=dummy_input)
