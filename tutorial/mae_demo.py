"""
MAE (Masked Autoencoder) 训练与推理示例

MAE 的核心思路：
1. 预训练阶段：随机遮住 75% 的 patch，让模型从剩余 25% 重建原图
   → 这不是为了"生成"图像，而是让 Encoder 学会理解图像的通用表示
2. 下游任务：丢弃 Decoder，用预训练好的 Encoder + 分类头做微调
   → Encoder 已经学到了丰富的视觉特征，微调效率远高于从头训练

与 ConvNeXt V2 FCMAE 的对比：
- MAE Encoder: ViT (Transformer) → FCMAE Encoder: ConvNeXt V2 (纯卷积)
- MAE Decoder: Transformer blocks → FCMAE Decoder: 转置卷积
- MAE 遮罩 75% → FCMAE 遮罩 60%（卷积需要更多上下文）
- MAE 预测原图像素 → FCMAE 预测 stem 特征图

典型应用场景：
- 医学影像（标注数据极少且昂贵）
- 遥感图像（大量无标注卫星图）
- 工业质检（正常样本多，异常样本少）

ref: https://arxiv.org/abs/2111.06377
"""

import torch
import torch.nn as nn
import torch.optim as optim

from models.mae import mae_vit_base_patch16


# ======================== 1. MAE 自监督预训练 ========================

def demo_mae_pretrain(num_steps=20, batch_size=4, device='cpu'):
    """
    MAE 自监督预训练 Demo

    演示 MaskedAutoencoderViT 的完整预训练流程：
    1. 构建 MAE 模型（ViT Encoder + Transformer Decoder）
    2. 随机遮住 75% 的 patch
    3. Encoder 只处理可见的 25% patch（节省计算）
    4. Decoder 重建被遮住 patch 的像素值
    5. Loss 仅在被遮住的 patch 上计算

    关键点：不需要任何标签！模型通过高遮罩率被迫学习高层语义。
    """
    print("=" * 60)
    print("Step 1: MAE 自监督预训练")
    print("=" * 60)

    model = mae_vit_base_patch16().to(device)

    # 参数量统计（完整 Encoder = patch_embed + pos_embed + cls_token + blocks + norm）
    encoder_params = (
        sum(p.numel() for p in model.patch_embed.parameters()) +
        model.pos_embed.numel() +
        model.cls_token.numel() +
        sum(p.numel() for p in model.blocks.parameters()) +
        sum(p.numel() for p in model.norm.parameters())
    ) / 1e6
    decoder_params = (
        sum(p.numel() for p in model.decoder_embed.parameters()) +
        model.mask_token.numel() +
        model.decoder_pos_embed.numel() +
        sum(p.numel() for p in model.decoder_blocks.parameters()) +
        sum(p.numel() for p in model.decoder_norm.parameters()) +
        sum(p.numel() for p in model.decoder_pred.parameters())
    ) / 1e6
    total_params = sum(p.numel() for p in model.parameters()) / 1e6

    print(f"\n模型参数量:")
    print(f"  Encoder (ViT-Base): {encoder_params:.1f}M")
    print(f"  Decoder (8-layer):  {decoder_params:.1f}M")
    print(f"  总计:               {total_params:.1f}M")
    print(f"\n配置: img_size=224, patch_size=16, mask_ratio=0.75")
    print(f"  - 196 个 patch 中遮住 147 个，Encoder 只处理 49 个")
    print(f"  - 重建目标: 每个 patch 的像素值 (16×16×3 = 768 维)")
    print()

    # ---- 自监督训练循环 ----
    optimizer = optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=0.05)

    print("--- 自监督预训练（无标签）---")
    model.train()
    for step in range(num_steps):
        # 模拟无标签图像数据（实际场景使用 ImageNet-1K 等大规模数据集）
        imgs = torch.randn(batch_size, 3, 224, 224, device=device)

        # 前向传播：遮罩 → 编码（仅可见patch）→ 解码 → 计算重建损失
        loss, pred, mask = model(imgs, mask_ratio=0.75)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 5 == 0 or step == 0:
            print(f"  Step {step+1:2d}/{num_steps}  "
                  f"loss={loss.item():.4f}  "
                  f"pred_shape={tuple(pred.shape)}  "
                  f"mask_shape={tuple(mask.shape)}")

    print()
    print("预训练完成！")
    print("  pred: (B, 196, 768) → 每个 patch 重建的像素值 (patch_size²×3)")
    print("  mask: (B, 196) → 1=被遮住, 0=可见")
    print()
    print("下一步：丢弃 Decoder，提取 Encoder 权重用于下游任务。")
    print()

    return model


# ======================== 2. 重建可视化（可选）========================

@torch.no_grad()
def visualize_reconstruction(model, img, mask_ratio=0.75):
    """
    可视化 MAE 的重建效果：展示原图、遮罩、重建结果。

    这帮助我们直观理解 MAE 在预训练阶段到底在做什么：
    - 只看 25% 的 patch，就能重建出完整图像的大致结构
    - 说明 Encoder 学到了图像的高层语义理解

    参数:
        model: 预训练好的 MAE 模型
        img: 单张图像 (3, H, W)，值域 [0, 1]
        mask_ratio: 遮罩比例

    返回:
        fig (matplotlib Figure) 或 None（如果 matplotlib 不可用）
    """
    try:
        import numpy as np
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [跳过可视化] 需要 numpy 和 matplotlib: pip install numpy matplotlib")
        return None

    model.eval()
    device = next(model.parameters()).device
    img = img.to(device)

    # 前向传播
    loss, pred, mask = model(img.unsqueeze(0), mask_ratio=mask_ratio)

    # 重建图像（将预测的 patch 像素值还原为图像）
    pred_img = model.unpatchify(pred)  # (1, 3, H, W)
    pred_img = pred_img[0].cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)

    # 生成遮罩可视化
    orig_img = img.cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
    mask_img = orig_img.copy()

    patch_size = model.patch_embed.patch_size
    h = w = orig_img.shape[0] // patch_size
    mask_2d = mask[0].cpu().numpy().reshape(h, w)

    for i in range(h):
        for j in range(w):
            if mask_2d[i, j] == 1:  # 被遮住的 patch → 显示为灰色
                mask_img[i*patch_size:(i+1)*patch_size,
                         j*patch_size:(j+1)*patch_size] = 0.5

    # 对比：只在被遮区域叠加重建结果
    compare = orig_img.copy()
    for i in range(h):
        for j in range(w):
            if mask_2d[i, j] == 1:
                compare[i*patch_size:(i+1)*patch_size,
                        j*patch_size:(j+1)*patch_size] = \
                    pred_img[i*patch_size:(i+1)*patch_size,
                             j*patch_size:(j+1)*patch_size]

    # 绘图
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(np.clip(orig_img, 0, 1))
    axes[0].set_title("Original")
    axes[0].axis('off')

    axes[1].imshow(np.clip(mask_img, 0, 1))
    axes[1].set_title(f"Masked (keep {int((1-mask_ratio)*100)}%)")
    axes[1].axis('off')

    axes[2].imshow(np.clip(pred_img, 0, 1))
    axes[2].set_title("MAE Reconstruction")
    axes[2].axis('off')

    axes[3].imshow(np.clip(compare, 0, 1))
    axes[3].set_title("Orig + Recon of masked")
    axes[3].axis('off')

    plt.suptitle(f"MAE Reconstruction (Loss: {loss.item():.4f})", fontsize=14)
    plt.tight_layout()
    return fig


# ======================== 3. 权重提取与监督微调 ========================

def demo_finetune(pretrained_mae, num_classes=10, num_epochs=10, batch_size=4):
    """
    微调 Demo：从 MAE 预训练的 Encoder 提取权重，加载到 ViT 分类模型

    流程：
    1. 从 MAE 提取 Encoder 权重（patch_embed + blocks + norm，排除 Decoder）
    2. 构建 ViT 分类模型（复用 Encoder 结构 + 新分类头）
    3. load_state_dict(strict=False) 加载预训练权重
    4. 差异化学习率微调（backbone 小 LR，head 大 LR）

    与从零训练的关键区别：
    - Encoder 已经学到了通用视觉特征（边缘、纹理、物体部件等）
    - 只需要少量标注数据就能达到很好的效果
    - 训练更快、更稳定
    """
    device = next(pretrained_mae.parameters()).device
    print("=" * 60)
    print("Step 2: 权重提取与监督微调")
    print("=" * 60)

    # ---- Step 3.1: 提取 Encoder 权重 ----
    # MAE 的 state_dict 包含 encoder 和 decoder 的所有参数
    # 我们只需要 encoder 部分：patch_embed.*, cls_token, pos_embed, blocks.*, norm.*
    full_state_dict = pretrained_mae.state_dict()
    encoder_keys = [k for k in full_state_dict.keys()
                    if not k.startswith('decoder_') and k != 'mask_token']
    encoder_state_dict = {k: full_state_dict[k] for k in encoder_keys}
    print(f"\n从 MAE 提取 Encoder 权重: {len(encoder_state_dict)} 个参数张量")
    print(f"  排除: decoder_embed, decoder_blocks, decoder_norm, decoder_pred, mask_token")

    # ---- Step 3.2: 构建分类模型 ----
    # 复用 MAE 的 Encoder 结构，加上分类头
    classifier = MAEClassifier(num_classes=num_classes).to(device)

    # ---- Step 3.3: 加载预训练权重 ----
    missing, unexpected = classifier.load_state_dict(encoder_state_dict, strict=False)
    print(f"\n权重加载结果:")
    print(f"  缺少（新增的分类头）: {missing}")
    print(f"  多余: {unexpected}")

    # ---- Step 3.4: 差异化学习率微调 ----
    head_params = [p for n, p in classifier.named_parameters() if 'head' in n]
    backbone_params = [p for n, p in classifier.named_parameters() if 'head' not in n]
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': 1e-5},  # Encoder: 小学习率，保留预训练知识
        {'params': head_params,     'lr': 1e-3},  # 分类头: 大学习率，快速适应
    ], weight_decay=0.05)
    criterion = nn.CrossEntropyLoss()

    print(f"\n--- 监督微调 ({num_classes} 类) ---")
    print(f"  策略: backbone lr=1e-5, head lr=1e-3")

    classifier.train()
    for epoch in range(num_epochs):
        # 模拟带标签的训练数据（实际场景使用目标数据集）
        imgs = torch.randn(batch_size, 3, 224, 224, device=device)
        labels = torch.randint(0, num_classes, (batch_size,), device=device)

        logits = classifier(imgs)
        loss = criterion(logits, labels)
        acc = (logits.argmax(dim=1) == labels).float().mean().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:2d}/{num_epochs}  "
                  f"loss={loss.item():.4f}  "
                  f"acc={acc:.0%}")

    print()
    print("微调完成！")

    # ---- 推理演示 ----
    classifier.eval()
    with torch.no_grad():
        test_imgs = torch.randn(2, 3, 224, 224, device=device)
        test_logits = classifier(test_imgs)
        test_preds = test_logits.argmax(dim=1)
        print(f"\n推理示例:")
        print(f"  输入: {tuple(test_imgs.shape)}")
        print(f"  预测类别: {test_preds.tolist()}")
        print(f"  输出 logits: {test_logits.shape}")
    print()

    return classifier


class MAEClassifier(nn.Module):
    """
    用预训练好的 MAE Encoder 做图像分类。

    核心思想：
    - 预训练好的 Encoder 已经理解了图像的"语义"
    - 只需要在上面加一个简单的分类头，用少量标注数据微调即可
    - 效果远好于从零训练的 ViT（尤其在标注数据少的时候）

    与 MAE 预训练时的区别：
    - 预训练：遮住 75% patch，Encoder 只处理 25%（省计算）
    - 微调：完整图像通过 Encoder，不遮罩（需要完整理解）

    Args:
        img_size (int): 输入图像大小. Default: 224
        patch_size (int): Patch 大小. Default: 16
        embed_dim (int): 嵌入维度. Default: 768
        depth (int): Transformer 层数. Default: 12
        num_heads (int): 注意力头数. Default: 12
        num_classes (int): 分类类别数. Default: 10
    """
    def __init__(self, img_size=224, patch_size=16, embed_dim=768,
                 depth=12, num_heads=12, num_classes=10):
        super().__init__()
        from models.mae import PatchEmbed_bak, Block
        from functools import partial

        # Encoder 结构（与 MAE 的 Encoder 完全一致）
        self.patch_embed = PatchEmbed_bak(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)

        self.blocks = nn.ModuleList([
            Block(embed_dim=embed_dim, num_heads=num_heads,
                  mlp_dim=int(embed_dim * 4), dropout=0.1, bias=False)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        # 分类头：原版 MAE 论文使用单层 Linear
        self.head = nn.Linear(embed_dim, num_classes)

        # 初始化位置编码（与 MAE 一致的正弦余弦编码）
        self._init_pos_embed()

    def _init_pos_embed(self):
        """初始化固定的正弦余弦位置编码"""
        from models.mae import get_2d_sincos_pos_embed
        import numpy as np

        num_patches = self.patch_embed.num_patches
        grid_size = int(num_patches ** 0.5)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], grid_size, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def forward(self, x):
        # Patch embedding: (B, 3, H, W) → (B, N, D)
        x = self.patch_embed(x)

        # 加位置编码（不含 CLS 位置）
        x = x + self.pos_embed[:, 1:, :]

        # 拼接 CLS token（加上 CLS 的位置编码）
        B = x.shape[0]
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, N+1, D)

        # Transformer Encoder（完整图像通过，不遮罩）
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # 取 CLS token 输出 → 分类
        cls_output = x[:, 0]  # (B, D)
        return self.head(cls_output)  # (B, num_classes)


# ======================== 4. 完整流水线 ========================

def demo_full_pipeline():
    """
    完整流水线：MAE 自监督预训练 → 权重提取 → 分类微调

    这是 MAE 论文的标准训练范式：
    1. 在大规模无标签数据上预训练（遮住 75%，重建像素）
    2. 丢弃 Decoder，保留 Encoder
    3. 在下游任务上微调 Encoder + 新分类头
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print()

    # Phase 1: 自监督预训练
    pretrained_mae = demo_mae_pretrain(num_steps=20, batch_size=4, device=device)

    # 可视化重建效果（可选，需要 matplotlib）
    print("=" * 60)
    print("重建可视化")
    print("=" * 60)
    test_img = torch.rand(3, 224, 224)  # 值域 [0,1] 的随机图像
    fig = visualize_reconstruction(pretrained_mae, test_img, mask_ratio=0.75)
    if fig is not None:
        fig.savefig("mae_reconstruction.png", dpi=150, bbox_inches='tight')
        print("  已保存到 mae_reconstruction.png")
    print()

    # Phase 2: 权重提取与微调
    demo_finetune(pretrained_mae, num_classes=10, num_epochs=10, batch_size=4)

    # 总结
    print("=" * 60)
    print("总结")
    print("=" * 60)
    print("""
    MAE 的价值不在于"生成图像"，而在于：

    1. 预训练不需要任何标签
       - 互联网上有海量无标注图片
       - 遮罩 75% 迫使 Encoder 理解高层语义（不是简单的像素插值）

    2. Encoder 只处理可见 patch（省 75% 计算）
       - 与 FCMAE 的区别：FCMAE 处理完整特征图（教学简化）
       - MAE 通过 random_masking 真正跳过被遮 patch 的计算

    3. 微调效率极高
       - 预训练 Encoder + 单层 Linear 分类头
       - 差异化 LR：backbone 1e-5 / head 1e-3
       - 少量标注数据即可达到优秀效果

    4. 与 ConvNeXt V2 FCMAE 的对比：
       - MAE: ViT Encoder + Transformer Decoder, 遮罩 75%, 预测像素
       - FCMAE: ConvNeXt V2 Encoder + 转置卷积 Decoder, 遮罩 60%, 预测特征图
       - 两者殊途同归：无标签预训练 → 高效微调
    """)


if __name__ == '__main__':
    demo_full_pipeline()
