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

标准训练范式：
1. 在大规模无标签数据上用 FCMAE 自监督预训练
2. 丢弃 Decoder，保留 Encoder
3. 在下游任务上微调 Encoder + 新分类头

ref: https://arxiv.org/abs/2301.00808
"""

import torch
import torch.nn as nn
import torch.optim as optim

from models.convnext_official import convnextv2_tiny, convnextv2_atto, fcmae_convnextv2_tiny


# ======================== 1. FCMAE 自监督预训练 ========================

def demo_fcmae_pretrain(num_epochs=20, batch_size=4, device='cpu'):
    """
    FCMAE 自监督预训练 Demo

    演示 FCMAE_ConvNeXtV2 的完整训练流程：
    1. 构建 FCMAE 模型（Encoder + Decoder）
    2. 在随机数据上进行自监督预训练（重建被遮住的 stem 特征）
    3. 返回预训练好的模型（后续提取 Encoder 权重）

    关键点：不需要任何标签。模型通过遮住 60% 的 patch 并尝试重建它们，
    自行创建监督信号。
    """
    print("=" * 60)
    print("Step 1: FCMAE 自监督预训练")
    print("=" * 60)

    # ---- 构建 FCMAE 模型 ----
    model = fcmae_convnextv2_tiny(img_size=64).to(device)

    enc_params = sum(p.numel() for p in model.encoder.parameters()) / 1e6
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    dec_params = total_params - enc_params

    print(f"\n模型参数量:")
    print(f"  Encoder: {enc_params:.2f}M")
    print(f"  Decoder: {dec_params:.2f}M")
    print(f"  总计:    {total_params:.2f}M")
    print(f"\n配置: img_size=64, mask_ratio=0.6, stem输出: 16x16")
    print(f"模型只看到 40% 的 patch，必须通过上下文推理重建剩余 60%")
    print()

    # ---- 优化器 ----
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)

    # ---- 自监督训练循环 ----
    print("--- 自监督预训练（无标签）---")
    model.train()
    for epoch in range(num_epochs):
        # 模拟无标签图像数据（实际场景使用大规模无标签数据集）
        imgs = torch.randn(batch_size, 3, 64, 64, device=device)

        # 前向传播：遮罩 → 编码 → 解码 → 计算损失
        loss, pred, mask = model(imgs)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            mask_ratio_actual = mask.mean().item()
            print(f"  Epoch {epoch+1:2d}/{num_epochs}  "
                  f"loss={loss.item():.4f}  "
                  f"mask_ratio={mask_ratio_actual:.0%}  "
                  f"pred_shape={tuple(pred.shape)}")

    print()
    print("预训练完成！Encoder 已学会理解图像特征。")
    print("下一步：丢弃 Decoder，保留 Encoder 用于下游任务。")
    print()

    return model


# ======================== 2. 权重提取与监督微调 ========================

def demo_finetune(fcmae_model, num_classes=10, num_epochs=10, batch_size=4):
    """
    微调 Demo：从 FCMAE 预训练的 Encoder 提取权重，加载到分类模型

    流程：
    1. 从 FCMAE 提取 Encoder 权重（过滤 head）
    2. 构建 ConvNeXtV2 分类模型
    3. load_state_dict(strict=False) 加载预训练权重
    4. 差异化学习率微调（backbone 小 LR，head 大 LR）
    5. 推理演示
    """
    device = next(fcmae_model.parameters()).device
    print("=" * 60)
    print("Step 2: 权重提取与监督微调")
    print("=" * 60)

    # ---- Step 2.1: 提取 Encoder 权重 ----
    encoder_state_dict = fcmae_model.encoder.state_dict()
    # FCMAE 的 encoder 使用 num_classes=0，其 head 权重 shape 为 [0, dim]，
    # 与分类模型的 head shape 不匹配，需要手动过滤掉 head 相关的 key
    encoder_state_dict = {k: v for k, v in encoder_state_dict.items() if not k.startswith('head.')}
    print(f"\n从 FCMAE 提取 Encoder 权重: {len(encoder_state_dict)} 个参数张量（已排除 head）")

    # ---- Step 2.2: 构建分类模型（与 FCMAE Encoder 配置一致）----
    classifier = convnextv2_tiny(num_classes=num_classes).to(device)

    # ---- Step 2.3: 加载预训练权重 ----
    # strict=False 跳过不匹配的 key（新增的分类头）
    missing, unexpected = classifier.load_state_dict(encoder_state_dict, strict=False)
    print(f"\n权重加载结果:")
    print(f"  缺少（新增的分类头）: {[k for k in missing if 'head' in k]}")
    print(f"  多余（FCMAE独有）:   {unexpected}")

    # ---- Step 2.4: 差异化学习率微调 ----
    # 分类头用较大学习率（快速适应），Encoder 用较小学习率（保留预训练知识）
    head_params = [p for n, p in classifier.named_parameters() if 'head' in n]
    backbone_params = [p for n, p in classifier.named_parameters() if 'head' not in n]
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': 1e-5},  # Encoder: 小学习率
        {'params': head_params,     'lr': 1e-3},  # 分类头: 大学习率
    ], weight_decay=0.05)
    criterion = nn.CrossEntropyLoss()

    print(f"\n--- 监督微调 ({num_classes} 类) ---")
    print(f"  策略: backbone lr=1e-5, head lr=1e-3")
    classifier.train()

    for epoch in range(num_epochs):
        # 模拟带标签的训练数据（实际场景使用目标数据集）
        imgs = torch.randn(batch_size, 3, 64, 64, device=device)
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

    # ---- Step 2.5: 推理演示 ----
    classifier.eval()
    with torch.no_grad():
        test_imgs = torch.randn(2, 3, 64, 64, device=device)
        test_logits = classifier(test_imgs)
        test_preds = test_logits.argmax(dim=1)
        print(f"\n推理示例:")
        print(f"  输入: {tuple(test_imgs.shape)}")
        print(f"  预测类别: {test_preds.tolist()}")
        print(f"  输出 logits: {test_logits.shape}")
    print()

    return classifier


# ======================== 3. 从零训练 vs 预训练对比 ========================

def compare_training(device='cpu'):
    """
    对比实验：从零训练 vs FCMAE 预训练后微调

    展示预训练带来的收敛速度优势。
    在真实场景中，FCMAE 预训练模型收敛更快、精度更高，尤其在标签有限时。
    """
    print("=" * 60)
    print("Step 3: 从零训练 vs FCMAE 预训练对比")
    print("=" * 60)

    criterion = nn.CrossEntropyLoss()
    imgs = torch.randn(8, 3, 64, 64, device=device)
    labels = torch.randint(0, 10, (8,), device=device)

    # --- 从零训练 ---
    model_scratch = convnextv2_tiny(num_classes=10).to(device)
    opt_scratch = optim.AdamW(model_scratch.parameters(), lr=1e-4)

    model_scratch.train()
    losses_scratch = []
    for _ in range(10):
        logits = model_scratch(imgs)
        loss = criterion(logits, labels)
        opt_scratch.zero_grad()
        loss.backward()
        opt_scratch.step()
        losses_scratch.append(loss.item())

    # --- FCMAE 预训练后微调 ---
    fcmae = fcmae_convnextv2_tiny(img_size=64).to(device)
    opt_fcmae = optim.AdamW(fcmae.parameters(), lr=1e-4)  # 修复：优化器在循环外创建

    # 模拟预训练（实际应在大规模无标签数据上训练数百 epoch）
    fcmae.train()
    for _ in range(10):
        fake_imgs = torch.randn(8, 3, 64, 64, device=device)
        loss, _, _ = fcmae(fake_imgs)
        opt_fcmae.zero_grad()
        loss.backward()
        opt_fcmae.step()

    # 提取权重 → 构建分类模型 → 微调
    encoder_sd = fcmae.encoder.state_dict()
    encoder_sd = {k: v for k, v in encoder_sd.items() if not k.startswith('head.')}
    model_pretrained = convnextv2_tiny(num_classes=10).to(device)
    model_pretrained.load_state_dict(encoder_sd, strict=False)

    opt_pt = optim.AdamW(model_pretrained.parameters(), lr=1e-5)
    model_pretrained.train()
    losses_pt = []
    for _ in range(10):
        logits = model_pretrained(imgs)
        loss = criterion(logits, labels)
        opt_pt.zero_grad()
        loss.backward()
        opt_pt.step()
        losses_pt.append(loss.item())

    print(f"\n  从零训练:    loss {losses_scratch[0]:.4f} → {losses_scratch[-1]:.4f}")
    print(f"  FCMAE预训练: loss {losses_pt[0]:.4f} → {losses_pt[-1]:.4f}")
    print()
    print("  （在真实场景中，FCMAE 预训练模型收敛更快、精度更高，")
    print("    尤其在标注数据有限时优势明显）")
    print()


# ======================== 4. 架构对比 ========================

def show_architecture_comparison():
    """
    对比 ConvNeXt V2 不同规模变体，以及与 ViT/MAE 的架构差异。
    """
    print("=" * 60)
    print("Step 4: 架构对比")
    print("=" * 60)

    models = {
        "ConvNeXtV2-Atto": convnextv2_atto(num_classes=10),
        "ConvNeXtV2-Tiny": convnextv2_tiny(num_classes=10),
    }

    x = torch.randn(1, 3, 64, 64)

    print(f"\n  {'模型':<20s} {'参数量':>10s} {'输出形状':>14s}")
    print(f"  {'-'*20} {'-'*10} {'-'*14}")

    for name, model in models.items():
        params = sum(p.numel() for p in model.parameters()) / 1e6
        out = model(x)
        print(f"  {name:<20s} {params:>8.1f}M {str(list(out.shape)):>14s}")

    print()
    print("  与 ViT/MAE 的关键区别:")
    print("  - 纯卷积（无注意力机制）→ 小图像上更高效")
    print("  - 层级特征（多尺度输出）→ 更适合检测/分割")
    print("  - GRN 层使纯卷积网络也能有效进行 MAE 式预训练")
    print("  - 遮罩比例 60%（vs MAE 的 75%）→ 卷积需要更多上下文")
    print()


# ======================== 5. 完整流水线 ========================

def demo_full_pipeline():
    """
    完整流水线：FCMAE 自监督预训练 → 权重提取 → 分类微调 → 对比实验

    这是 ConvNeXt V2 论文的标准训练范式。
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print()

    # 架构总览
    show_architecture_comparison()

    # Phase 1: 自监督预训练
    fcmae_model = demo_fcmae_pretrain(num_epochs=20, batch_size=4, device=device)

    # Phase 2: 权重提取与微调
    demo_finetune(fcmae_model, num_classes=10, num_epochs=10, batch_size=4)

    # Phase 3: 对比实验
    compare_training(device=device)

    # 总结
    print("=" * 60)
    print("总结")
    print("=" * 60)
    print("""
    ConvNeXt V2 = ConvNeXt V1 + GRN + FCMAE

    核心创新:
    1. GRN (Global Response Normalization):
       - 为纯卷积网络引入通道间竞争机制
       - 初始化为 0，不影响早期训练
       - 使卷积网络在 MAE 框架下也能有效预训练

    2. FCMAE (Fully Convolutional Masked Autoencoder):
       - 在 stem 特征图层面遮住 60% 的 patch (H/4 x W/4)
       - Encoder 通过 ConvNeXt V2 处理可见 patch
       - Decoder 使用转置卷积上采样并预测被遮区域
       - Loss 仅在被遮住的 patch 上计算（MSE on stem features）

    3. 与 ViT-MAE 的对比:
       - Encoder: Transformer blocks → ConvNeXt V2 blocks
       - Decoder: Transformer blocks → 转置卷积
       - 遮罩比例: 75% → 60%（卷积需要更多上下文）
       - 预测目标: 像素值 → stem 特征图值

    训练范式:
       无标签数据 → FCMAE 预训练 → 丢弃 Decoder → 微调 Encoder + 分类头
    """)


if __name__ == '__main__':
    demo_full_pipeline()
