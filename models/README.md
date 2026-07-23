# Models — 视觉神经网络教学案例集

本文件夹包含一系列经典视觉神经网络的 PyTorch 实现，面向初学者展示从模型定义到训练流程搭建的核心内容。模型按照发展脉络组织，从 CNN 奠基到 Transformer 时代，形成完整的学习路径。

---

## 模块一：CNN 的奠基之路 — 从 LeNet 到 ResNet

沿着 CNN 发展史上的里程碑模型，逐步展示网络设计的演进思路。

| 模型 | 文件 | 核心要点 |
|------|------|----------|
| **LeNet-5** | `lenet.py` | CNN 的开山之作（1998），用于手写数字识别。展示了卷积 + 池化 + 全连接的经典范式，结构简洁，适合入门。 |
| **AlexNet** | `alexnet.py` | 2012 年 ImageNet 冠军，首次证明了深度学习在大规模视觉任务上的威力。引入了 ReLU 激活、Dropout、GPU 训练等关键技术。 |
| **VGG** | `vgg.py` | 2014 年 ImageNet 亚军，用连续的 3×3 小卷积核替代大卷积核，证明了"更深更规整"的网络结构能有效提升性能。包含 VGG11/13/16/19 多种配置，通过 cfg 字典统一管理。 |
| **GoogLeNet** | `googlenet.py` | 2014 年 ImageNet 冠军（与 VGG 同年），提出了 Inception 模块——在同一层并行使用多种尺度的卷积核（1×1、3×3、5×5）和池化，让网络自适应地选择最优的局部特征提取方式。还引入了辅助分类头和 1×1 瓶颈降维。 |
| **ResNet** | `resnet.py` | 2015 年 ImageNet 冠军，提出**残差连接（Skip Connection）**，解决了深层网络的退化问题。实现了 BasicBlock 和 Bottleneck 两种残差块，支持 ResNet18/34/50/101/152 多种变体。是后续几乎所有深度网络的基石。 |
| **UNet** | `unet.py` | 2015 年提出的语义分割经典架构，采用对称的编码器-解码器结构，通过**跳跃连接**将浅层细节与深层语义融合。包含标准 UNet 以及以 ResNet18、MobileNetV2 为编码器的变体版本。 |

---

## 模块二：面向特性与工业应用的网络设计

在网络结构创新或工业落地方面做出重要贡献的模型。

| 模型 | 文件 | 核心要点 |
|------|------|----------|
| **FCN** | `fcn.py` | 全卷积网络，**语义分割领域的开创性工作**。将全连接层替换为卷积层和转置卷积，使网络输出像素级预测图。实现了 FCN-32s/16s/8s 三种渐进融合方案及 SimpleFCN 教学版本。 |
| **DenseNet** | `densenet.py` | 密集连接卷积网络，每层都与前面所有层直接相连，实现**特征复用最大化**和极强的梯度流。通过 Growth Rate 控制通道增长，Transition 层负责下采样和通道压缩。 |
| **MobileNetV2** | `mobilenetv2.py` | 面向移动端的轻量化网络。核心贡献：**倒残差结构**（先升维再降维）和**线性瓶颈**（瓶颈处去掉 ReLU 避免信息损失），深度可分离卷积大幅降低计算量。 |
| **MobileNetV3** | `mobilenetv3.py` | 在 V2 基础上引入三大升级：**SE 注意力模块**（通道注意力）、**h-swish 激活函数**（硬件友好）、**NAS 神经架构搜索**。分为 Large 和 Small 两个版本。 |
| **DeepLabV3+** | `deeplab3plus.py` | 工业级语义分割方案，Encoder-Neck-Decoder 三段式架构：backbone 提取特征 → **ASPP** 捕获多尺度上下文 → Decoder 融合低层细节恢复空间信息。 |

---

## 模块三：CNN 的现代化收尾 — ConvNeXt

| 模型 | 文件 | 核心要点 |
|------|------|----------|
| **ConvNeXt** (torchvision) | `convnext.py` | 基于 torchvision 官方 API 的实现，支持加载预训练权重。适合快速使用和迁移学习。 |
| **ConvNeXt V1/V2** (facebookresearch) | `convnext_official.py` | 基于 Meta 官方仓库的教学复现V1 和 V2：V1 使用 Layer Scale，V2 用 **GRN（Global Response Normalization）** 替代。同时包含 **FCMAE（全卷积遮罩自编码器）** 实现，展示纯卷积网络如何适配 MAE 自监督预训练框架。 |

ConvNeXt 系统性地借鉴 Transformer 设计（LayerNorm、GELU、7×7 大卷积核、Stochastic Depth 等），对纯卷积网络进行现代化改造，在 ImageNet 上达到与 Swin Transformer 相当的精度。作为 CNN 系列的收尾，为切入 Transformer 时代做铺垫。

---

## 模块四：Transformer 时代 — ViT 与 MAE

| 模型 | 文件 | 核心要点 |
|------|------|----------|
| **ViT** | `vit.py` | Vision Transformer，将图像切分为 patch 序列输入 Transformer Encoder。基于 torchvision 官方实现，支持 ViT-B/16、B/32、L/16、L/32、H/14 多种配置及预训练权重。 |
| **MAE** | `mae.py` | Masked Autoencoder（2021），自监督预训练的代表性工作。随机遮住 75% 的 patch，Encoder 只处理可见部分（省 75% 计算），Decoder 重建被遮区域的像素值。包含完整的 ViT 组件教学实现（PatchEmbed、MHSA、MLP、Block）。 |

**配套 Demo 脚本**（位于项目根目录）：
- `mae_demo.py`：MAE 自监督预训练 → 权重提取 → 分类微调完整流水线
- `convnextv2_demo.py`：FCMAE 预训练 → 微调，含与 ViT-MAE 的对比分析

---

## 目录结构

```
models/
├── backbone/               # 编码器（特征提取器）
│   ├── alexnet.py          #   AlexNet Encoder
│   ├── vgg.py              #   VGG16 / VGG19 Encoder
│   ├── resnet.py           #   ResNet18 / ResNet50 Encoder
│   ├── mobilenet.py        #   MobileNetV2 / V3 Encoder
│   └── densenet.py         #   DenseNet121 / DenseNet169 Encoder
├── neck/                   # 颈部模块（多尺度特征增强）
│   └── aspp.py             #   ASPP（空洞空间金字塔池化）
├── block/                  # 可复用的网络组件
│   ├── basic.py            #   基础模块
│   └── attention.py        #   注意力模块
├── head/                   # 任务头
│   └── seg_head.py         #   语义分割头
├── utils/                  # 工具函数
│   ├── pytorch_api.py      #   PyTorch/torchvision API 辅助
│   ├── convnext_api.py     #   ConvNeXt 组件（GRN, LayerNorm, PatchMerging）
│   └── utils.py            #   通用工具（DropPath 等）
│
│  ── 模块一：CNN 奠基 ──
├── lenet.py                # LeNet-5
├── alexnet.py              # AlexNet
├── vgg.py                  # VGG (11/13/16/19)
├── googlenet.py            # GoogLeNet (Inception V1)
├── resnet.py               # ResNet (18/34/50/101/152)
├── unet.py                 # UNet (标准 / ResNet18 / MobileNetV2)
│
│  ── 模块二：特性与工业应用 ──
├── fcn.py                  # FCN (32s / 16s / 8s / SimpleFCN)
├── densenet.py             # DenseNet (121/161/169/201)
├── mobilenetv2.py          # MobileNetV2
├── mobilenetv3.py          # MobileNetV3 (Large / Small)
├── deeplab3plus.py         # DeepLabV3+
│
│  ── 模块三：CNN 现代化收尾 ──
├── convnext.py             # ConvNeXt (torchvision 版本，支持预训练权重)
├── convnext_official.py    # ConvNeXt V1/V2 统一实现 + FCMAE (facebookresearch 复现)
│
│  ── 模块四：Transformer 时代 ──
├── vit.py                  # Vision Transformer (torchvision 版本)
└── mae.py                  # Masked Autoencoder (教学复现)
```

---

## 设计说明

- **模块化架构**：`backbone`、`neck`、`head` 的分离设计，体现现代视觉系统的模块化思想，便于灵活组合（如 UNet 可替换不同 encoder，DeepLabV3+ 可搭配不同 backbone）。
- **双版本 ConvNeXt**：`convnext.py` 面向快速使用（torchvision 预训练权重），`convnext_official.py` 面向教学理解（逐行复现官方实现，含 V1/V2 对比和 FCMAE 自监督）。
- **从 CNN 到 Transformer 的过渡**：ConvNeXt 展示了 CNN 如何借鉴 Transformer 设计思想，ViT/MAE 则正式进入 Transformer 范式，形成自然的知识衔接。
- **教学注释**：代码中附有详细的中文注释，解释每一层的作用和维度变化，帮助初学者理解数据在网络中的流动过程。
- **独立运行**：每个文件底部都有 `if __name__ == "__main__"` 测试代码，可使用 `torchinfo.summary()` 查看模型结构和参数量。
