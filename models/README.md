# Models — 卷积神经网络教学案例集

本文件夹包含一系列经典卷积神经网络（CNN）的 PyTorch 实现，面向初学者展示从模型定义到训练流程搭建的核心内容。这些模型按照 CNN 的发展脉络组织，分为三个递进的教学模块。

---

## 模块一：CNN 的奠基之路 — 从 LeNet 到 ResNet

这一模块沿着 CNN 发展史上的里程碑模型，逐步展示网络设计的演进思路。

| 模型 | 文件 | 核心要点 |
|------|------|----------|
| **LeNet-5** | `lenet.py` | CNN 的开山之作（1998），用于手写数字识别。展示了卷积 + 池化 + 全连接的经典范式，结构简洁，适合入门。 |
| **AlexNet** | `alexnet.py` | 2012 年 ImageNet 冠军，首次证明了深度学习在大规模视觉任务上的威力。引入了 ReLU 激活、Dropout、GPU 训练等关键技术。 |
| **VGG** | `vgg.py` | 2014 年 ImageNet 亚军，用连续的 3×3 小卷积核替代大卷积核，证明了"更深更规整"的网络结构能有效提升性能。包含 VGG11/13/16/19 多种配置，通过 cfg 字典统一管理。 |
| **GoogLeNet** | `googlenet.py` | 2014 年 ImageNet 冠军（与 VGG 同年），提出了 Inception 模块——在同一层并行使用多种尺度的卷积核（1×1、3×3、5×5）和池化，让网络自适应地选择最优的局部特征提取方式。还引入了辅助分类头和 1×1 瓶颈降维。 |
| **ResNet** | `resnet.py` | 2015 年 ImageNet 冠军，提出**残差连接（Skip Connection）**，解决了深层网络的退化问题。实现了 BasicBlock 和 Bottleneck 两种残差块，支持 ResNet18/34/50/101/152 多种变体。是后续几乎所有深度网络的基石。 |
| **UNet** | `unet.py` | 2015 年提出的语义分割经典架构，采用对称的编码器-解码器结构，通过**跳跃连接（Skip Connection）** 将浅层细节与深层语义融合。包含标准 UNet 以及以 ResNet18、MobileNetV2 为编码器的变体版本，展示了如何用不同 backbone 构建分割网络。 |

---

## 模块二：面向特性与工业应用的网络设计

这一模块介绍在网络结构创新或工业落地方面做出重要贡献的模型。

| 模型 | 文件 | 核心要点 |
|------|------|----------|
| **FCN** | `fcn.py` | 全卷积网络（Fully Convolutional Network），**语义分割领域的开创性工作**。核心思想是将分类网络中的全连接层替换为卷积层和转置卷积（上采样），使网络能输出像素级的预测图。实现了 FCN-32s/16s/8s 三种渐进式融合方案，以及一个简洁的 SimpleFCN 教学版本。 |
| **DenseNet** | `densenet.py` | 密集连接卷积网络，每个层都与前面所有层直接相连，实现了**特征复用最大化**和极强的梯度流。通过 Growth Rate 控制通道增长，Transition 层负责下采样和通道压缩。 |
| **MobileNetV2** | `mobilenetv2.py` | 面向移动端/嵌入式的轻量化网络。核心贡献：**倒残差结构**（先升维再降维）和**线性瓶颈**（瓶颈处去掉 ReLU 避免信息损失），以及深度可分离卷积大幅降低计算量。 |
| **MobileNetV3** | `mobilenetv3.py` | 在 V2 基础上引入三大升级：**SE 注意力模块**（通道注意力）、**h-swish 激活函数**（硬件友好）、以及 **NAS 神经架构搜索**自动设计网络结构。分为 Large 和 Small 两个版本。 |
| **DeepLabV3+** | `deeplab3plus.py` | 工业级语义分割方案，采用 Encoder-Neck-Decoder 三段式架构：backbone 提取特征 → **ASPP（空洞空间金字塔池化）** 捕获多尺度上下文 → Decoder 融合低层细节恢复空间信息。展示了模块化设计的工程实践。 |

---

## 模块三：走向 Transformer 时代 — ConvNeXt

| 模型 | 文件 | 核心要点 |
|------|------|----------|
| **ConvNeXt** | `convext.py` | 2022 年的工作，标题即"A ConvNet for the 2020s"。通过系统性地借鉴 Transformer 的设计思想（如 LayerNorm、GELU 激活、大卷积核 7×7、Stochastic Depth、Layer Scale 等），对纯卷积网络进行现代化改造，使其在 ImageNet 上达到与 Swin Transformer 相当的精度。作为 CNN 系列的收尾，为后续切入 Transformer 时代做好铺垫。 |

---

## 目录结构

```
models/
├── backbone/           # 编码器（特征提取器）定义
│   ├── vgg.py          #   VGG16 Encoder
│   ├── resnet.py       #   ResNet18 / ResNet50 Encoder
│   ├── mobilenet.py    #   MobileNetV2 Encoder
│   └── densenet.py     #   DenseNet121 / DenseNet169 Encoder
├── neck/               # 颈部模块（多尺度特征增强）
│   └── aspp.py         #   ASPP（空洞空间金字塔池化）
├── block/              # 可复用的网络组件/注意力模块
│   ├── basic.py
│   └── attention.py
├── head/               # 任务头（分割/检测等）
│   └── seg_head.py
├── utils/              # 工具函数
│   ├── pytorch_api.py  #   PyTorch API 辅助工具
│   └── utils.py        #   通用工具
│
├── lenet.py            # LeNet-5
├── alexnet.py          # AlexNet
├── vgg.py              # VGG (VGG16 / VGG19 / 通用 VGG)
├── googlenet.py        # GoogLeNet (Inception V1)
├── resnet.py           # ResNet (18/34/50/101/152)
├── densenet.py         # DenseNet (121/161/169/201)
├── unet.py             # UNet (标准 / ResNet18 / MobileNetV2)
├── fcn.py              # FCN (32s / 16s / 8s / SimpleFCN)
├── mobilenetv2.py      # MobileNetV2
├── mobilenetv3.py      # MobileNetV3 (Large / Small)
├── deeplab3plus.py     # DeepLabV3+
└── convext.py          # ConvNeXt (Tiny / Small / Base / Large)
```

---

## 设计说明

- **模块化架构**：`backbone`、`neck`、`head` 的分离设计，体现了现代计算机视觉系统的模块化思想，便于灵活组合（如 UNet 可替换不同的 encoder）。
- **预训练权重支持**：大部分模型通过 `build_xxx()` 或 `xxx()` 工厂函数支持加载 torchvision 预训练权重，方便进行迁移学习。
- **教学注释**：代码中附有详细的中文注释，解释每一层的作用和维度变化，帮助初学者理解数据在网络中的流动过程。
- **独立运行**：每个文件底部都有 `if __name__ == "__main__"` 测试代码，可使用 `torchinfo.summary()` 查看模型结构和参数量。
