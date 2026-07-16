import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class ResNet50Backbone(nn.Module):
    def __init__(self, weights=models.ResNet50_Weights.IMAGENET1K_V1,
                 num_output_features=2048):
        """
        初始化ResNet50Backbone

        :param weights: 是否加载预训练权重，默认为True
        :param weights: 输出特征层的维度，默认为True
        """
        super(ResNet50Backbone, self).__init__()
        resnet50 = models.resnet50(weights=weights)

        # 移除最后的全连接层和全局平均池化层
        self.encoder = nn.Sequential(*list(resnet50.children())[:-2])
        # 由于没有添加额外的层来改变输出通道数，所以num_output_features实际上是ResNet-50最后一个卷积层的输出通道数
        self.num_output_features = resnet50.fc.in_features if not hasattr(self, 'fc') else num_output_features

    def forward(self, x):
        """
        前向传播

        :param x: 输入张量
        :return: 提取的特征图
        """
        features = self.encoder(x)  # 输出形状: [batch_size, 2048, h, w]
        return features


class ClassifierHead(nn.Module):
    def __init__(self, num_features, num_classes=10):
        """
        初始化分类头

        :param num_features: 输入特征向量的维度
        :param num_classes: 分类任务的类别数
        """
        super(ClassifierHead, self).__init__()
        self.fc1 = nn.Linear(num_features, 512)  # 添加一个隐藏层
        self.fc2 = nn.Linear(512, num_classes)  # 添加输出层

    def forward(self, x):
        """
        前向传播

        :param x: 输入特征向量
        :return: 分类结果（类别得分）
        """
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        """
        初始化分割头

        :param in_channels: 输入特征图的通道数（即ResNet-50最后一个卷积层的输出通道数）
        :param num_classes: 分割任务的类别数
        """
        super(SegmentationHead, self).__init__()

        # 上采样和卷积层
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, padding=1),  # 减少通道数
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),  # 进一步减少通道数
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 上采样
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 上采样
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 上采样
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # 上采样
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, num_classes, kernel_size=4, stride=2, padding=1),  # 输出类别数
        )

    def forward(self, x):
        """
        前向传播

        :param x: 输入特征图
        :return: 分割结果（每个像素的类别得分）
        """
        return self.decoder(x)


class ResNet50MultiTask(nn.Module):
    def __init__(self, num_classes_classifier=10,
                 num_classes_segmentation=10,
                 weights=models.ResNet50_Weights.IMAGENET1K_V1,
                 use_classifier=True,
                 use_segmentation=True):
        """
        初始化ResNet50多任务模型

        :param num_classes_classifier: 分类任务的类别数
        :param num_classes_segmentation: 分割任务的类别数
        :param weights: 是否加载预训练权重，默认为True
        :param use_classifier: 是否使用分类头
        :param use_segmentation: 是否使用分割头
        """
        super(ResNet50MultiTask, self).__init__()
        self.backbone = ResNet50Backbone(weights=weights)
        self.use_classifier = use_classifier
        self.use_segmentation = use_segmentation

        # 分类头
        if use_classifier:
            self.classifier_head = ClassifierHead(num_features=2048, num_classes=num_classes_classifier)

        # 分割头
        if use_segmentation:
            self.segmentation_head = SegmentationHead(in_channels=2048, num_classes=num_classes_segmentation)

    def forward(self, x):
        """
        前向传播

        :param x: 输入张量
        :return: 分类结果和/或分割结果
        """
        features = self.backbone(x)  # 提取特征

        outputs = {}
        if self.use_classifier:
            # 全局平均池化 + 分类头
            pooled_features = F.adaptive_avg_pool2d(features, (1, 1))  # 全局平均池化
            pooled_features = pooled_features.view(pooled_features.size(0), -1)  # 展平
            outputs["classifier"] = self.classifier_head(pooled_features)  # 分类结果

        if self.use_segmentation:
            # 分割头
            outputs["segmentation"] = self.segmentation_head(features)  # 分割结果

        return outputs


class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes=10,
                 weights=models.ResNet50_Weights.IMAGENET1K_V1):
        super(ResNet50Classifier, self).__init__()
        self.backbone = ResNet50Backbone(weights=weights)
        self.head = ClassifierHead(num_features=self.backbone.num_output_features,
                                   num_classes=num_classes)

    def forward(self, x):
        features = self.backbone(x) # [1, 2048, 7, 7]
        # 全局平均池化 + 分类头
        pooled_features = F.adaptive_avg_pool2d(features, (1, 1))  # 全局平均池化
        pooled_features = pooled_features.view(pooled_features.size(0), -1)  # 展平
        logits = self.head(pooled_features)  # 分类结果
        return logits


class ResNet50Segmentation(nn.Module):
    def __init__(self, num_classes=10,
                 in_channels=2048,
                 weights=models.ResNet50_Weights.IMAGENET1K_V1,
                 ):
        super(ResNet50Segmentation, self).__init__()
        self.backbone = ResNet50Backbone(weights=weights)
        self.head = SegmentationHead(in_channels=self.backbone.num_output_features,
                                     num_classes=num_classes)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.head(features)
        return logits


def multi_task():
    model = ResNet50MultiTask(num_classes_classifier=10,
                              num_classes_segmentation=21,
                              use_classifier=True,
                              use_segmentation=True)
    # 假设输入是一个4D张量 (batch_size, channels, height, width)
    input_tensor = torch.randn(1, 3, 224, 224)
    outputs = model(input_tensor)
    # # 打印输出形状
    if "classifier" in outputs:
        print("Classifier output shape:", outputs["classifier"].shape)  # 分类输出形状: (1, 10)
    if "segmentation" in outputs:
        print("Segmentation output shape:", outputs["segmentation"].shape)  # 分割输出形状: (1, 21, 224, 224)


if __name__ == "__main__":

    from torchinfo import summary

    model = ResNet50Segmentation(num_classes=10)
    # model = ResNet50Classifier(num_classes=10)
    # print(model)

    # 假设输入是一个4D张量 (batch_size, channels, height, width)
    input_size = (1, 3, 512, 512)
    # input_tensor = torch.randn(input_size)  # 示例输入
    # outputs = model(input_tensor)
    # print(outputs.shape)
    summary(model,
            input_size=input_size,
            col_width=20,
            col_names=['input_size', 'output_size', 'num_params', 'trainable'],
            row_settings=['var_names'],
            verbose=True
            )

