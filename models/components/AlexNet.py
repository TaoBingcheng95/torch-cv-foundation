import torch
import torch.nn as nn
import torch.nn.functional as F
from models.components.head.seg_head import GenericSegmentationHead

class AlexNet(nn.Module):
    """
    """
    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 1000,
                 dropout: float = 0.5,
                 init_weights: bool = True) -> None:
        """
        dropout : Dropout概率，默认为	0.5
        init_weights : 是否自动初始化权重，默认为True
        """
        super().__init__()
        self.features = nn.Sequential(
            # nn.Conv2d(in_channels, 64, kernel_size=11, stride=4, padding=2), # 原版参数
            nn.Conv2d(in_channels, 64, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # 自适应平均池化层
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        self.seg_head = GenericSegmentationHead(in_channels=256,
                                                num_classes=num_classes,
                                                target_resolution=(224, 224))
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x) # 1, 256, 2, 2
        x = self.avgpool(x) # 1, 256, 6, 6
        x = torch.flatten(x, 1) # 1, 9216
        x = self.classifier(x)
        # _, features_channel, h, w = x.shape
        # x = self.seg_head(x, (h, w))
        return x

    def _initialize_weights(self):
        """
        卷积层：使用Kaiming初始化（适合ReLU激活）
        全连接层：使用小标准差的正态分布
        偏置项：全部初始化为0
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  #何教授方法
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)  #正态分布赋值
                nn.init.constant_(m.bias, 0)



if __name__ == '__main__':
    try:
        from torchinfo import summary
    except ImportError as e:
        print(e)
        exit()

    model = AlexNet(in_channels=1, num_classes=10)
    input_size=(1, 1, 32, 32)
    # input_data = torch.randn(input_size)
    # output = model(input_data)
    # print(output.shape)

    summary(model,
            input_size=input_size,
            col_width=20,
            col_names=['input_size', 'output_size', 'num_params', 'trainable'],
            row_settings=['var_names'],
            verbose=True
            )
