
"""
LeNet-5用于解决手写数字的识别，输入手写数字图片(28×28)，输出所属数字类别。

LeNet一共有7层， 包含3个卷积层(C)，2个采样层/池化层(S)，和2个全连接层(F)
也可以将它看作[C1,S2]+[C3,S4]+[C5]+F6+F7的结构，这样则是五大层
其中前三大层由卷积和池化组成，用于特征提取，而后两个全连接层则用于拟合输出
通过添加一个自适应池化层，使其适配32×32的数字图片。
# https://www.bbbdata.com/text/812
"""

import torch
import torch.nn as nn
# import torch.nn.functional as F



class LeNet5(nn.Module):
    """
    对于接收标准的 1x28x28 图像, C1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
    若要原生接受的 1x32x32 图像, C1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
    """
    def __init__(self, num_classes: int=10) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2), # C1 : 1->6, 28x28 -> 28x28 / 特征图数, 输出尺寸
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)) # S2 : 6->6, 28x28 -> 14x14
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0), # C3 : 6->16, 14x14 -> 10x10
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)) # S4 : 16->16，10x10 -> 5x5
                
        # 🔥 核心修改：添加自适应池化，强制输出为 5x5
        self.adaptive_pool = nn.AdaptiveMaxPool2d((5, 5))       # 无论输入是6x6还是4*4，都变成5x5

        # an affine operation: y = Wx + b
        self.fc = nn.Linear(16*5*5, 120) # C5 : 16->120, 5x5->1×1
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(120, 84) # C6 : 120->84, 1×1->1×1
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(84, num_classes) # C7 : 84->num_classes, 1×1->1×1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FeatExtractor
        out = self.layer1(x) # [C1, S2] (1, 28, 28) -> (6, 28, 28) -> (6, 14, 14)
        out = self.layer2(out) # [C3, S4] (6, 14, 14) -> (16, 10, 10) -> (16, 5, 5)

        # 🔥 在这里应用自适应池化
        out = self.adaptive_pool(out)

        # Classifier
        out = torch.flatten(out, 1)
        out = self.fc(out) # C5  转化为一维向量 output(32*5*5)
        out = self.relu(out)
        out = self.fc1(out) # F6 output(84)
        out = self.relu1(out)
        out = self.fc2(out) # F7 output(10)
        return out



if __name__ == "__main__":
    from torchinfo import summary
    
    model = LeNet5(num_classes=10)
    input_size = (1, 1, 28, 28)
    # input_data = torch.rand(input_size)
    # out = model(input_data)
    # print(out.shape)

    summary(model, input_size=input_size)
