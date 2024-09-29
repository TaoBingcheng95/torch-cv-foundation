import torch
from torch import nn
import torch.nn.functional as F

from torchinfo import summary
 
# LeNet Model definition
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc1_drop = nn.Dropout()
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.reshape(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class LeNetV1(nn.Module):
    def __init__(self, inchannels=1, num_classes=10):
        super(LeNetV1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inchannels, 6, 5),  # in_channels, out_channels, kernel_size
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),  # kernel_size, stride
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            # 这里是16x4x4的原因是，数据源大小为28x28，而不是32x32
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, out_features=num_classes)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNet().to(device)
    # 输入数据的形状为(1, 1, 28, 28)，即单通道28x28的灰度图像（典型的MNIST图像）
    summary(model, (1, 1, 28, 28))
