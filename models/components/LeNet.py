import torch
from torch import nn
import torch.nn.functional as F
try:
    from torchinfo import summary
except ImportError as e:
    print(e)


class LeNetFeatExtractor(nn.Module):
    def __init__(self):
        super(LeNetFeatExtractor, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, 3)
        self.conv2 = nn.Conv2d(128, 16, 3)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        return x

class LeNetClassifier(nn.Module):
    def __init__(self):
        super(LeNetClassifier, self).__init__()
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.feat = LeNetFeatExtractor()
        self.classifer = LeNetClassifier()

    def forward(self, x):
        x = self.feat(x)
        x = self.classifer(x)
        return x


class LeNetv2(nn.Module):
    """
    support 28x28 MNIST data
    """

    def __init__(self):
        super(LeNetv2, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            # nn.Conv2d(in_channels=16, out_channels=256, kernel_size=5, stride=1, padding=0),
            # nn.ReLU(),
            # nn.AvgPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(16*4*4, 120),  # 120
            nn.ReLU(),
            nn.Linear(120, 84), # 120
            nn.ReLU(),
            nn.Linear(84, 10),

        )

    def forward(self, x):
        a1 = self.feature_extractor(x)
        a1 = torch.flatten(a1, 1)
        a2 = self.classifier(a1)
        a3 = F.softmax(a2, dim=0)
        return a3



if __name__ == '__main__':
    model = LeNetv2()
    input_size = (1, 1, 28, 28)
    summary(model, input_size=input_size)