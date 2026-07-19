from collections import OrderedDict

from torch import nn
from torchvision import models


class PredictionHead(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.head = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(in_features, in_features)),
            ('relu1', nn.ReLU()),
            ('final', nn.Linear(in_features, out_features))
        ]))
    def forward(self, x):
        return self.head(x)


class HydraNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1) # ResNet18_Weights.DEFAULT
        self.n_features = self.net.fc.in_features
        self.net.fc = nn.Identity()
        self.age_head = PredictionHead(self.n_features, 1)
        self.gender_head = PredictionHead(self.n_features, 1)
        self.ethnicity_head = PredictionHead(self.n_features, 5)

    def forward(self, x):
        features = self.net(x)
        age_head = self.age_head(features)
        gender_head = self.gender_head(features)
        ethnicity_head = self.ethnicity_head(features)
        return age_head, gender_head, ethnicity_head
