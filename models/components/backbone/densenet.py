
# import torch
from typing import Tuple
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.densenet as densenet


class densenet121(nn.Module):
    def __init__(self,weights=densenet.DenseNet121_Weights.IMAGENET1K_V1,
                 progress=True,
                 **kwargs):
        super(densenet121,self).__init__()
        backbone = densenet.densenet121(weights=weights, progress=progress, **kwargs)
        self.backbone = backbone.features
    def forward(self,x):
        out = F.relu(self.backbone(x), inplace=True)
        return out # 1024 channel


class densenet169(nn.Module):
    def __init__(self,weights=densenet.DenseNet169_Weights.IMAGENET1K_V1,
                 progress=True,
                 **kwargs):
        super(densenet169,self).__init__()
        backbone = densenet.densenet169(weights=weights,
                                        progress=progress,
                                        **kwargs)
        self.backbone = backbone.features
    def forward(self,x):
        out = F.relu(self.backbone(x), inplace=True)
        return out # 1664 channel



if __name__ == "__main__":
    from torchinfo import summary
    backbone = densenet121(weights=None, progress=False)
    summary(backbone,input_size=(1,3,224,224))
