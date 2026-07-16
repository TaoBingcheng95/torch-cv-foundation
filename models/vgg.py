
from typing import cast, Optional, Union

import torch
import torch.nn as nn
# import torch.functional as F
from torchvision.models import WeightsEnum

try:
    from .utils.pytorch_api import _ovewrite_named_param
except ImportError as e:
    _ovewrite_named_param = None


cfgs: dict[str, list[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"], # VGG11
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"], # VGG13
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"], # VGG16
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"], # VGG19
}


class VGG16(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            # C1 : 112×112×64 
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 1/2

            # C2 : 56×56×128 
            nn.Conv2d(64, 128, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 1/4

            # C3 : 28×28×256
            nn.Conv2d(128, 256, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 1/8

            # C4 : 14×14×512
            nn.Conv2d(256, 512, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 1/16

            # C5 : 7×7×512
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 1/32
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), # F6 nn.Linear(512, 4096) without self.avgpool
            nn.ReLU(inplace=True), 
            nn.Dropout(),
            nn.Linear(4096, 4096),  # F7
            nn.ReLU(inplace=True), 
            nn.Dropout(),
            nn.Linear(4096, num_classes) # F8
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1) # flatten to (batch_size, 512)
        x = self.classifier(x)
        return x


class VGG19(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            # C1 : 112×112×64 
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 1/2

            # C2 : 56×56×128 
            nn.Conv2d(64, 128, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 1/4

            # C3 : 28×28×256
            nn.Conv2d(128, 256, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 1/8

            # C4 : 14×14×512
            nn.Conv2d(256, 512, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 1/16

            # C5 : 7×7×512
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 1/32
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), # F6 nn.Linear(512, 4096) without self.avgpool
            nn.ReLU(inplace=True), 
            nn.Dropout(),
            nn.Linear(4096, 4096),  # F7
            nn.ReLU(inplace=True), 
            nn.Dropout(),
            nn.Linear(4096, num_classes) # F8
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1) # flatten to (batch_size, 512)
        x = self.classifier(x)
        return x




class VGG(nn.Module):
    def __init__(
        self, features: nn.Module, num_classes: int = 1000, init_weights: bool = True, dropout: float = 0.5
    ) -> None:
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x



def _make_layers(cfg: list[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)



def _vgg(cfg: str, batch_norm: bool, weights: Optional[WeightsEnum], progress: bool, **kwargs):
 
    if weights is not None and _ovewrite_named_param:
        kwargs["init_weights"] = False
        if weights.meta["categories"] is not None:
            _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = VGG(_make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if weights:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True), strict=False)
    return model
 
 
def build_vgg(arch='vgg16', cfg='D', weights: Optional[WeightsEnum] = None, progress=True, **kwargs):
    return _vgg(cfg=cfg, batch_norm=False,  weights=weights, progress=progress, **kwargs)
 


if __name__ == "__main__":

    from torchvision.models.vgg import VGG16_Weights , vgg16
    from torchinfo import summary

    model = build_vgg(weights= VGG16_Weights.DEFAULT)
    num_classes = 10
    model.classifier[6] = torch.nn.Linear(4096, num_classes)   # 替换最后一层

    data = torch.randn(1, 3, 244, 244)
    # # model = vgg16()
    # print(model)
    # res = model(data)
    # print(res.shape)
    summary(model, input_data=data)
