
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50
from torchinfo import summary

from models.model_config import SemanticSegmentationModel


def prepare_model(num_classes=2):
    model = deeplabv3_resnet50(weights='DEFAULT')
    model.classifier[4] = nn.Conv2d(256, num_classes, 1)
    model.aux_classifier[4] = nn.Conv2d(256, num_classes, 1)
    return model

# model = SemanticSegmentationModel(model='unet', backbone='resnet50', weights=True)
# print(model)

model = prepare_model()
summary(model, (1, 3, 256, 256))
