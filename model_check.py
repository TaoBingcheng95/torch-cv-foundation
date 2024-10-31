import torch
from torch import nn
import torchvision.models as models
from models.components.BiSeNet import BiSeNetV1
from models.components.DenseNet import DenseNets
from torchinfo import summary

def modify_resnet(model, in_channels=1):
    in_channels = 1
    model = models.resnet18(pretrained=True)
    original_conv1 = model.conv1
    model.conv1 = nn.Conv2d(in_channels, original_conv1.out_channels, 
                            kernel_size=original_conv1.kernel_size, 
                            stride=original_conv1.stride, 
                            padding=original_conv1.padding, 
                            bias=original_conv1.bias is not None)

if __name__ == '__main__':
    model = DenseNets(input_channels=4, out_features=5)  # For a single-channel output (e.g., binary segmentation)
    # print(model)

    input_size = (1, 4, 512, 512) # (1, 3, 224, 224)
    # summary(model, input_size=input_size)
    
    x = torch.randn(input_size)  # Example input tensor (batch_size, channels, height, width)
    output = model(x)
    print(output.shape)  # Should be (1, 1, 256, 256) for this example

    # backbone = models.resnet101(pretrained=False)
    
    # print(backbone.conv1)
    # backbone.conv1= nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3,bias=False)
    # print(backbone.conv1)
    


