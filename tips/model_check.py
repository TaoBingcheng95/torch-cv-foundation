import torch
from torch import nn
import torchvision.models as models
from models.components.DenseNet import DenseNet
from torchinfo import summary

def print_model_layers(model, indent=0):
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    for name, module in model._modules.items():
        if module is not None:
            print('  ' * indent + name)
            print_model_layers(module, indent + 1)


def modify_backbone(in_channels=1):
    backbone = models.resnet18(weights=None)
    original_conv1 = backbone.conv1
    print(original_conv1)
    backbone.conv1 = nn.Conv2d(in_channels, original_conv1.out_channels,
                            kernel_size=original_conv1.kernel_size[0],
                            stride=original_conv1.stride[0],
                            padding=original_conv1.padding,
                            bias=original_conv1.bias is not None)
    print(backbone.conv1)


def modify_output(out_features=100):
    # 加载预训练过的AlexNet模型
    model = models.alexnet(weights='AlexNet_Weights.DEFAULT')

    # 如果需要固定卷积层部分的权重，则在定义新的全连接层前加入以下两行代码:
    #for param in model.parameters():
    #    param.requires_grad = False
    # 基于原AlexNet的模型结构，修改全连接层
    # 相当于重写classifier部分，但实际只是修改最后的输出层为400个类别
    model.classifier = nn.Sequential(nn.Dropout(0.5),
                                     nn.Linear(9216, 4096, bias = True),
                                     nn.ReLU(inplace = True),
                                     nn.Dropout(0.5),
                                     nn.Linear(4096, 4096, bias = True),
                                     nn.ReLU(inplace = True),
                                     nn.Linear(4096, out_features))



if __name__ == '__main__':
    # For a single-channel output (e.g., binary segmentation)
    model = DenseNet(in_channels=1, num_init_features=5)
    # print(model)
    input_size = (1, 1, 512, 512) # (1, 3, 224, 224)
    summary(model, input_size=input_size, )

    # modify_backbone(in_channels=1)
    # modify_output(out_features=100)
