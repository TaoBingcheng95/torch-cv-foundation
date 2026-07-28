import platform

import numpy as np
import torch
import torch.nn as nn
from  torchvision import transforms

# from models.deeplab3plus import DeepLabV3Plus
# from models.backbone import ResNet18Encoder, VGG16Encoder, MobileNetV2Encoder
# from models.unet import UNet_ResNet18, UNet_MobileNetV2, UNet

# from models.convnext_official import convnext_base
# from models.convnext import convnext_base, CNBlock

# from models.convnextv2 import convnextv2_base
from models import AlexNet
# from torchvision.datasets import VOCSegmentation
from dataset import VOC2012ClassSegLoader



if __name__ == '__main__':
    
    # 通道数现在由 backbone.out_channels 自动推导，无需手动传入
    # VGG16Encoder()       → low_level=128, high_level=512
    # MobileNetV2Encoder() → low_level=24,  high_level=96
    # ResNet18Encoder()    → low_level=64,  high_level=256
    # model = DeepLabV3Plus(MobileNetV2Encoder())

    # model  = AlexNet()

    # input_size = (1,3,224,224)
    # dummy_input = torch.randn(input_size)
    # o = model(dummy_input)
    # print(o.shape)

    # summary(model, input_size=input_size)

    # block = CNBlock(dim=96, layer_scale = 1e-6, stochastic_depth_prob=0.0) # , out_dim=192, kernel_size=3
    # print(block)


    # transform = transforms.Compose([
    #         transforms.ToTensor()
    #     ])
    # dm = VOCSegmentation(root='./data', year="2012",image_set='val', transform=transform, target_transform=transform)
    # print(len(dm))
    # x, y = dm[10]
    # print(x.shape)
    # print(y.shape)
    # print(np.unique(y.numpy()))

    dm = VOC2012ClassSegLoader(root='./data',
                                    batch_size=4,
                                    )
    train_dl = dm.train_dataloader()
    val_dl = dm.val_dataloader()
    test_dl = dm.test_dataloader()

    x, y = next(iter(train_dl))
    x = x.numpy()
    y = y.numpy()
    print(np.max(x))
    print(np.min(x))
    print(np.max(y))
    print(np.min(y))

