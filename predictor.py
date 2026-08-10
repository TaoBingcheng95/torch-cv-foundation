import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
# import segmentation_models_pytorch as smp
from models.deeplab3plus import DeepLabV3Plus
from models.backbone import ResNet18Encoder, ResNet50Encoder

import matplotlib.pyplot as plt

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


if __name__ == "__main__":

    ckpt_fn = Path("checkpoints\\20260729_000940\\best.pt")
    map_location = torch.device('cuda')

    # model = smp.Unet(
    #     encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    #     encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
    #     in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    #     classes=21,  # model output channels (number of classes in your dataset)
    # ).to(map_location)
    model = DeepLabV3Plus(
            backbone=ResNet50Encoder(weights=False),
            num_classes=21,
        )
    model.to(map_location)
    model.load_state_dict(torch.load(ckpt_fn, weights_only=True, map_location=map_location)['model'], strict=True)
    model.eval()

    IMAGENET_MEAN = IMAGENET_MEAN.reshape(-1,1,1)
    IMAGENET_STD = IMAGENET_STD.reshape(-1,1,1)
    std = torch.Tensor(IMAGENET_STD)
    mean = torch.Tensor(IMAGENET_MEAN)

    img = 'data\\VOCdevkit\\VOC2012\\JPEGImages\\2007_000033.jpg'
    img = Image.open(img).convert('RGB')
    img = np.array(img)
    img = img/255.0
    
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    img = (img - IMAGENET_MEAN) / IMAGENET_STD

    img = img.unsqueeze(0)
    img = img.to(map_location)
    print(img.shape)
    with torch.no_grad():
        pred = model(img)
        pred = pred.argmax(dim=1).cpu().numpy()
        print(np.unique(pred[0]))
    pred = pred[0]
    # pred[pred!=1] = 0
    plt.imshow(pred)
    plt.show()
