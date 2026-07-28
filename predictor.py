import argparse
import numpy as np
from PIL import Image
import torch
import segmentation_models_pytorch as smp

import matplotlib.pyplot as plt

from dataset import auto_pin_memory, get_smart_num_workers
from dataset import VOC2012ClassSegLoader
from trainers import BaseTrainer
from utils.hardware import select_device
from loss import CEWithLogitsLoss


if __name__ == "__main__":

    ckpt_fn = "checkpoints/20260728_151016/best.pt"
    map_location = torch.device('mps')

    model = smp.Unet(
        encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=21,  # model output channels (number of classes in your dataset)
    ).to(map_location)
    model.load_state_dict(torch.load(ckpt_fn, weights_only=True, map_location=map_location)['model'], strict=True)


    img = 'data/VOCdevkit/VOC2012/JPEGImages/2007_000033.jpg'
    img = Image.open(img).convert('RGB')
    img = np.array(img)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    img = img.unsqueeze(0)
    img = img.to(map_location)

    pred = model(img)
    pred = pred.argmax(dim=1).cpu().numpy()
    print(np.unique(pred[0]))
    pred = pred[0]
    pred[pred!=1] = 0
    plt.imshow(pred)
    plt.show()
