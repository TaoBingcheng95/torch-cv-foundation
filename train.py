# https://blog.itpub.net/18841117/viewspace-3015295/

import os
import sys
import time
import datetime
import copy

from PIL import Image
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
# import logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.datasets import FashionMNIST
# from torch.utils.tensorboard import SummaryWriter

from models.mynet.LeNet import LeNetV1
from trainers import Trainer

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'



def FashionMNIST_loader(root='./data', val_ratio=0.4, batch_size=32, num_workers=0):
    """
    加载FashionMNIST训练集数据
    此时train_data是完整数据集 未划分
    """

    # 定义数据预处理操作，主要包括图像调整大小和归一化
    image_size = 28
    transform = transforms.Compose([
        transforms.Resize(size=image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 对于Windows用户，这里应设置为0，否则会出现多线程错误
    if sys.platform.startswith('win'):
        num_workers = 0
    else:
        num_workers = 4

    train_ds = FashionMNIST(
        root=root,
        train=True,
        download=True,
        transform=transform
    )
    train_count = len(train_ds)
    train_data, val_data = random_split(train_ds,
                                        [round((1-val_ratio) * train_count), round(val_ratio * train_count)])

    test_ds = FashionMNIST(
        root=root,
        train=False,
        download=True,
        transform=transform
    )

    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers # 同样使用8个子进程加载数据。Windows用户请将此参数改为0或1
    )
 
    val_dataloader = DataLoader(
        dataset=val_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_dataloader = DataLoader(
        dataset=test_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
 
    return train_dataloader, val_dataloader, test_dataloader




if __name__ == '__main__':

    device_flag = 'cuda:0'
    FashionMNIST_dir = './data'

    # 超参数
    batch_size = 32
    # 对于Windows用户，这里应设置为0，否则会出现多线程错误
    if sys.platform.startswith('win'):
        num_workers = 0
    else:
        num_workers = 4
    lr = 1e-4
    epochs = 3

    LeNet = LeNetV1()

    # 加载并处理训练和验证数据
    train_dl, val_dl, test_dl = FashionMNIST_loader(root=FashionMNIST_dir, batch_size=batch_size,
                                                           num_workers=num_workers)

    x, y = next(iter(test_dl))

    # criterion = nn.CrossEntropyLoss() # torch.nn.modules.loss.CrossEntropyLoss

    aa = Trainer(LeNet,
                   num_classes=10,
                   train_dataloader=train_dl,
                   val_dataloader=val_dl,
                   test_dataloader = test_dl,
                   device=device_flag,
                   epochs=epochs,
                   optims='adam',
                   resume='checkpoints/20240930_105131/epoch_3_acc_0.7721.pth',
                   cls=True)
    aa.fit()

    res = aa.predict(x[0,:,:,:].numpy())
    print(f"predict : {res.cpu().numpy()}")
    print(f"truth : {y[0].numpy()}")
