from loguru import logger
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pprint import pprint

from dataset.tianchi_building import TianchiDataset
from trainers.pt_trainer import Trainer

# from config.load_config import TrainConfig
# from datasets.transform import transform
from models.mynet.Unet import UNetV1
# from utils import parse_args, setup_logging
# from tqdm import tqdm
# from datetime import datetime
# from metrics import Metrics


if __name__ == '__main__':

    Tianchi_dir = 'D:\\myspace\\dataset\\segemnt\\tianchi\\train'
    val_ratio = 0.1
    test_ratio = 0.8
    num_classes = 2
    ds = TianchiDataset(root=Tianchi_dir, img_folder='image', label_folder='label')
    train_ds, val_ds, test_ds = random_split(ds,
                                             [len(ds)-int(len(ds)*val_ratio)-int(len(ds)*test_ratio),
                                              int(len(ds)*val_ratio),
                                              int(len(ds)*test_ratio)])

    batch_size = 4
    if sys.platform.startswith('win'):
        num_workers = 0
    else:
        num_workers = 4
    train_dl = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=num_workers, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=4, shuffle=True, num_workers=num_workers, drop_last=True)
    test_dl = DataLoader(test_ds, batch_size=4, shuffle=True, num_workers=num_workers, drop_last=True)
    x, y = next(iter(train_dl))
    print(x.dtype)
    print(y.dtype)

    model = UNetV1(in_channels=3, out_channels=2)

    learning_rate = 0.001
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    criterion = nn.CrossEntropyLoss() #nn.MSELoss()
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,
        gamma=0.1
    )

    epochs = 1
    tr = Trainer(model, num_classes=num_classes,
                 train_dataloader=train_dl,
                 val_dataloader=val_dl,
                 test_dataloader=val_dl,
                 epochs = epochs,
                 criterion=criterion,
                 # optimizer=optimizer,
                 # scheduler=scheduler
                 )
    tr.fit()

