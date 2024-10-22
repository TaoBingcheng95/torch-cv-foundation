import os
from pathlib import Path
import datetime
from loguru import logger
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

from dataset.mnist_datamodule import MNISTDataModule
from models.components import SimpleDenseNet
from metrics import Metrics
from trainers import BaseTrainer

def setup_logging(output_dir):
    """
    设置loguru的日志记录，按照当前运行时间作为文件名保存

    :param output_dir: 日志文件保存的目录
    """
    os.makedirs(output_dir, exist_ok=True)
    log_file_path = os.path.join(output_dir, "training.log")
    logger.remove()
    logger.add(log_file_path, rotation="500 MB", retention="10 days", level="INFO")
    logger.add(lambda msg: print(msg, end=""), level="INFO")
    logger.info(f"Logging is set up. Logs are being saved to {log_file_path}.")



if __name__ == '__main__':

    dm = MNISTDataModule(data_dir='./data',
                         batch_size=8,
                         pin_memory=True)
    dm.prepare_data()
    dm.setup()
    train_dl = dm.train_dataloader()
    val_dl = dm.val_dataloader()
    test_dl = dm.test_dataloader()
    # print(len(train_dl),len(val_dl),len(test_dl))
    x, y = next(iter(train_dl))
    # print(x.shape)

    model = SimpleDenseNet(output_size=dm.num_classes)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=1e-3,
                                 weight_decay=0.0)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
    #                                            mode='min',
    #                                            patience=10,
    #                                            factor=0.1)

    # compile model for faster training with pytorch 2.0
    compile= True

    tt = BaseTrainer(model=model,
                       device='cuda:0',
                       train_dataloader=train_dl,
                       val_dataloader=val_dl,
                       test_dataloader=test_dl,
                       num_classes=10,
                       optimizer_type='sgd',
                       epochs=3,
                       compile=compile
                       )
    tt.fit()
