"""
Author: sihui3 sihui3@staff.weibo.com
Date: 2023-03-14 17:03:31
LastEditors: sihui3 sihui3@staff.weibo.com
LastEditTime: 2023-03-14 18:53:34
FilePath: /my_code/pytorch_lightning/classifier/minist.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
"""
# https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/08-deep-autoencoders.html
import os
from functools import partial
# from loguru import logger

import torch
# from sentry_sdk.utils import epoch
# from torch import nn
# import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from dataset.mnist_datamodule import MNISTDataModule
from models.components import SimpleDenseNet
from models import MNISTLitModule

torch.set_float32_matmul_precision('medium') # 'medium' | 'high'


def prepare_data():
    train_full_ds = MNIST('./data',
                     train=True,
                     download=True,
                     transform=transforms.ToTensor())
    print(len(train_full_ds))
    train_ds, val_ds = random_split(train_full_ds, [55000, 5000])
    test_ds = MNIST('./data',
                     train=False,
                     download=True,
                     transform=transforms.ToTensor())
    print(len(test_ds))
    train_dl = DataLoader(train_ds,
                          batch_size=32,
                          num_workers=0,
                          pin_memory=True)
    val_dl = DataLoader(val_ds,
                          batch_size=32,
                          num_workers=0,
                          pin_memory=True)
    test_dl = DataLoader(test_ds,
                          batch_size=32,
                          num_workers=0,
                          pin_memory=True)
    return train_dl, val_dl, test_dl


if __name__ == "__main__":

    mnist_dm = MNISTDataModule(data_dir='./data/classification',
                               batch_size=32,
                               pin_memory=True,
                               resize_32=False)

    # autoencoder = LitAutoEncoder()
    net = SimpleDenseNet()
    # optimizer = torch.optim.Adam(net.parameters(),
    #                              lr=1e-3,
    #                              weight_decay= 0.0)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
    #                                                       mode='min',
    #                                                       patience=10,
    #                                                       factor=0.1)
    # compile model for faster training with pytorch 2.0
    compile = False
    mm = MNISTLitModule(net,
                        partial(torch.optim.Adam, lr=1e-3,weight_decay= 0.0),
                        partial(torch.optim.lr_scheduler.ReduceLROnPlateau,mode='min',patience=10,factor=0.1),
                        compile)

    # model_checkpoint = pl.callbacks.ModelCheckpoint(monitor='val/loss',
    #                                                 # filename='mnist-{epoch:02d}-{val_loss:.2f}',
    #                                                 filename='epoch_{epoch:03d}',
    #                                                 # save_top_k=1,
    #                                                 save_last= True,
    #                                                 auto_insert_metric_name= False,
    #                                                 mode='max')

    model_checkpoint = ModelCheckpoint(
        # save_top_k=1,
        monitor="val_loss",
        # mode="min",
        save_last=True,
        filename='best-{epoch:02d}-{val_loss:.3f}',
        # auto_insert_metric_name = True,
    )

    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=100,
                                   mode='min',
                                   min_delta=0.00,
                                   verbose=False)
    # model_summary = pl.callbacks.RichModelSummary(max_depth=-1)
    rich_progress_bar = pl.callbacks.RichProgressBar()

    trainer = pl.Trainer(max_epochs=5,
                         devices=1,
                         accelerator="gpu",
                         callbacks = [model_checkpoint,
                                      early_stopping,
                                      # model_summary,
                                      rich_progress_bar
                                      ]
                         )

    trainer.fit(mm, # autoencoder
                datamodule=mnist_dm)

    trainer.validate(datamodule=mnist_dm, ckpt_path='best')
    trainer.test(datamodule=mnist_dm, ckpt_path='best')
