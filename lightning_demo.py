"""
Author: sihui3 sihui3@staff.weibo.com
Date: 2023-03-14 17:03:31
LastEditors: sihui3 sihui3@staff.weibo.com
LastEditTime: 2023-03-14 18:53:34
FilePath: /my_code/pytorch_lightning/classifier/minist.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
"""
import os
import torch
from sentry_sdk.utils import epoch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
# import pytorch_lightning as pl
import lightning.pytorch as pl
from loguru import logger

from dataset.datamodule.mnist_datamodule import MNISTDataModule
from models.components import SimpleDenseNet
from models.module import MNISTLitModule

torch.set_float32_matmul_precision('medium') # 'medium' | 'high'

class LitAutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 28 * 28))

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


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

    mnist_dm = MNISTDataModule(data_dir='./data',
                             batch_size=32,
                             pin_memory=True)

    # autoencoder = LitAutoEncoder()
    net = SimpleDenseNet()
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=1e-3,
                                 weight_decay= 0.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                          mode='min',
                                                          patience=10,
                                                          factor=0.1)
    # compile model for faster training with pytorch 2.0
    compile = False
    mm = MNISTLitModule(net,
                        torch.optim.Adam,
                        torch.optim.lr_scheduler.ReduceLROnPlateau,
                        compile)

    model_checkpoint = pl.callbacks.ModelCheckpoint(monitor='val/loss',
                                                    # dirpath='./checkpoints',
                                                    filename='mnist-{epoch:02d}-{val_loss:.2f}',
                                                    # save_top_k=1,
                                                    save_last= True,
                                                    auto_insert_metric_name= False,
                                                    mode='max')
    early_stopping = pl.callbacks.EarlyStopping(monitor='val/loss',
                                                patience=100,
                                                mode='max')
    model_summary = pl.callbacks.RichModelSummary(max_depth=-1)
    rich_progress_bar = pl.callbacks.RichProgressBar()

    trainer = pl.Trainer(max_epochs=10,
                         devices=1,
                         accelerator="gpu",
                         # callbacks = [model_checkpoint, early_stopping, model_summary]
                         )

    trainer.fit(mm, # autoencoder
                datamodule=mnist_dm)
