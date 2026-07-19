import warnings
warnings.filterwarnings("ignore")
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# sns.set_theme(style="darkgrid", font_scale=1.5, font="SimHei", rc={"axes.unicode_minus":False})

import torch
import torchmetrics
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

class VAE(nn.Module):
    def __init__(self, input_dim=784, h_dim=400, z_dim=20):
        super(VAE, self).__init__()

        self.input_dim = input_dim
        self.h_dim = h_dim
        self.z_dim = z_dim

        # Encoder
        self.fc1 = nn.Linear(input_dim, h_dim)
        self.fc21 = nn.Linear(h_dim, z_dim)  # mu
        self.fc22 = nn.Linear(h_dim, z_dim)  # log_var

        # Decoder
        self.fc3 = nn.Linear(z_dim, h_dim)
        self.fc4 = nn.Linear(h_dim, input_dim)

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        mean = self.fc21(h)
        log_var = self.fc22(h)
        return mean, log_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.fc3(z))
        out = torch.sigmoid(self.fc4(h))
        return out

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        reconstructed_x = self.decode(z)
        return reconstructed_x, mean, log_var


def loss_function(x_hat, x, mu, log_var, KLD_weight=1):
    BCE_loss = F.binary_cross_entropy(x_hat, x, reduction="sum") # 重构损失
    KLD_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) # KL 散度损失
    loss = BCE_loss + KLD_loss * KLD_weight
    return loss, BCE_loss, KLD_loss


class LitModel(pl.LightningModule):
    def __init__(self, input_dim=784, h_dim=400, z_dim=20):
        super().__init__()
        self.model = VAE(input_dim, h_dim, z_dim)

    def forward(self, x):
        x = self.model(x)
        return x

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(), lr=lr, betas=(0.9, 0.99), eps=1e-08, weight_decay=1e-5
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        reconstructed_x, mean, log_var = self(x)
        loss, BCE_loss, KLD_loss = loss_function(reconstructed_x, x, mean, log_var, KLD_weight=KLD_weight)
        self.log("loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict(
            {
                "BCE_loss": BCE_loss,
                "KLD_loss": KLD_loss,
            },
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        return loss

    def decode(self, z):
        out = self.model.decode(z)
        return out


def metrics_plot(metrics_dir):
    log_path = os.path.join(metrics_dir, "metrics.csv")
    metrics = pd.read_csv(log_path)
    x_name = "epoch"

    plt.figure(figsize=(8, 6), dpi=100)
    sns.lineplot(x=x_name, y="loss", data=metrics, label="Loss", linewidth=2, marker="o", markersize=10)
    sns.lineplot(x=x_name, y="BCE_loss", data=metrics, label="BCE Loss", linewidth=2, marker="^", markersize=12)
    sns.lineplot(x=x_name, y="KLD_loss", data=metrics, label="KLD Loss", linewidth=2, marker="s", markersize=10)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    batch_size = 64

    epochs = 10
    KLD_weight = 1
    lr = 0.001

    input_dim = 784  # 28 * 28
    h_dim = 256  # 隐藏层维度
    z_dim = 2  # 潜变量维度

    train_dataset = datasets.MNIST(root="data", train=True, transform=transforms.ToTensor(), download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # vae = VAE(input_dim, h_dim, z_dim)
    # x = torch.randn((10, input_dim))
    # reconstructed_x, mean, log_var = vae(x)
    # print(reconstructed_x.shape, mean.shape, log_var.shape)
    # torch.Size([10, 784]) torch.Size([10, 2]) torch.Size([10, 2])

    # model = LitModel(input_dim, h_dim, z_dim)
    model = LitModel.load_from_checkpoint("logs/lightning_logs/version_0/checkpoints/epoch=9-step=9380.ckpt",
                                          input_dim=784,
                                          h_dim=h_dim,
                                          z_dim=z_dim)
    # logger = CSVLogger("./logs")
    # early_stop_callback = EarlyStopping(monitor="loss", min_delta=0.00, patience=5, verbose=False, mode="min")
    # trainer = pl.Trainer(
    #     max_epochs=epochs,
    #     enable_progress_bar=True,
    #     logger=logger,
    #     callbacks=[early_stop_callback],
    # )
    # trainer.fit(model, train_loader)
    # metrics_plot("./logs/lightning_logs/version_0")

    row, col = 2, 4
    z = torch.randn(row * col, z_dim).to(model.device)
    random_res = model.model.decode(z).view(-1, 1, 28, 28).detach().cpu().numpy()

    plt.figure(figsize=(col, row))
    for i in range(row * col):
        plt.subplot(row, col, i + 1)
        plt.imshow(random_res[i].squeeze(), cmap="gray")
        plt.xticks([])
        plt.yticks([])
        plt.axis("off")
    plt.show()

