# https://lightning.ai/docs/pytorch/stable/starter/introduction.html
# https://zhuanlan.zhihu.com/p/116769890

import os
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

# import lightning.pytorch as pl


class LitAutoEncoder(nn.Module): # pl.LightningModule
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, 32),
                                     nn.ReLU(),
                                     nn.Linear(32, 3),
                                     )
        self.decoder = nn.Sequential(nn.Linear(3, 32),
                                     nn.ReLU(),
                                     nn.Linear(32, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 28 * 28),)

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        x = x.view(x.size(0), -1)
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

############################################

class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 16)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.relu(self.fc2(x))


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(64, 128)
        self.fc2 = torch.nn.Linear(128, 28 * 28)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))


class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


class AutoEncoderModules(nn.Module): #pl.LightningModule
    def __init__(self, auto_encoder):
        super().__init__()
        self.auto_encoder = auto_encoder
        self.metric = torch.nn.MSELoss()

    def forward(self, x):
        return self.auto_encoder(x)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self.auto_encoder(x)
        loss = self.metric(x, x_hat)
        return loss

    def validation_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "test")

    def _shared_eval(self, batch, batch_idx, prefix):
        x, _ = batch
        x_hat = self.auto_encoder(x)
        loss = self.metric(x, x_hat)
        self.log(f"{prefix}_loss", loss)


if __name__ == "__main__":
    from torchinfo import summary
    from torchvision.datasets import MNIST
    from torchvision.transforms import ToTensor

    # model = AutoEncoder()
    # model = LitAutoEncoder(model)

    autoencoder = LitAutoEncoder()

    input_size = (1, 1, 28, 28)
    # output = autoencoder(torch.randn(input_size))
    # print(output.shape)
    # summary(autoencoder, input_size=input_size)

    # # setup data
    # dataset = MNIST("../data", download=True, train=True, transform=ToTensor())
    # train_loader = DataLoader(dataset)
    # x, y = next(iter(train_loader))
    # # xx = x.view(28, 28).cpu().numpy()
    # # plt.title(f'{y.cpu().numpy()[0]}')
    # # plt.imshow((255*xx).astype(np.uint8), cmap='gray')
    # # plt.show()

    # # train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
    # trainer = pl.Trainer(limit_train_batches=100, max_epochs=100)
    # # trainer.fit(model=autoencoder, train_dataloaders=train_loader)
    # # load checkpoint
    # checkpoint = "./lightning_logs/version_2/checkpoints/epoch=99-step=10000.ckpt"
    # net = LitAutoEncoder.load_from_checkpoint(checkpoint).to(device=autoencoder.device) # , encoder=encoder, decoder=decoder

    # # choose your trained nn.Module
    # encoder = net.encoder
    # encoder.eval()

    # # # embed 4 fake images!
    # # fake_image_batch = torch.rand(4, 28 * 28, device=autoencoder.device)
    # # embeddings = encoder(fake_image_batch)
    # # print("⚡" * 20, "\nPredictions (4 image embeddings):\n", embeddings, "\n", "⚡" * 20)

    # result = encoder(x.to(autoencoder.device).view(x.size(0), -1))
    # pred = net.decoder(result).view(28,28)

    # fig, axis = plt.subplots(1,2)
    # axis[0].imshow(x[0].view(28,28))
    # axis[1].imshow(pred.cpu().detach().numpy())
    # plt.show()
