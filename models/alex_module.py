import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
import lightning.pytorch as pl

from components import AlexNet


class AlexNetLightning(pl.LightningModule):
    def __init__(self,
                 num_classes: int = 5,
                 dropout: float = 0.5,
                 init_weights: bool = False):
        super().__init__()
        self.model = AlexNet(num_classes=num_classes,
                             dropout=dropout,
                             init_weights=init_weights)
        self.loss = nn.CrossEntropyLoss()
        self.lr = 0.0005

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch):
        images, labels = batch
        logits = self(images)
        loss = F.cross_entropy(logits, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        logits = self(images)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == labels).float() / labels.size(0)
        self.log('val_acc', acc)


if __name__ == '__main__':
    model = AlexNetLightning()
    print(model)
