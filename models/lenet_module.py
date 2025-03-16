# models/detection/lenet.py
"""
PyTorch reference: https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
"""

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics


class LeNetLitModule(pl.LightningModule):
    def __init__(self, in_channels: int, out_channels: int, lr: float = 2e-4):
        """
        Args:
        - in_channels: One for grayscale input image (which is the case for MNIST), 3 for RGB input image.
        - out_channels: Number of classes of the classifier. 10 for MNIST.
        """
        super().__init__()
        # Debugging tool to display intermediate input/output size of all your layer (called before fit)
        # self.example_input_array = torch.Tensor(16, in_channels, 32, 32)
        self.learning_rate = lr

        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=out_channels)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=out_channels)
        self.test_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=out_channels)

        # [img_size] 32 -> conv -> 32 -> (max_pool) -> 16
        # with 6 output activation maps
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=6,
                kernel_size=5,
                stride=1,
                # Either resize (28x28) MNIST images to (32x32) or pad the input to be 32x32
                # padding=2,
            ),
            nn.MaxPool2d(kernel_size=2),
        )
        # [img_size] 16 -> (conv) -> 10 -> (max pool) 5
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # The activation size (number of values after passing through one layer) is getting gradually smaller
        # and smaller.
        # The output is flattened and then used as a long input into the next dense layers.
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)  # 5 from the image dimension
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        # "Softmax" layer = Linear + Softmax.
        self.fc3 = nn.Linear(in_features=84, out_features=out_channels)

    # Method of LetNet class in models/detection/lenet.py
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv_layer1(x))
        x = F.relu(self.conv_layer2(x))
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def configure_optimizers(self) -> torch.optim.Adam:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


    def training_step(
            self,
            batch: list[torch.Tensor, torch.Tensor],
            batch_idx: int,
    ) -> torch.Tensor:
        """
        Function called when using `trainer.fit()` with trainer a lightning `Trainer` instance.
        """
        x, y = batch
        logit_preds = self(x)
        loss = F.cross_entropy(logit_preds, y)
        self.train_accuracy.update(torch.argmax(logit_preds, dim=1), y)
        self.log("train_acc_step", self.train_accuracy, on_step=True, on_epoch=True, logger=True)
        # logs metrics for each training_step, and the average across the epoch, to the progress bar and logger
        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True)
        return loss


    def validation_step(
            self,
            batch: list[torch.Tensor, torch.Tensor],
            batch_idx: int,
            verbose: bool = True,
    ) -> torch.Tensor:
        """
        Function called when using `trainer.validate()` with trainer a lightning `Trainer` instance.
        """
        x, y = batch
        logit_preds = self(x)
        loss = F.cross_entropy(logit_preds, y)
        self.val_accuracy.update(torch.argmax(logit_preds, dim=1), y)
        self.log("val_loss", loss)
        self.log("val_acc", self.val_accuracy, on_epoch=True)
        return loss


    def test_step(
            self,
            batch: list[torch.Tensor, torch.Tensor],
            batch_idx: int,
    ):
        """
        Function called when using `trainer.test()` with trainer a lightning `Trainer` instance.
        """
        x, y = batch
        logit_preds = self(x)
        loss = F.cross_entropy(logit_preds, y)
        self.test_accuracy.update(torch.argmax(logit_preds, dim=1), y)
        self.log_dict({"test_loss": loss, "test_acc": self.test_accuracy})

    def predict_step(
            self,
            batch: list[torch.Tensor, torch.Tensor],
            batch_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Function called when using `trainer.predict()` with trainer a lightning `Trainer` instance.
        """
        x, _ = batch
        logit_preds = self(x)
        softmax_preds = F.softmax(logit_preds, dim=1)
        return x, softmax_preds


if __name__ == "__main__":
    from torchinfo import summary

    inputs_size = (1, 1, 32, 32)
    model = LeNetLitModule(in_channels=1, out_channels=10)
    summary(model, inputs_size)
