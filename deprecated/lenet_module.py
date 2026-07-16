from typing import Tuple, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer

import torchmetrics
import lightning.pytorch as pl

from models.components import LeNet5



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
            batch: Tuple[torch.Tensor, torch.Tensor],
            batch_idx: int,
    ) -> torch.Tensor:
        """
        Function called when using `trainer.fit()` with trainer a lightning `Trainer` instance.
        """
        x, y = batch
        logit_preds = self(x)
        loss = F.cross_entropy(logit_preds, y)
        self.train_accuracy.update(torch.argmax(logit_preds, dim=1), y)
        self.log("train_acc_step", self.train_accuracy, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        # logs metrics for each training_step, and the average across the epoch, to the progress bar and logger
        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True, prog_bar=True)
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
        return softmax_preds



class LeNet5LitModule(pl.LightningModule):
    def __init__(
        self, 
        in_channels: int = 1, 
        num_classes: int = 10, 
        lr: float = 1e-3,
        optimizer_type: str = "adam"
    ):
        """
        LightningModule 是连接模型与 Trainer 的桥梁。
        它不再定义网络层，而是实例化之前的 LeNet5。
        """
        super().__init__()
        self.save_hyperparameters()  # 自动保存参数，方便复现和 checkpoint 加载
        
        # 实例化基础模型 (nn.Module)
        self.model = LeNet5(num_classes=num_classes)
        
        # 定义损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 定义评估指标 (torchmetrics)
        # Lightning 会自动处理 metric 的 reset() 和 compute()
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

        # 示例输入数组，用于打印模型结构摘要
        self.example_input_array = torch.zeros(1, in_channels, 32, 32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 直接调用内部模型的 forward
        return self.model(x)

    def configure_optimizers(self) -> Optimizer:
        """
        配置优化器。Lightning 会自动调用此函数。
        """
        if self.hparams.optimizer_type == "adam":
            return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        elif self.hparams.optimizer_type == "sgd":
            return torch.optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {self.hparams.optimizer_type}")

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        单个训练步骤。无需手动 zero_grad, backward, step，Lightning 自动处理。
        无需手动 .to(device)，Lightning 自动处理。
        """
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # 更新指标
        preds = torch.argmax(logits, dim=1)
        self.train_acc.update(preds, y)
        
        # 记录日志
        # on_step=True: 每个 batch 记录一次 (进度条会动)
        # on_epoch=True: 每个 epoch 结束记录一次平均值
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", self.train_acc, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        self.val_acc.update(preds, y)
        
        # 验证集通常只看 epoch 平均值
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", self.val_acc, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        self.test_acc.update(preds, y)
        
        self.log("test_loss", loss, on_epoch=True, logger=True)
        self.log("test_acc", self.test_acc, on_epoch=True, logger=True)

    def predict_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        用于推理。返回概率分布。
        """
        x, _ = batch
        logits = self(x)
        probs = F.softmax(logits, dim=1)
        return probs



if __name__ == "__main__":
   
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    from torchinfo import summary

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    inputs_size = (1, 1, 32, 32)
    model = LeNet5LitModule(in_channels=1, out_channels=10)
    summary(model, inputs_size)

    train_dataset = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST(root='../data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # 初始化 LightningModule
    lit_model = LeNetLitModule(in_channels=1, num_classes=10, lr=1e-3, optimizer_type="adam")

    # 初始化 Trainer (核心部分)
    # max_epochs: 最大训练轮数
    # accelerator: 'gpu', 'cpu', 'tpu' 自动选择
    # devices: 使用几个设备
    # log_every_n_steps: 日志频率
    trainer = pl.Trainer(
        max_epochs=5,
        accelerator="auto",  # 自动检测 GPU/CPU
        devices="auto",      # 自动检测可用设备数量
        log_every_n_steps=50,
        enable_checkpointing=True, # 自动保存 checkpoint
    )

    print("Start Training...")
    trainer.fit(lit_model, train_loader, val_loader)
    
    # 测试与预测
    # 加载测试集
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    print("Start Testing...")
    trainer.test(lit_model, test_loader)
    
    # 预测示例
    # predictions = trainer.predict(lit_model, test_loader)
    # print(f"Prediction shape: {predictions[0].shape}")
