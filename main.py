import os
import time
import json
from datetime import datetime
from typing import Dict, Optional

import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from models import LeNet5

mnist_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((32,32)),
     transforms.Normalize(mean=(0.1307),std= (0.3081))
     ])


def train_one_epoch(model: nn.Module, 
                    train_loader: DataLoader, 
                    criterion: nn.Module, 
                    optimizer: torch.optim.Optimizer, 
                    device: torch.device,
                    epoch: int,
                    writer: Optional[SummaryWriter] = None,
                    log_interval: int = 200
                    ):
    """
    Here, we use enumerate(train_loader) instead of iter(train_loader) so that we can track the batch index and do some intra-epoch reporting
    """
    model.train()  # 设置模型为训练模式 (开启 Dropout, BN 等)
    running_loss = 0.
    total_steps = len(train_loader)
    start_time = time.time()

    # 记录当前学习率
    current_lr = optimizer.param_groups[0]['lr']

    for idx, (inputs, labels) in enumerate(train_loader):
        # Every data instance is an input + label pair
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()
        # Make predictions for this batch
        outputs = model(inputs)
        # Compute the loss and its gradients
        loss = criterion(outputs, labels)
        #Backward and optimize
        loss.backward()
        # Adjust learning weights
        optimizer.step()

        # 累加损失 (使用 .item() 获取标量值，避免构建计算图)
        running_loss += loss.item()

        # 记录 batch 级别指标
        if writer is not None and (idx + 1) % log_interval == 0:
            global_step = epoch * total_steps + idx + 1
            writer.add_scalar("train/batch_loss", loss.item(), global_step)
            writer.add_scalar("train/learning_rate", current_lr, global_step)

        if (idx+1) % log_interval == 0:
            elapsed = time.time() - start_time
            samples_per_sec = (idx + 1) * train_loader.batch_size / elapsed
            print(f'  Batch [{idx+1}/{total_steps}], Loss: {loss.item():.4f}, '
                  f'Speed: {samples_per_sec:.0f} samples/sec')
    # 计算并返回 epoch 级别指标
    avg_loss = running_loss/total_steps
    epoch_time = time.time() - start_time

    # 记录 epoch 级别指标
    if writer is not None:
        writer.add_scalar("train/epoch_loss", avg_loss, epoch)
        writer.add_scalar("train/epoch_time", epoch_time, epoch)
    
    return {
        "loss": avg_loss,
        "time": epoch_time,
        "lr": current_lr
    }


@torch.no_grad()
def validate_one_epoch(model: nn.Module,
                       val_loader: DataLoader,
                       criterion: nn.Module,
                       device: torch.device,
                       epoch: int,
                       writer: Optional[SummaryWriter] = None
                       ):
    """
    Disable gradient computation and reduce memory consumption.
    """
    model.eval()  # 设置模型为评估模式 (关闭 Dropout, BN 使用统计值)
    running_loss = 0.0
    correct = 0
    total = 0
    total_steps = len(val_loader)
    start_time = time.time()

    for (inputs, labels) in val_loader:

        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        running_loss += loss.item()

        # 计算准确率
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = running_loss / total_steps
    accuracy = 100 * correct / total
    epoch_time = time.time() - start_time

    # 记录验证集指标
    if writer is not None:
        writer.add_scalar("val/epoch_loss", avg_loss, epoch)
        writer.add_scalar("val/epoch_acc", accuracy, epoch)
        writer.add_scalar("val/epoch_time", epoch_time, epoch)

    return {
        "loss": avg_loss,
        "acc": accuracy,
        "time": epoch_time}



if __name__ == '__main__':

    # 🔧 超参数配置（集中管理，方便复现）
    CONFIG = {
        "exp_name": "MNIST_LeNet5",
        "batch_size": 64,
        "learning_rate": 1e-3,
        "epochs": 10,
        "seed": 42,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "log_interval": 200,
        "early_stop_acc": 98.0,  # 提前停止阈值
    }
    # 设置随机种子（保证复现性）
    torch.manual_seed(CONFIG["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CONFIG["seed"])
    print(f"🚀 Using device: {CONFIG['device']}")
    print(f"📋 Config: {json.dumps(CONFIG, indent=2)}")

    # 准备日志和检查点目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f"./logs/{CONFIG['exp_name']}/{timestamp}"
    ckpt_dir = f"./checkpoints/{CONFIG['exp_name']}/{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    # 保存配置快照
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump(CONFIG, f, indent=2)
    
    # 初始化 TensorBoard Writer
    writer = SummaryWriter(log_dir=log_dir)
    print(f"📊 TensorBoard logs: {log_dir}")
    print(f"💡 启动命令: tensorboard --logdir ./logs")

    training_set = MNIST(root='data', train=True, transform=mnist_transform, download=True)
    validation_set = MNIST(root='data', train=False, transform=mnist_transform, download=True)
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    print(f'Training set has {len(training_set)} instances')
    print(f'Validation set has {len(validation_set)} instances')
    train_loader = DataLoader(dataset=training_set, batch_size=CONFIG["batch_size"],  shuffle=True)
    val_loader = DataLoader(dataset=validation_set, batch_size=CONFIG["batch_size"], shuffle=False )
    
    device = torch.device(CONFIG["device"])
    model = LeNet5().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3)

    # 训练循环
    best_vloss = float('inf')
    best_acc = 0.0
    patience_counter = 0  # 早停计数器

    for epoch in range(1, CONFIG["epochs"] + 1):
        print(f'\n{"="*50}\n Epoch {epoch}/{CONFIG["epochs"]}\n{"="*50}')
        epoch_start = time.time()

        # Make sure gradient tracking is on, and do a pass over the data
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, 
            device, epoch, writer=writer, log_interval=CONFIG["log_interval"])
        
        # Set the model to evaluation mode, disabling dropout and using population statistics for batch normalization.
        val_metrics = validate_one_epoch(model, val_loader, criterion, device, epoch, writer=writer)

        # 调度器步长（ReduceLROnPlateau 需要手动 step）
        scheduler.step(val_metrics["loss"])

        # print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        epoch_time = time.time() - epoch_start
        print(f'\n📈 Epoch Summary:')
        print(f'  Train Loss: {train_metrics["loss"]:.4f} | Time: {train_metrics["time"]:.1f}s')
        print(f'  Val   Loss: {val_metrics["loss"]:.4f} | Acc: {val_metrics["acc"]:.2f}% | Time: {val_metrics["time"]:.1f}s')
        print(f'  LR: {train_metrics["lr"]:.2e} | Total Epoch Time: {epoch_time:.1f}s')
        # 记录汇总指标到 TensorBoard
        if writer is not None:
            writer.add_scalar("summary/epoch_time", epoch_time, epoch)
            writer.add_scalar("summary/learning_rate", train_metrics["lr"], epoch)

        # 保存模型
        last_model_path = os.path.join(ckpt_dir, 'last.pt')
        best_loss_path = os.path.join(ckpt_dir, 'best_loss.pt')
        best_acc_path = os.path.join(ckpt_dir, 'best_acc.pt')
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            }, last_model_path)

        # 基于损失保存最佳模型
        if val_metrics["loss"] < best_vloss:
            best_vloss = val_metrics["loss"]
            torch.save(model.state_dict(), best_loss_path)
            print(f'  ✨ New best model saved by loss! (Val Loss: {best_vloss:.4f})')
            if writer:
                writer.add_scalar("best/val_loss", best_vloss, epoch)
            patience_counter = 0  # 重置早停计数
        else:
            patience_counter += 1
        
        # 基于准确率保存最佳模型
        if val_metrics["acc"] > best_acc:
            best_acc = val_metrics["acc"]
            torch.save(model.state_dict(), best_acc_path)
            print(f'  ✨ New best model saved by acc! (Val Acc: {best_acc:.2f}%)')
            if writer:
                writer.add_scalar("best/val_acc", best_acc, epoch)
        
        # 早停检查
        if patience_counter >= 5:
            print(f'  ⚠️ Early stopping triggered (no improvement for {patience_counter} epochs)')
            break
        
        if val_metrics["acc"] >= CONFIG["early_stop_acc"]:
            print(f"  🎯 Accuracy reached {CONFIG['early_stop_acc']}%, stopping early.")
            break
  
    writer.close()
    
    print(f"\n{'='*50}")
    print(f"✅ Training Finished!")
    print(f"📁 Checkpoints: {os.path.abspath(ckpt_dir)}")
    print(f"📊 Logs: {os.path.abspath(log_dir)}")
    print(f"🏆 Best Val Loss: {best_vloss:.4f}")
    print(f"🏆 Best Val Acc : {best_acc:.2f}%")
    print(f"💡 View logs: tensorboard --logdir ./logs")
