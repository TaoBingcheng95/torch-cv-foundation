import os
import torch
from torch import nn, optim
import sys
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split

from dataset.tianchi import TianchiDataset
from models.mynet.Unet import UNet

if __name__ == '__main__':


    model = UNet(n_channels=3, n_classes=2)
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    lr = 0.001
    optimizer = optim.SGD((param for param in model.parameters() if param.requires_grad),
                          lr=lr,
                          weight_decay=0)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    Tianchi_dir = 'D:\\myspace\\dataset\\segemnt\\tianchi\\train'
    val_ratio = 0.4
    test_ratio = 0.2
    num_classes = 2
    ds = TianchiDataset(root=Tianchi_dir, img_folder='image', label_folder='label')
    train_ds, val_ds, test_ds = random_split(ds,
                                             [len(ds) - int(len(ds) * val_ratio) - int(len(ds) * test_ratio),
                                              int(len(ds) * val_ratio),
                                              int(len(ds) * test_ratio)])

    batch_size = 16
    if sys.platform.startswith('win'):
        num_workers = 0
    else:
        num_workers = 4
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
    # x, y = next(iter(train_dl))

    train_loss = 0.0
    # train_acc = 0.0
    total_correct = 0
    total_samples = 0
    test_count = len(test_ds)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_dl, total=test_count, desc='Test'):
        # for (batch_idx, batch) in enumerate(test_dl):
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            bs = targets.size(0)

            # # 清零梯度
            # optimizer.zero_grad()
            # # 前向传播
            outputs = model(inputs)
            # # 计算损失
            # loss = criterion(outputs, target)
            # # 反向传播
            # loss.backward()
            # # 更新参数
            # optimizer.step()
            # train_loss += loss.item() * bs

            # 使用 torch.argmax 找到模型输出中预测的类别（最大概率的类别索引）
            preds = torch.argmax(outputs, dim=1)
            # 计算本批次预测正确的样本数，并累加
            correct = torch.sum((preds == targets.data).float().mean())
            total_correct += correct.item()
            total_samples += bs

        # lr_scheduler.step()
        # train_loss /= total_samples
        train_acc = total_correct / total_samples
        print(total_correct)
        print(total_samples)
        print(train_acc)
        # self.train_loss_all.append(train_loss)
        # self.train_acc_all.append(train_acc)
        # logger.info(f"Train Loss: {train_loss:.4f}")
        # logger.info(f"Train Acc: {train_acc:.4f}")


