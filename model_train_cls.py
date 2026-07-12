# https://mp.weixin.qq.com/s/ZsKwD-Cb1ynqvCdBIWlZgw
import os
import psutil

import torch
# from torchvision.datasets import MNIST
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from dataset.components import MNISTDataLoader, FashionMNISTDataLoader
# from dataset.mnist_datamodule import MNISTDataModule
from dataset.mnist_datamodule import MNISTDataModule
from models.components import SimpleDenseNet, LeNet5
# from metrics import Metrics
from trainers import BaseTrainer

torch.set_float32_matmul_precision('high')
os.environ['TORCHDYNAMO_VERBOSE'] = '1' # 避免显存碎片化导致的 OOM 错误



def get_smart_num_workers():
    """获取一个智能且稳健的 num_workers 基准值。"""
    # 1. 获取物理核心数
    physical_cores = psutil.cpu_count(logical=False)
    if physical_cores is None:
        physical_cores = os.cpu_count() or 1

    # 2. 根据物理核心数设置一个安全的基准值 (例如，不超过核心数，且最大值设为8)
    # 对于大多数消费级CPU，8个worker通常足够
    safe_value = min(physical_cores, 8)
    
    # 3. 为关键任务保留一个核心
    if safe_value > 1:
        safe_value -= 1
        
    return max(1, safe_value) # 至少为1


if __name__ == '__main__':
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    output_dir='checkpoints\\FashionMNIST_LeNet5'
    ephocs = 100

    if device.type == 'cpu':
        pin_memory = False
    else:
        pin_memory = True
    optimal_workers = get_smart_num_workers()
    print(f"根据您的硬件，推荐的基准 num_workers 值为: {optimal_workers}")

    # dm = MNISTDataModule(data_dir='./data',
    #                      batch_size=8,
    #                      num_workers=0, # 4 * num_GPU
    #                      pin_memory=True,
    #                      resize_32=True)
    # dm.prepare_data()
    # dm.setup()
    # train_dl = dm.train_dataloader()
    # val_dl = dm.val_dataloader()
    # test_dl = dm.test_dataloader()
    # num_calsses = dm.num_classes

    dm = FashionMNISTDataLoader(root='./data',
                                download=True,
                                use_normalize=True,
                                val_split=0.2,
                                batch_size=16,
                                pin_memory=pin_memory, # torch.cuda.is_available()
                                num_workers=0
                                )
    train_dl = dm.train_dataloader()
    val_dl = dm.val_dataloader()
    test_dl = dm.test_dataloader()
    num_calsses = dm.num_classes

    model = LeNet5() # SimpleDenseNet(output_size=dm.num_classes)
    criterion = nn.CrossEntropyLoss()
    learning_rate = 1e-3
    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.1)

    # compile model for faster training with pytorch 2.0
    compile_model= False

    tt = BaseTrainer(model=model,
                     device='cuda:0' if torch.cuda.is_available() else 'cpu',
                     output_dir=output_dir,
                     # resume=resume,
                     epochs=ephocs,
                     
                     num_classes=num_calsses,
                     train_dataloader=train_dl,
                     val_dataloader=val_dl,
                     test_dataloader=test_dl,

                     criterion=criterion,
                     optimizer_cfg={
                        "type": "adamw",
                        "lr": learning_rate, # 5e-4,
                        "weight_decay": 1e-4,
                        "betas": (0.9, 0.999),},
                    scheduler_cfg={
                        "type": "reduceLROnPlateau",
                        "mode": "min",
                        "patience": 5,
                        "factor": 0.5,},

                    compile_model=compile_model,
                    )
    tt.fit()
