# https://mp.weixin.qq.com/s/ZsKwD-Cb1ynqvCdBIWlZgw
import os
from tkinter.constants import N
import psutil

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from dataset import MNISTDataLoader, FashionMNISTDataLoader
from models import AlexNet, LeNet5
from trainers import BaseTrainer
# from optimizers import build_optimizer, build_scheduler, clip_grad_norm
from loss import CEWithLogitsLoss


torch.set_float32_matmul_precision('medium')
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
    output_dir='checkpoints'
    ephocs = 100

    if device.type == 'cpu':
        pin_memory = False
    else:
        pin_memory = True
    
    optimal_workers = get_smart_num_workers()
    print(f"根据您的硬件，推荐的基准 num_workers 值为: {optimal_workers}")

    dm = FashionMNISTDataLoader(root='./data',
                                download=True,
                                use_normalize=True,
                                val_split=0.2,
                                batch_size=16,
                                pin_memory=pin_memory, # torch.cuda.is_available()
                                num_workers=optimal_workers
                                )
    train_dl = dm.train_dataloader()
    val_dl = dm.val_dataloader()
    test_dl = dm.test_dataloader()
    num_calsses = dm.num_classes

    x, y = next(iter(train_dl))
    print(x.shape, y.shape)

    model = LeNet5(num_classes=num_calsses) # SimpleDenseNet(output_size=dm.num_classes)
    criterion = CEWithLogitsLoss()  # nn.CrossEntropyLoss()
    learning_rate = 1e-3
    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.1)
    optim_cfg = {
        "type": "adamw",
        "lr": learning_rate, # 5e-4,
        "weight_decay": 1e-4,
        "betas": (0.9, 0.999),}
    # optimizer = build_optimizer(model, optim_cfg)
    sched_cfg = {
        "type": "reduceLROnPlateau",
        "mode": "min",
        "patience": 5,
        "factor": 0.5,}
    # scheduler = build_scheduler(optimizer, sched_cfg)

    # compile model for faster training with pytorch 2.0
    compile_model= False

    tt = BaseTrainer(model=model,
                     device=device,
                     output_dir=output_dir,
                     # resume=resume,
                     epochs=ephocs,
                     
                     num_classes=num_calsses,
                     train_dataloader=train_dl,
                     val_dataloader=val_dl,
                     test_dataloader=test_dl,

                     criterion=criterion,
                     optimizer_cfg=optim_cfg,
                     scheduler_cfg=sched_cfg,

                    compile_model=compile_model,
                    )
    tt.fit()
