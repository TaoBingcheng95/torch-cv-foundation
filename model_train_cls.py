import os
import sys
import psutil
import argparse

import torch
# from torch import nn 
# from torch.utils.tensorboard import SummaryWriter

from dataset import auto_pin_memory, get_smart_num_workers, MNISTDataLoader, FashionMNISTDataLoader, CIFAR10DataLoader
from models import AlexNet, LeNet5, build_vgg
from trainers import BaseTrainer
# from optimizers import build_optimizer, build_scheduler, clip_grad_norm
from loss import CEWithLogitsLoss

from utils.hardware import select_device

torch.set_float32_matmul_precision('medium')
# os.environ['TORCHDYNAMO_VERBOSE'] = '1'



def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Training Script')
    parser.add_argument('--device', default='auto', help='Device to use (auto/cpu/cuda/mps)')
    parser.add_argument('--output_dir', default='checkpoints', help='Output directory for checkpoints')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs to train')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size for training')
    parser.add_argument('--learning_rate', default=1e-3, type=float, help='Learning rate for optimizer')
    parser.add_argument('--resume', default='', help='Path to the checkpoint to resume from')
    parser.add_argument('--compile', action='store_true', help='Compile model for faster training')
    return parser.parse_args()



if __name__ == '__main__':

    args = parse_args()

    device = select_device(args.device)
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    output_dir=args.output_dir
    epochs = args.epochs
    learning_rate = args.learning_rate

    optimal_workers = get_smart_num_workers()
    pin_memory = auto_pin_memory(device, num_workers=optimal_workers)
    
    print(f"根据硬件环境，推荐的基准 num_workers 值为: {optimal_workers}, pin_memory={pin_memory}")

    dm = FashionMNISTDataLoader(root='./data',
                                download=False,
                                use_normalize=True,
                                val_split=0.2,
                                batch_size=args.batch_size,
                                pin_memory=pin_memory,
                                num_workers=optimal_workers
                                )
    train_dl = dm.train_dataloader()
    val_dl = dm.val_dataloader()
    test_dl = dm.test_dataloader()
    num_classes = dm.num_classes
    x, y = next(iter(train_dl))
    print(x.shape, y.shape)

    model = LeNet5(num_classes=num_classes)
    # model = build_vgg(arch='vgg11')
    criterion = CEWithLogitsLoss()  # nn.CrossEntropyLoss()
    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.1)
    optim_cfg = {
        "type": "adamw",
        "lr": learning_rate, # 5e-4,
        "weight_decay": 1e-4,
        "betas": (0.9, 0.999),}
    # optimizer = build_optimizer(model, optim_cfg)
    # sched_cfg = {
    #     "type": "reduceLROnPlateau",
    #     "mode": "min",  # 与 BaseTrainer 默认 monitor='acc'（max 方向）对齐
    #     "patience": 5,
    #     "factor": 0.5,}
    sched_cfg =  {"type": "warmup_cosine",
                  "total_epochs": 30, 
                  "warmup_epochs": 5}
    # scheduler = build_scheduler(optimizer, sched_cfg)

    # compile model for faster training with pytorch 2.0
    compile_model= args.compile

    tt = BaseTrainer(model=model,
                     device=device,
                     output_dir=output_dir,
                     # resume=resume,
                     epochs=epochs,
                     num_classes=num_classes,
                     train_dataloader=train_dl,
                     val_dataloader=val_dl,
                     test_dataloader=test_dl,
                     criterion=criterion,
                     optimizer_cfg=optim_cfg,
                     scheduler_cfg=sched_cfg,
                    compile_model=compile_model,
                    )
    tt.fit()
    # fit/test 已解耦：训练结束后手动调用 test()（内部已恢复 best.pt 权重）
    tt.test(report_results=True, save_predictions=True)
