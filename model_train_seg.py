import argparse

import torch

from dataset import auto_pin_memory, get_smart_num_workers
from dataset import VOC2012ClassSegLoader
from models.deeplab3plus import DeepLabV3Plus
from models.backbone import ResNet18Encoder
from trainers import BaseTrainer
from utils.hardware import select_device
from loss import CEDiceLoss

torch.set_float32_matmul_precision('medium')
# os.environ['TORCHDYNAMO_VERBOSE'] = '1'



def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Training Script')
    parser.add_argument('--device', default='auto', help='Device to use (auto/cpu/cuda/mps)')
    parser.add_argument('--output_dir', default='checkpoints', help='Output directory for checkpoints')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs to train')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size for training')
    parser.add_argument('--learning_rate', default=2e-4, type=float, help='Learning rate for optimizer')
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

    dm = VOC2012ClassSegLoader(root='./data',
                                batch_size=args.batch_size,
                                pin_memory=pin_memory,
                                num_workers=optimal_workers
                                )
    train_dl = dm.train_dataloader() 
    val_dl = dm.val_dataloader()
    test_dl = dm.test_dataloader()
    num_classes = dm.num_classes

    # DeepLabV3+：backbone → ASPP (neck) → decoder，本仓库自建模块化实现；
    # encoder 加载 ImageNet 预训练权重，与 loader 的 ImageNet 归一化一致；
    # 换 ResNet50Encoder 只需改这一行（通道数由 backbone.out_channels 自动适配）
    model = DeepLabV3Plus(
        backbone=ResNet18Encoder(weights=True),
        num_classes=num_classes,
    )

    # CE + Dice 组合损失：CE 保证逐像素稳定收敛，Dice 直接优化区域重叠（与 miou 监控指标对齐）；
    # ignore_index=255 忽略 VOC 的 void 边界像素
    criterion = CEDiceLoss(ce_weight=1.0, dice_weight=1.0)
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
                  "total_epochs": epochs, 
                  "warmup_epochs": 5}
    # scheduler = build_scheduler(optimizer, sched_cfg)

    # compile model for faster training with pytorch 2.0
    compile_model= args.compile

    tt = BaseTrainer(model=model,
                     device=device,
                     output_dir=output_dir,
                     epochs=epochs,
                     num_classes=num_classes,
                     train_dataloader=train_dl,
                     val_dataloader=val_dl,
                     test_dataloader=test_dl,
                     criterion=criterion,
                     optimizer_cfg=optim_cfg,
                     scheduler_cfg=sched_cfg,
                     compile_model=compile_model,
                     is_classification=False,
                     monitor='miou',
                     eval_interval=1,
                     class_names=dm.classes,
                     # use_tensorboard=True,  # 默认启用；writer 由 trainer 内部创建/关闭，
                     #                        # 日志在 save_dir/tensorboard，查看：
                     #                        # tensorboard --logdir checkpoints
                     )
    tt.fit()
    # fit/test 已解耦：训练结束后手动调用 test()（内部已恢复 best.pt 权重）
    tt.test(report_results=True)