"""model_train_seg.py 的冒烟验证脚本

与正式脚本同构（同模型/损失/优化器/调度器/Trainer 配置），
仅用 Subset 截取少量样本、跑 2 个 epoch，快速验证
fit → best.pt/last.pt → test 全链路是否畅通。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader, Subset

from dataset.voc_dataset import VOCSegmentationDataLoader
from models.deeplab3plus import DeepLabV3Plus
from models.backbone import ResNet18Encoder
from trainers import BaseTrainer
from optimizers import build_optimizer, build_scheduler
from utils.hardware import select_device
from loss import CEWithLogitsLoss
from metrics.general import MulticlassSegmentationMetric

torch.set_float32_matmul_precision('medium')

if __name__ == '__main__':
    device = select_device('auto')
    print(f"Using device: {device}")

    dm = VOCSegmentationDataLoader(root='./data', batch_size=4, num_workers=0)
    num_classes = dm.num_classes

    # 截取少量样本，快速跑通全流程
    train_ds = Subset(dm.data_train, range(16))
    val_ds = Subset(dm.data_val, range(8))
    test_ds = Subset(dm.data_test, range(8))
    train_dl = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=4, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=4, shuffle=False)

    model = DeepLabV3Plus(
        backbone=ResNet18Encoder(weights=True),
        num_classes=num_classes,
    )

    criterion = CEWithLogitsLoss()
    optim_cfg = {"type": "adamw", "lr": 2e-4,
                 "weight_decay": 1e-4, "betas": (0.9, 0.999)}
    epochs = 2
    sched_cfg = {"type": "warmup_cosine",
                 "total_epochs": epochs, "warmup_epochs": 1}

    optimizer = build_optimizer(model, optim_cfg)
    scheduler = build_scheduler(optimizer, sched_cfg,
                                total_epochs=epochs,
                                steps_per_epoch=len(train_dl))

    tt = BaseTrainer(model=model,
                     device=device,
                     output_dir='tmp/smoke_ckpt',
                     epochs=epochs,
                     num_classes=num_classes,
                     train_dataloader=train_dl,
                     val_dataloader=val_dl,
                     test_dataloader=test_dl,
                     criterion=criterion,
                     metric=MulticlassSegmentationMetric(
                         num_classes=num_classes, ignore_index=255),
                     optimizer=optimizer,
                     scheduler=scheduler,
                     is_classification=False,
                     monitor='val/iou',
                     eval_interval=1,
                     class_names=dm.classes,
                     )
    tt.fit()
    results = tt.test(report_results=True)
    print(f"\n✅ SMOKE OK | test loss={results['loss']:.4f} "
          f"oa={results['acc']:.4f} samples={results['samples']}")
