"""BaseTrainer 冒烟测试：合成数据 + 极小模型，验证与新组件的集成。"""
import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from trainers import BaseTrainer
from optimizers import build_optimizer, build_scheduler
from metrics.general import MulticlassClassificationMetric

OUT_DIR = Path(__file__).parent / '_smoke_out'


def make_loader(n=64, num_classes=3, batch_size=16):
    x = torch.randn(n, 3, 8, 8)
    y = torch.randint(0, num_classes, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


def run_case(name, optim_cfg, sched_cfg, epochs=2, eval_interval=1):
    print(f"\n===== case: {name} =====")
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 3))
    train_loader = make_loader()
    optimizer = build_optimizer(model, optim_cfg)
    scheduler = build_scheduler(optimizer, sched_cfg,
                                total_epochs=epochs,
                                steps_per_epoch=len(train_loader))
    trainer = BaseTrainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=make_loader(32),
        test_dataloader=make_loader(32),
        num_classes=3,
        epochs=epochs,
        eval_interval=eval_interval,
        optimizer=optimizer,
        scheduler=scheduler,
        device='cpu',
        output_dir=str(OUT_DIR),
        max_grad_norm=1.0,
        class_names=['cat', 'dog', 'bird'],
        metric=MulticlassClassificationMetric(num_classes=3)
    )
    trainer.fit()
    # fit/test 已解耦：混淆矩阵等产物由 test() 生成
    trainer.test(report_results=False)
    # 断言：指标键名对齐 MulticlassClassificationMetric（键名含 val/ 前缀）
    assert 'val/acc' in trainer.val_metrics_result, trainer.val_metrics_result.keys()
    assert 'val/f1' in trainer.val_metrics_result
    assert 'oa' not in trainer.val_metrics_result
    assert 'acc' in trainer.val_metrics_result  # evaluate_epoch 手动添加的短名
    # 断言：训练历史与 lr 记录正常（训练阶段只记损失，不算指标）
    assert len(trainer.train_loss_all) <= epochs and len(trainer.train_loss_all) == len(trainer.lr_history)
    # 断言：验证次数与 eval_interval 匹配（val_epochs 记录实际验证轮次）
    # 验证指标统一存入 val_metrics_history（键名与 metrics.csv 列名一致）
    val_loss_hist = trainer.val_metrics_history.get('val/loss', [])
    val_acc_hist = trainer.val_metrics_history.get('val/acc', [])
    assert len(trainer.val_epochs) == len(val_loss_hist) == len(val_acc_hist)
    if eval_interval > 1:
        expected = [e for e in range(1, len(trainer.train_loss_all) + 1)
                    if e % eval_interval == 0 or e == epochs]
        assert trainer.val_epochs == expected, f"val_epochs={trainer.val_epochs}, expected={expected}"
    # 断言：混淆矩阵为 numpy 且尺寸正确
    assert trainer.cnf_matrix is not None and trainer.cnf_matrix.shape == (3, 3)
    # 断言：产物文件齐全
    for fn in ['best.pt', 'last.pt', 'loss_curve.png', 'val_metrics.png', 'lr_curve.png',
               'confusion_matrix.png', 'confusion_matrix_normalized.png']:
        assert (trainer.save_dir / fn).exists(), f"missing {fn}"
    # 断言：fit 结束后内存模型已恢复为 best.pt 的权重（早停时尤其关键）
    best_ckpt = torch.load(trainer.save_dir / 'best.pt', weights_only=False, map_location='cpu')
    for k, v in trainer.model.state_dict().items():
        assert torch.equal(v.cpu(), best_ckpt['model'][k]), f"weight mismatch after restore: {k}"
    print(f"case [{name}] OK, scheduler={type(trainer.scheduler).__name__ if trainer.scheduler else None}")


def run_tb_case():
    """TensorBoard 集成：验证标量/逐类指标/batch 级 loss/结构图/超参均写入 event 文件。"""
    print("\n===== case: tensorboard integration =====")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 3))
    train_loader = make_loader()
    optim_cfg = {'type': 'adamw', 'lr': 1e-3}
    sched_cfg = {'type': 'steplr', 'step_size': 1, 'gamma': 0.5}
    optimizer = build_optimizer(model, optim_cfg)
    scheduler = build_scheduler(optimizer, sched_cfg, total_epochs=2, steps_per_epoch=len(train_loader))
    trainer = BaseTrainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=make_loader(32),
        num_classes=3,
        epochs=2,
        eval_interval=1,
        log_interval=1,  # 每个 batch 都写 batch 级标量，便于断言点数
        optimizer=optimizer,
        scheduler=scheduler,
        device='cpu',
        output_dir=str(OUT_DIR),
        class_names=['cat', 'dog', 'bird'],
        metric=MulticlassClassificationMetric(num_classes=3)
        # use_tensorboard 默认 True：writer 由 init_settings 创建，fit 结束时关闭
    )
    tb_dir = trainer.save_dir / 'tensorboard'
    assert trainer.tb_logger.enabled and tb_dir.exists(), "tb_logger not created in init_settings"
    trainer.fit()

    ea = EventAccumulator(str(tb_dir))
    ea.Reload()
    tags = set(ea.Tags()['scalars'])
    # epoch 级标量 + metrics.compute() 全量指标（键名已含 val/ 前缀）+ batch 级 loss
    expected = {'train/epoch_loss', 'train/learning_rate', 'train/batch_loss',
                'train/batch_lr', 'val/epoch_loss', 'val/epoch_acc',
                'val/acc', 'val/f1', 'val/kappa'}
    missing = expected - tags
    assert not missing, f"missing scalar tags: {missing}"
    # batch 级 loss 横轴为 global_step（2 epoch × 4 batch = 8 个点）
    assert len(ea.Scalars('train/batch_loss')) == 8, len(ea.Scalars('train/batch_loss'))
    # 模型结构图已写入
    assert ea.Graph() is not None
    # 超参对照表已写入（add_hparams 的指标会作为标量出现）
    assert f'hparam/best_{trainer.monitor}' in tags, tags
    print("case [tensorboard integration] OK")


if __name__ == '__main__':
    run_case('adamw + steplr',
             {'type': 'adamw', 'lr': 1e-3},
             {'type': 'steplr', 'step_size': 1, 'gamma': 0.5})
    run_case('sgd + plateau (patience=1, 易触发早停)',
             {'type': 'sgd', 'lr': 1e-2, 'momentum': 0.9},
             {'type': 'plateau', 'patience': 1})
    run_case('adam + onecycle (batch-level)',
             {'type': 'adam', 'lr': 1e-3},
             {'type': 'onecyclelr'})
    run_case('adamw + warmup_cosine (builder 新增类型)',
             {'type': 'adamw', 'lr': 1e-3, 'head_lr_scale': 2.0},
             {'type': 'warmup_cosine', 'warmup_epochs': 1, 'total_epochs': 2})
    run_case('no scheduler', {'type': 'adam', 'lr': 1e-3}, None)
    run_case('eval_interval=2 (间隔验证，末轮强制验证)',
             {'type': 'adamw', 'lr': 1e-3},
             {'type': 'steplr', 'step_size': 1, 'gamma': 0.5},
             epochs=3, eval_interval=2)
    run_case('eval_interval=2 + plateau (跳过轮不 step)',
             {'type': 'sgd', 'lr': 1e-2, 'momentum': 0.9},
             {'type': 'plateau', 'patience': 2},
             epochs=3, eval_interval=2)
    run_tb_case()

    shutil.rmtree(OUT_DIR, ignore_errors=True)
    print('\nAll smoke tests passed!')
