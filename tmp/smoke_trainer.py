"""BaseTrainer 冒烟测试：合成数据 + 极小模型，验证与新组件的集成。"""
import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from trainers import BaseTrainer

OUT_DIR = Path(__file__).parent / '_smoke_out'


def make_loader(n=64, num_classes=3, batch_size=16):
    x = torch.randn(n, 3, 8, 8)
    y = torch.randint(0, num_classes, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


def run_case(name, optimizer_cfg, scheduler_cfg, epochs=2, eval_interval=1):
    print(f"\n===== case: {name} =====")
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 3))
    trainer = BaseTrainer(
        model=model,
        train_dataloader=make_loader(),
        val_dataloader=make_loader(32),
        num_classes=3,
        epochs=epochs,
        eval_interval=eval_interval,
        optimizer_cfg=optimizer_cfg,
        scheduler_cfg=scheduler_cfg,
        device='cpu',
        output_dir=str(OUT_DIR),
        max_grad_norm=1.0,
        class_names=['cat', 'dog', 'bird'],
    )
    trainer.fit()
    # 断言：指标键名对齐新版 Metrics
    assert 'oa' in trainer.val_metrics_result, trainer.val_metrics_result.keys()
    assert f'recall_{2}' in trainer.val_metrics_result
    # 断言：训练历史与 lr 记录正常（训练阶段只记损失，不算指标）
    assert len(trainer.train_loss_all) <= epochs and len(trainer.train_loss_all) == len(trainer.lr_history)
    # 断言：验证次数与 eval_interval 匹配（val_epochs 记录实际验证轮次）
    assert len(trainer.val_epochs) == len(trainer.val_loss_all) == len(trainer.val_acc_all)
    if eval_interval > 1:
        expected = [e for e in range(1, len(trainer.train_loss_all) + 1)
                    if e % eval_interval == 0 or e == epochs]
        assert trainer.val_epochs == expected, f"val_epochs={trainer.val_epochs}, expected={expected}"
    # 断言：混淆矩阵为 numpy 且尺寸正确
    assert trainer.cnf_matrix is not None and trainer.cnf_matrix.shape == (3, 3)
    # 断言：产物文件齐全
    for fn in ['best.pt', 'last.pt', 'acc_loss.png', 'lr_curve.png',
               'confusion_matrix.png', 'confusion_matrix_normalized.png']:
        assert (trainer.save_dir / fn).exists(), f"missing {fn}"
    # 断言：fit 结束后内存模型已恢复为 best.pt 的权重（早停时尤其关键）
    best_ckpt = torch.load(trainer.save_dir / 'best.pt', weights_only=False, map_location='cpu')
    for k, v in trainer.model.state_dict().items():
        assert torch.equal(v.cpu(), best_ckpt['model'][k]), f"weight mismatch after restore: {k}"
    print(f"case [{name}] OK, scheduler={type(trainer.scheduler).__name__ if trainer.scheduler else None}")


if __name__ == '__main__':
    run_case('adamw + steplr',
             {'type': 'adamw', 'lr': 1e-3},
             {'type': 'steplr', 'step_size': 1, 'gamma': 0.5})
    run_case('sgd + plateau (patience=1, 易触发早停)',
             {'type': 'sgd', 'lr': 1e-2, 'momentum': 0.9},
             {'type': 'reducelronplateau', 'patience': 1})
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
             {'type': 'reducelronplateau', 'patience': 2},
             epochs=3, eval_interval=2)

    shutil.rmtree(OUT_DIR, ignore_errors=True)
    print('\nAll smoke tests passed!')
