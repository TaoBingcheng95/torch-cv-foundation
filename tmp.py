import torch
import torchvision
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR

# 假设已经定义了 model, optimizer, num_epochs=50
num_epochs = 50
device = torch.device('mps')
model = torchvision.models.resnet18(weights=None, num_classes=10).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

# --- 学习率调度配置 ---
warmup_epochs = 10 
decay_epochs = 40 # 50 - 10

# 初始学习率
initial_lr = 1e-3 

# 设置初始学习率，以便 WarmupLR 可以从它开始
for param_group in optimizer.param_groups:
    param_group['lr'] = 0.0 # 从 0 开始热身

# Warmup 调度器：从 0 到 initial_lr 线性增加
scheduler_warmup = LinearLR(
    optimizer, 
    start_factor=1e-6, # 从非常小的数开始
    end_factor=1.0,    # 结束在 initial_lr
    total_iters=warmup_epochs 
)

# Decay 调度器：余弦退火
scheduler_decay = CosineAnnealingLR(
    optimizer, 
    T_max=decay_epochs, # 衰减持续时间
    eta_min=1e-6 # 最小学习率
)

# 使用 SequentialLR 串联（官方推荐方式）
sequential_scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer, 
    schedulers=[scheduler_warmup, scheduler_decay],
    milestones=[warmup_epochs] # 在第 10 轮结束时切换
)

print(sequential_scheduler.state_dict())
