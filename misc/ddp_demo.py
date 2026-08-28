"""
# 1. MPS单机单进程训练（Mac设备）
uv run torchrun --nproc_per_node=1 torch_ddp.py
# 输出关键结果：
# Total training time: 73.91s
# Overall throughput: 676.50 samples/sec

# 2. CPU双进程训练（本地设备）
uv run torchrun --nproc_per_node=2 torch_ddp.py
# 输出关键结果：
# Total training time: 526.98s
# Overall throughput: 94.88 samples/sec

# 3. CPU四进程训练（服务器设备）
uv run torchrun --nproc_per_node=4 torch_ddp.py
# 输出关键结果：
# Total training time: 66.00s
# Overall throughput: 757.52 samples/sec

# 4. GPU单机单进程训练（服务器，1块3090）
uv run torchrun --nproc_per_node=1 torch_ddp.py
# 输出关键结果：
# Total training time: 34.80s
# Overall throughput: 1436.94 samples/sec

# 5. GPU单机四进程训练（服务器，4块3090）
uv run torchrun --nproc_per_node=4 torch_ddp.py
# 输出关键结果：
# Total training time: 13.18s
# Overall throughput: 3792.22 samples/sec

# 6. GPU单机八进程训练（服务器，8块3090）
uv run torchrun --nproc_per_node=8 torch_ddp.py
# 输出关键结果：
# Total training time: 8.13s
# Overall throughput: 6149.27 samples/sec
"""
import os
import time
from contextlib import contextmanager
from typing import NamedTuple
import torch
import torch.distributed as dist
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel


class EnvConfig(NamedTuple):
    """分布式环境配置容器"""
    world_size: int  # 总进程数
    rank: int        # 全局进程编号
    local_rank: int  # 本地进程编号
    master_addr: str # 主节点地址
    master_port: str # 主节点端口


def get_env_config() -> EnvConfig:
    """从环境变量自动解析分布式配置，兼容torchrun启动"""
    return EnvConfig(
        world_size=int(os.environ.get('WORLD_SIZE', 1)),
        rank=int(os.environ.get('RANK', 0)),
        local_rank=int(os.environ.get('LOCAL_RANK', 0)),
        master_addr=os.environ.get('MASTER_ADDR', '127.0.0.1'),
        master_port=os.environ.get('MASTER_PORT', '23456')
    )


def get_device_for_ddp(rank: int, world_size: int) -> tuple[torch.device, str, str]:
    """自动匹配设备与通信后端，处理MPS兼容性问题"""
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{os.environ.get("LOCAL_RANK", rank)}')
        return device, 'cuda', 'nccl'  # CUDA优先使用NCCL后端
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        if world_size > 1:
            if rank == 0:
                print("Warning: MPS多进程DDP兼容性有限，降级至CPU训练")
            return _get_cpu_config()
        return torch.device('mps'), 'mps', 'gloo'
    return _get_cpu_config()

def _get_cpu_config() -> tuple[torch.device, str, str]:
    """CPU环境配置辅助函数"""
    return torch.device('cpu'), 'cpu', 'gloo'


def init_process(env_config: EnvConfig) -> None:
    """初始化分布式进程组"""
    _, _, backend = get_device_for_ddp(env_config.rank, env_config.world_size)
    dist.init_process_group(
        backend=backend,
        init_method='env://',  # 从环境变量读取主节点信息
        rank=env_config.rank,
        world_size=env_config.world_size
    )
    if env_config.rank == 0:
        print(f"分布式进程组初始化完成，backend: {backend}, world_size: {env_config.world_size}")


def build_cifar10_dataloader(
    split: str, rank: int, world_size: int, batch_size: int = 32
) -> DataLoader:
    """构建CIFAR-10分布式数据加载器，统一训练/测试逻辑"""
    is_train = split == 'train'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.CIFAR10(
        root='./data', train=is_train, download=True, transform=transform
    )
    
    # 分布式采样器配置
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=is_train
    ) if world_size > 1 else None
    
    _, device_type, _ = get_device_for_ddp(rank, world_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=is_train and sampler is None,
        num_workers=0 if device_type == 'mps' else 4,  # MPS禁用多worker
        pin_memory=(device_type == 'cuda')  # CUDA启用pin_memory加速
    )


def build_model(rank: int, world_size: int) -> tuple[torch.nn.Module, torch.device]:
    """构建分布式模型，自动适配DDP封装"""
    device, device_type, _ = get_device_for_ddp(rank, world_size)
    model = torchvision.models.resnet18(weights=None, num_classes=10).to(device)
    
    if world_size > 1:
        # CUDA设备需指定device_ids，CPU/MPS无需指定
        if device_type == 'cuda':
            return DistributedDataParallel(model, device_ids=[device]), device
        return DistributedDataParallel(model), device
    return model, device


@contextmanager
def ddp_sync_barrier(world_size: int):
    """DDP同步屏障上下文，确保多进程关键节点协同"""
    if world_size > 1:
        dist.barrier()
    try:
        yield
    finally:
        if world_size > 1:
            dist.barrier()


def evaluate_model(
    model: torch.nn.Module, test_loader: DataLoader, device: torch.device, rank: int
) -> tuple[float, float]:
    """分布式模型评估，支持多进程结果聚合"""
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    loss_sum, correct, total = 0.0, 0, 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            batch_size = target.size(0)
            
            loss_sum += loss.item() * batch_size
            correct += (outputs.argmax(dim=1) == target).sum().item()
            total += batch_size
    
    # 多进程结果聚合至主节点
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size > 1:
        metrics = torch.tensor([loss_sum, correct, total], dtype=torch.float32, device=device)
        dist.reduce(metrics, dst=0, op=dist.ReduceOp.SUM)
        if rank == 0:
            loss_sum, correct, total = metrics.tolist()
        else:
            return 0.0, 0.0
    
    avg_loss = loss_sum / total if total > 0 else 0.0
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    
    if rank == 0:
        print(f"\n测试集评估结果：Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%")
    return accuracy, avg_loss


def train(
    epochs: int = 5,
    batch_size: int = 32,
    test_every_epoch: bool = True) -> torch.nn.Module:
    """分布式训练主函数"""
    env_config = get_env_config()
    rank, world_size = env_config.rank, env_config.world_size
    
    # 初始化进程组（仅多进程场景）
    if world_size > 1:
        init_process(env_config)
    
    # 数据与模型准备
    train_loader = build_cifar10_dataloader('train', rank, world_size, batch_size)
    test_loader = build_cifar10_dataloader('test', rank, world_size, batch_size) if test_every_epoch else None
    model, device = build_model(rank, world_size)
    
    # 训练组件配置
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    total_start = time.time()
    
    # 训练循环
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0.0
        
        # 关键：更新sampler epoch确保数据打乱一致性
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # 主进程打印进度（降低日志频率）
            if rank == 0 and batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        # Epoch统计
        avg_epoch_loss = epoch_loss / len(train_loader)
        epoch_time = time.time() - epoch_start
        if rank == 0:
            print(f"Epoch [{epoch+1}/{epochs}] 完成，耗时: {epoch_time:.2f}s, 平均Loss: {avg_epoch_loss:.4f}")
        
        # 分布式评估
        if test_every_epoch and test_loader is not None:
            with ddp_sync_barrier(world_size):
                evaluate_model(model, test_loader, device, rank)
    
    # 模型保存（仅主进程）
    with ddp_sync_barrier(world_size):
        if rank == 0:
            save_path = "resnet18_cifar10_ddp.pth"
            # 处理DDP模型的module属性
            state_dict = model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict()
            torch.save(state_dict, save_path)
            print(f"模型已保存至: {save_path}")
    
    # 清理资源
    if world_size > 1:
        dist.destroy_process_group()
    
    if rank == 0:
        total_time = time.time() - total_start
        print(f"\n训练完成，总耗时: {total_time:.2f}s, 平均每epoch耗时: {total_time/epochs:.2f}s")
    return model



if __name__ == "__main__":
    train(epochs=5, batch_size=32, test_every_epoch=True)