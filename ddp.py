
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '5678'

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


def main(rank, world_size):
    # 初始化进程组
    dist.init_process_group(backend="nccl",
                            rank=rank,
                            world_size=int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1) # nccl

    # 创建模型并移动到GPU
    model = SimpleModel().to(rank)

    # 包装模型为DDP模型
    ddp_model = DDP(model, device_ids=[rank])


if __name__ == '__main__':
    model = SimpleModel()

    # 使用 DataParallel 将模型分布到多个 GPU 上
    # model = nn.DataParallel(model)

    import torch.multiprocessing as mp

    print(torch.__config__.show())

    # 世界大小：总共的进程数
    world_size = 4

    # 使用mp.spawn启动多个进程
    # mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
