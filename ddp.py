# https://blog.csdn.net/weixin_68094467/article/details/141060901
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '5678'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)



class ModelParallelModel(nn.Module):
    def __init__(self):
        super(ModelParallelModel, self).__init__()
        self.fc1 = nn.Linear(10, 10).to('cuda:0')
        self.fc2 = nn.Linear(10, 1).to('cuda:1')

    def forward(self, x):
        x = x.to('cuda:0')
        x = self.fc1(x)
        x = x.to('cuda:1')
        x = self.fc2(x)
        return x



class PipelineParallelModel(nn.Module):
    def __init__(self):
        super(PipelineParallelModel, self).__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        if self.fc1.weight.device != x.device:
            x = x.to(self.fc1.weight.device)
        x = self.fc1(x)
        if self.fc2.weight.device != x.device:
            x = x.to(self.fc2.weight.device)
        x = self.fc2(x)
        return x



def ddp_check(rank, world_size):
    # 初始化进程组
    dist.init_process_group(backend="nccl",
                            rank=rank,
                            world_size=world_size #int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
                            ) # nccl

    # 创建模型并移动到GPU
    model = SimpleModel().to(rank)

    # 包装模型为DDP模型
    ddp_model = DDP(model, device_ids=[rank])


def train_dp(rank, world_size):
    """
    Data Parallelism
    """
    dist.init_process_group(backend='nccl', 
                            init_method='tcp://127.0.0.1:29500', 
                            rank=rank, 
                            world_size=world_size)

    model = SimpleModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    criterion = nn.MSELoss().to(rank)
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

    inputs = torch.randn(64, 10).to(rank)
    targets = torch.randn(64, 1).to(rank)

    outputs = ddp_model(inputs)
    loss = criterion(outputs, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    dist.destroy_process_group()


def train_mp(rank, world_size):
    print(f"Rank: {rank}, World Size: {world_size}")
    dist.init_process_group(backend='nccl', 
                            init_method='tcp://127.0.0.1:29500', 
                            rank=rank, 
                            world_size=world_size
                            )

    model = ModelParallelModel()
    ddp_model = DDP(model, device_ids=[rank], output_device=rank)

    criterion = nn.MSELoss().to('cuda:1')
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

    inputs = torch.randn(64, 10).to('cuda:0')
    targets = torch.randn(64, 1).to('cuda:1')

    outputs = ddp_model(inputs)
    loss = criterion(outputs, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    dist.destroy_process_group()


def train(rank, world_size):
    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:29500', rank=rank, world_size=world_size)

    model = PipelineParallelModel()
    model.fc1.to('cuda:0')
    model.fc2.to('cuda:1')

    ddp_model = DDP(model)

    criterion = nn.MSELoss().to('cuda:1')
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

    inputs = torch.randn(64, 10).to('cuda:0')
    targets = torch.randn(64, 1).to('cuda:1')

    outputs = ddp_model(inputs)
    loss = criterion(outputs, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    dist.destroy_process_group()


if __name__ == '__main__':
    model = SimpleModel()

    # 使用 DataParallel 将模型分布到多个 GPU 上
    # model = nn.DataParallel(model)

    # 世界大小：总共的进程数
    world_size = 4

    # 使用mp.spawn启动多个进程
    # mp.spawn(ddp_check, args=(world_size,), nprocs=world_size, join=True)
    # mp.spawn(train_dp, args=(world_size,), nprocs=world_size, join=True)
    world_size = 2
    # mp.spawn(train_mp, args=(world_size,), nprocs=world_size, join=True)
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
