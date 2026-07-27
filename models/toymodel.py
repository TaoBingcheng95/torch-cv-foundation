# 定义简单的神经网络模型用于测试
import torch
from torch import nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader,BatchSampler
from torch.utils.data.distributed import DistributedSampler



class ToyModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(1, 1)

    def forward(self, x):
        return self.layer(x)



class MyDataset(Dataset):

    def __init__(self):
        super().__init__()
        self.data = torch.tensor([1, 2, 3, 4], dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index:index + 1]



class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.layer = nn.Linear(10, 2)
        self.layer2 = nn.Linear(2, 10)

    def forward(self, input):
        return self.layer(input) 



class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

