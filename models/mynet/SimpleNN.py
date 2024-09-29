# https://mp.weixin.qq.com/s/uEzUqjOjoWRbPZhLE24nsg
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

class MyNet(nn.Module):

    def __init__(self, in_channels=3, num_classes=10) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.model(x)
        return x


class Net(nn.Module):
    # 定义Net的初始化函数，这个函数定义了该神经网络的基本结构
    def __init__(self):
        super(Net, self).__init__()
        # 复制并使用Net的父类的初始化方法，即先运行nn.Module的初始化函数
        self.conv1 = nn.Conv2d(3, 6, 5)
        # 定义conv1函数是图像卷积函数：输入为3张特征图
        # 输出为 6幅特征图， 卷积核为5×5的正方形
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 定义conv2函数的是图像卷积函数：输入为6幅特征图，输出为16幅特征图
        # 卷积核为5×5的正方形
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # 定义fc1（fullconnect）全连接函数1为线性函数：y = Wx + b
        # 并将16×5×5个节点连接到120个节点上
        self.fc2 = nn.Linear(120, 84)
        # 定义fc2（fullconnect）全连接函数2为线性函数：y = Wx + b
        # 并将120个节点连接到84个节点上
        self.fc3 = nn.Linear(84, 10)
        # 定义fc3（fullconnect）全连接函数3为线性函数：y = Wx + b
        # 并将84个节点连接到10个节点上

    # 定义该神经网络的向前传播函数，该函数必须定义
    # 一旦定义成功，向后传播函数也会自动生成（autograd）
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 输入x经过卷积conv1之后，经过激活函数ReLU
        # 使用2×2的窗口进行最大池化，然后更新到x
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # 输入x经过卷积conv2之后，经过激活函数ReLU
        # 使用2×2的窗口进行最大池化，然后更新到x
        x = x.view(-1, self.num_flat_features(x))
        # view函数将张量x变形成一维的向量形式
        # 总特征数并不改变，为接下来的全连接作准备
        x = F.relu(self.fc1(x))
        # 输入x经过全连接1，再经过ReLU激活函数，然后更新x
        x = F.relu(self.fc2(x))
        # 输入x经过全连接2，再经过ReLU激活函数，然后更新x
        x = self.fc3(x)
        # 输入x经过全连接3，然后更新x
        return x

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # 输入层到隐藏层
        self.fc2 = nn.Linear(128, 10)       # 隐藏层到输出层

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # 展平输入
        x = torch.relu(self.fc1(x))  # 激活函数ReLU
        x = self.fc2(x)
        return x


class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(CustomLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        return torch.matmul(x, self.weight.t()) + self.bias


if __name__ == '__main__':
    # 创建模型实例
    model = SimpleNN()

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 示例输入
    inputs = torch.randn(1, 28, 28)  # 随机生成一个28x28的输入
    output = model(inputs)  # 前向传播
    loss = criterion(output, torch.tensor([3]))  # 假设真实标签为3

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("输出：", output)
    print("损失：", loss.item())