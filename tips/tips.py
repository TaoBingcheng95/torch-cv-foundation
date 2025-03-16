# https://www.zhihu.com/question/4031711262

import torch
from torch import nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        # parameters会自动被追踪
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

# 简化版的反向传播
def my_backward():
    # 1. 计算所有参数的梯度
    grad_output = torch.ones_like(loss)
    # 2. 通过计算图反向传播
    for param in model.parameters():
        if param.requires_grad:
            param.grad = compute_grad(param, grad_output)


# 简化版的SGD优化器
class SimpleSGD:
    def __init__(self, parameters, lr):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        with torch.no_grad():  # 不要记录计算图
            for param in self.parameters:
                if param.grad is not None:
                    # 更新参数
                    param.data -= self.lr * param.grad


# 训练一个批次
def train_step(model, data, optimizer):
    # 1. 前向传播
    output = model(data)
    loss = criterion(output, target)

    # 2. 反向传播
    optimizer.zero_grad()  # 清零历史梯度
    loss.backward()  # 计算新梯度

    # 3. 参数更新
    optimizer.step()  # 用梯度更新参数

    return loss.item()


if __name__ == "__main__":
    model = SimpleModel()
    # 看看模型的参数
    # for name, param in model.named_parameters():
    #     print(f"{name}: {param.requires_grad}")  # 默认requires_grad=True

    x = torch.randn(1, 10, requires_grad=True)
    output = model(x)
    loss = output.mean()

    # 看看计算图
    print(f"Loss grad_fn: {loss.grad_fn}")  # 会显示计算历史
