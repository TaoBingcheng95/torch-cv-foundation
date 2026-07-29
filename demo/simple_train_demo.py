import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(0)

# 定义一个简单的神经网络
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.linear = nn.Linear(1, 1)  # 输入和输出都是1维

    def forward(self, x):
        return self.linear(x)


if __name__ == '__main__':
    # 生成一些随机数据
    input_data = torch.randn(100, 1)  # 100个样本，每个样本有2个特征
    target_data = 2 * input_data + 1  # 简单的线性关系

    # 创建数据加载器
    dataset = TensorDataset(input_data, target_data)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    # 实例化模型、损失函数和优化器
    model = SimpleNet()
    criterion = nn.MSELoss()  # 均方误差损失
    optimizer = optim.SGD(model.parameters(), lr=0.01)  # 随机梯度下降优化器

    # 训练模型
    num_epochs = 20
    for epoch in range(num_epochs):
        train_loss = 0.0
        total_samples = 0
        for inputs, targets in dataloader:
            bs = targets.size(0)
            optimizer.zero_grad()  # 清空梯度
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, targets)  # 计算损失
            train_loss += loss.item() * bs
            total_samples += bs
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
        if epoch % 5 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss/total_samples}')

    # 生成一组测试输入数据
    test_input = torch.randn(10, 1)  # 生成10个随机样本，每个样本1个特征
    test_output = 2 * test_input + 1  # 期望的输出

    # 推理
    model.eval()
    with torch.no_grad():
        output = model(test_input)
        for i in range(len(test_input)):
            print(f'''Test Input: {test_input[i].item()},
            Test Output: {output[i].item()},
            Actual Output: {test_output[i].item()},
            Diff: {output[i].item() - test_output[i].item()},
            Loss: {abs(output[i].item() - test_output[i].item()) / abs(test_output[i].item())* 100:.2f}%\n''')
