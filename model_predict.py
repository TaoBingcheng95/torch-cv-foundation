import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FashionMNIST, MNIST
from models.mynet.LeNet import LeNet

transform = transforms.Compose([
    # transforms.Resize(size=224),  # 将输入的28x28图像调整为224x224（用于较大的CNN模型）
    transforms.ToTensor(),  # 将图像转换为PyTorch的Tensor格式
    transforms.Normalize((0.5,), (0.5,))  # 可以选择添加标准化，像素值范围[-1, 1]
])

# 定义处理测试数据的函数
def val_data_process():

    test_data = MNIST(
        root='./data',  # 指定数据集存储路径，如果不存在会自动下载并创建
        train=False,  # 设置为False表示加载测试集
        download=True,  # 如果本地没有数据集，则自动从网上下载
        transform=transform  # 使用预处理操作，将图像大小调整并转换为Tensor格式
    )
    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=1,
        shuffle=True,
        num_workers=0  # Windows系统下设置为0，Linux/Mac环境可以设置更大的值以加速数据加载
    )
    return test_dataloader  # 返回DataLoader对象以便模型测试时使用

def model_process(model, test_dataloader):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()  # 设置模型为评估模式(前向传播)

    test_corrects = 0.0 # 初始化正确预测的计数
    test_num = 0    # 初始化样本数量

    with torch.no_grad():
        for test_data_x, test_data_y in test_dataloader:  #因为每次只有一个样本 所以可以直接在test_dataloader中获得
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)
            output = model(test_data_x)
            pre_lab = torch.argmax(output, dim=1) #argmax包含sigmax和返回最大下标
            test_corrects += torch.sum(pre_lab == test_data_y) #得出的结果和标记比较
            test_num += test_data_x.size(0) # 更新样本数量

    # 计算测试准确率
    test_acc = test_corrects.item() / test_num
    print(f"测试的准确率为：{test_acc}")


if __name__=="__main__":
    model = LeNet() # 实例化模型
    model.load_state_dict(torch.load('checkpoints/lenet_mnist_model.pth',weights_only=True)) # 加载保存的最佳模型权重
    test_dl = val_data_process()
    model_process(model, test_dl)
