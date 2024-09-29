import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, FashionMNIST, Cityscapes
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader, random_split



class FashionMNISTLoader:
    def __init__(self, root='./data', train=True, transform=None, target_transform=None, download=False, val_split = 0.4):
        super(FashionMNISTLoader, self).__init__()

        # 定义数据的预处理变换(归一化)
        transform = transforms.Compose([
            # transforms.Resize(size=224),    #图像大小设为224
            transforms.ToTensor(),  # 将图像转换为Tensor格式
            # transforms.Normalize((0.5,), (0.5,))  # 对图像进行标准化，使像素值在[-1, 1]之间
        ])

        self.train_ds = FashionMNIST(
            root=root,  # 指定数据集的存储路径
            train=True,  # 加载训练集
            download=download,  # 如果数据集不存在，自动下载
            transform=transform  # 应用预处理操作
        )

        self.test_ds = FashionMNIST(
            root=root,  # 指定数据集的存储路径
            train=False,  # 加载测试集
            download=download,  # 如果数据集不存在，自动下载
            transform=transform  # 应用预处理操作
        )
        self.classes = self.train_ds.classes
        self.class_to_idx = self.train_ds.class_to_idx
        self.idx_to_class = {value: key for key, value in self.class_to_idx.items()}
        if val_split:
            train_count = len(self.train_ds)
            self.train_ds, self.val_ds = random_split(self.train_ds,
                                                      [train_count - int(train_count * val_split), int(train_count * val_split)])
        else:
            self.val_ds = self.test_ds

    def loader(self, batch_size=32):

        self.train_loader = DataLoader(
            dataset=self.train_ds,  # 传入训练数据集
            batch_size=batch_size,  # 每个批次的样本数量
            shuffle=True  # 每个epoch打乱数据
        )

        self.val_loader = DataLoader(
            dataset=self.val_ds,
            batch_size=batch_size,
            shuffle=False
        )

        self.test_loader = DataLoader(
            dataset=self.test_ds,  # 传入测试数据集
            batch_size=batch_size,  # 每个批次的样本数量
            shuffle=False  # 测试集不需要打乱
        )

    def plot_sample(self):
        x, y = self.val_ds[0]
        fig, axis = plt.subplots()
        axis.imshow(x.permute(1, 2, 0), cmap='viridis')
        axis.set_title(self.idx_to_class[y])
        plt.show()




class CIFAR10Loader:
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, val_split = 0.4):
        super(CIFAR10Loader, self).__init__()
        train_transform = transforms.Compose([# transforms.RandomResizedCrop(224),
                                              # transforms.RandomCrop(32, padding=4),
                                              # transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
        val_transform = transforms.Compose([#transforms.Resize(256),
                                           #transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                            ]
                                           )

        self.train_ds = CIFAR10(root=root, train=True, transform=val_transform, download=download)
        self.test_ds = CIFAR10(root=root, train=False, transform=val_transform, download=download)
        self.class_to_idx = self.train_ds.class_to_idx
        self.idx_to_class = {value: key for key, value in self.class_to_idx.items()}

        # val_split = 0.4
        if val_split:
            train_count = len(self.train_ds)
            self.train_ds, self.val_ds = random_split(self.train_ds,
                                                      [train_count - int(train_count * val_split), int(train_count * val_split)])

    def plot_sample(self):
        x, y = self.val_ds[0]
        fig, axis = plt.subplots()
        axis.imshow(x.permute(1, 2, 0))
        axis.set_title(self.idx_to_class[y])
        plt.show()

if __name__ == '__main__':
    dataset = CIFAR10Loader(root='./data', train=False)
    dataset.plot_sample()

