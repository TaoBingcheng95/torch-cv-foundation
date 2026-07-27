from dataset.torch_dataset import CIFAR10DataLoader, MNISTDataLoader
from torchvision.datasets import CIFAR10, FashionMNIST, MNIST
from  torchvision import transforms


if __name__ == "__main__":
    # ds = CIFAR10DataLoader()
    # # print(ds.classes)
    # # print(ds.class_to_idx)
    # # print(ds.idx_to_class)
    # ds.plot_sample()

    transform = transforms.Compose([
            # transforms.Resize(32),  # LeNet-5 经典输入是 32x32，MNIST 原是 28x28
            transforms.ToTensor()
        ])
    mm = CIFAR10(root='./data',transform=transform)
    x, y = mm[0]
    print(x.shape)
