from dataset.torch_dataset import CIFAR10DataLoader, MNISTDataLoader
from torchvision.datasets import CIFAR10, FashionMNIST, MNIST
from  torchvision import transforms
from dataset import VOC2012ClassSegLoader


if __name__ == "__main__":
    # ds = CIFAR10DataLoader()
    # # print(ds.classes)
    # # print(ds.class_to_idx)
    # # print(ds.idx_to_class)
    # test_loader = ds.test_dataloader()
    # ds.plot_sample()

    dm = VOC2012ClassSegLoader(root='data', batch_size=8, val_split=0.1, img_size=320)
    train_loader = dm.train_dataloader()
    x, y = next(iter(train_loader))
    print(x.shape, y.shape)
    dm.plot_sample()

    # transform = transforms.Compose([
    #         # transforms.Resize(32),  # LeNet-5 经典输入是 32x32，MNIST 原是 28x28
    #         transforms.ToTensor()
    #     ])
    # mm = CIFAR10(root='./data',transform=transform, train=False)
    # print((len(mm))) # 50000
