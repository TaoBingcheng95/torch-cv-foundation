from dataset.torch_dataset import CIFAR10DataLoader, MNISTDataLoader


if __name__ == "__main__":
    ds = MNISTDataLoader()
    print(ds.classes)
    print(ds.class_to_idx)
    print(ds.idx_to_class)
    ds.plot_sample()
