
# from .components import MNISTDataLoader
# from .mnist_datamodule import MNISTDataModule
# from .tianchi_datamodule import TianchiDataModule

from .torch_dataset import MNISTDataLoader, FashionMNISTDataLoader, CIFAR10DataLoader

__all__  = [
    'MNISTDataLoader',  
    'FashionMNISTDataLoader',
    'CIFAR10DataLoader',
]
