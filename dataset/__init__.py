
# from .components import MNISTDataLoader
# from .mnist_datamodule import MNISTDataModule
# from .tianchi_datamodule import TianchiDataModule

from .torch_dataset import MNISTDataLoader, FashionMNISTDataLoader, CIFAR10DataLoader
from .utils import auto_pin_memory, get_smart_num_workers

__all__  = [
    'MNISTDataLoader',  
    'FashionMNISTDataLoader',
    'CIFAR10DataLoader',
    'auto_pin_memory',
    'get_smart_num_workers'
]
