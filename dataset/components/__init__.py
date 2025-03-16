from .base import BaseSegmentationDataset
from .torch_dataset import MNISTLoader, FashionMNISTLoader, CIFAR10Loader
from .my_ds import NAIPDataset, JiageDataset, TianchiDataset, WHDLDDataset

__all__  = ['MNISTLoader',
            'FashionMNISTLoader',
            'CIFAR10Loader',
            'NAIPDataset',
            'JiageDataset',
            'TianchiDataset',
            'WHDLDDataset'
            ]
