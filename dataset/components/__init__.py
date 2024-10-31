from .base import BaseSegmentationDataset
from .torch_dataset import MNISTLoader, FashionMNISTLoader, CIFAR10Loader
from .my_dl import NAIPDataset, JiageDataset, TianchiDataset, WHDLDDataset

all = ['MNISTLoader', 
       'FashionMNISTLoader', 
       'CIFAR10Loader', 
       'NAIPDataset',
       'JiageDataset',
       'TianchiDataset',
       'WHDLDDataset'
       ]
