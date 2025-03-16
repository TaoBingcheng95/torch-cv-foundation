from .components.base import BaseSegmentationDataset
from .mnist_datamodule import MNISTDataModule
from .tianchi_datamodule import TianchiDataModule

__all__  = [
    'MNISTDataModule',
    'TianchiDataModule'
]
