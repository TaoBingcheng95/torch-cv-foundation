
from .components import MNISTDataLoader


from .mnist_datamodule import MNISTDataModule
# from .tianchi_datamodule import TianchiDataModule

__all__  = [
    'MNISTDataLoader',  
    'MNISTDataModule',
    # 'TianchiDataModule'
]
