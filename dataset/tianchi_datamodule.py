from typing import Any, Dict, Optional, Tuple
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
# from torchvision.transforms import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

from .components import TianchiDataset



class TianchiDataModule(LightningDataModule):
    """
    `LightningDataModule` for the MNIST dataset.

    The MNIST database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples.
    It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a
    fixed-size image. The original black and white images from NIST were size normalized to fit in a 20x20 pixel box
    while preserving their aspect ratio. The resulting images contain grey levels as a result of the anti-aliasing
    technique used by the normalization algorithm. the images were centered in a 28x28 image by computing the center of
    mass of the pixels, and translating the image to position this point at the center of the 28x28 field.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        root: str = "data/",
        train_val_test_split: Tuple[float, float, float] = (0.4,0.4,0.2),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """
        Initialize a `TianchiDataModule`.

        :param root: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.num_workers = None
        self.save_hyperparameters(logger=False)

        # data transformations
        # self.transforms = transforms.Compose(
        #     [
        #      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #      transforms.ToTensor()
        #      ]
        #      )
        self.transforms = A.Compose([
            A.HorizontalFlip(p=0.5),  # 随机水平翻转
            A.RandomCrop(width=256, height=256),  # 随机裁剪
            A.RandomBrightnessContrast(p=0.2),  # 随机亮度对比度
            A.HueSaturationValue(p=0.2),  # 随机色调饱和度
            A.Rotate(limit=35, p=0.5),  # 随机旋转
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),  # 高斯噪声
            # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # 归一化
            ToTensorV2()  # 转换为PyTorch张量
            ]) # , additional_targets={'mask': 'mask'}

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size
        # self.init_num_workers()

    @property
    def num_classes(self) -> int:
        """
        Get the number of classes.

        :return: The number of Tianchi classes (2).
        """
        return 2

    @property
    def dict_classes(self) -> int:
        """
        Get the text for classes id.

        :return: The list of Tianchi classes (2).
        """
        return ['background', 'building']

    
    def init_num_workers(self) -> int:
        if sys.platform.startswith('win'):
            self.num_workers = 0
        else:
            pass
            # num_workers = 4


    def prepare_data(self) -> None:
        """
        Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        TianchiDataset(self.hparams.root)

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            dataset = TianchiDataset(self.hparams.root)
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )
            self.data_train.transforms = self.transforms

    def train_dataloader(self) -> DataLoader[Any]:
        """
        Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=True,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """
        Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=True,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """
        Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=True,
            shuffle=False,
        )

    def predict_dataloader(self) -> DataLoader:
        """Called by Trainer `predict()` method. Use the same data as the test_dataloader."""
        return DataLoader(self.data_test,
                          batch_size=self.batch_size_per_device,
                          num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory,
                          drop_last=True,
                          shuffle=False,
                          )

    def teardown(self, stage: Optional[str] = None) -> None:
        """
        Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """
        Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass
    
    def plot(self, save=True, plot=False, cmap='gray', save_dir='./'):
        
        x, y = next(iter(self.train_dataloader()))
        B,C,H,W = x.shape
        if B>4:
            B = 4
            print('Too many samples, only plot 4 samples.')
        
        fig, axis = plt.subplots(2, B)
        for idx in range(B):
            axis[0][idx].imshow(x[idx].permute(1, 2, 0))
            axis[1][idx].imshow(y[idx], cmap=cmap)
            axis[0][idx].axis('off')
            axis[1][idx].axis('off')
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(save_dir,'sample.png'))
        if plot:
            try:
                plt.show()
            except:
                pass
        plt.close()


if __name__ == "__main__":
    _ = TianchiDataset()