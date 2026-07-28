import os

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from lightning.pytorch import LightningDataModule


class FlowerDataModule(LightningDataModule):
    def __init__(self, data_dir: str,
                 batch_size: int = 32,
                 image_size: int = 224,
                 num_work: int = 0):
        super().__init__()
        self.test_dataset = None
        self.val_dataset = None
        self.train_dataset = None
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_work
        self.transform = self._get_transforms()

    def _get_transforms(self):
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def prepare_data(self):
        # Download data if needed (not used in this example)
        pass

    def setup(self, stage: str = None):
        # Assign train/val/test datasets
        if stage == 'fit' or stage is None:
            # Assuming a separate directory for train and validation
            train_dir = os.path.join(self.data_dir, 'train')

            train_transforms = transforms.Compose([transforms.RandomResizedCrop(self.image_size),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize((0.5, 0.5, 0.5),
                                                                        (0.5, 0.5, 0.5))])
            self.train_dataset = ImageFolder(root=train_dir, transform=train_transforms)
            val_dir = os.path.join(self.data_dir, 'val')
            self.val_dataset = ImageFolder(root=val_dir, transform=self.transform)

        if stage == 'test' or stage is None:
            test_dir = os.path.join(self.data_dir, 'test')
            self.test_dataset = ImageFolder(root=test_dir, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          # num_workers=self.num_workers,
                          drop_last=True
                          )

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          # num_workers=self.num_workers,
                          drop_last=False
                          )

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          # persistent_workers=True,
                          # num_workers=self.num_workers,
                          drop_last=False
                          )


if __name__ == '__main__':
    # 使用方法
    data_module = FlowerDataModule(data_dir='../../data/flower_data', batch_size=64, image_size=64)
    data_module.prepare_data()  # 准备数据，如果需要下载的话
    data_module.setup('fit')  # 设置数据，为训练做准备

    train_loader = data_module.train_dataloader()
    images, labels = next(iter(train_loader))
    print(images.shape, labels.shape)
