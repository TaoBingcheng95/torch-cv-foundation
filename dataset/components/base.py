import os
from abc import ABC
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


class BaseSegmentationDataset(Dataset, ABC):
    def __init__(self, root=None, img_folder='image',  label_folder='label',
                 transform=None,
                 suffix='.jpg',
                 band_reversal=False, **kwargs):
        super().__init__()
        self.root = root
        self.transform = transform
        self.band_reversal = band_reversal
        self.masks = glob(os.path.join(self.root, label_folder, f'*{suffix}'))
        self.images = [i.replace(label_folder, img_folder) for i in self.masks]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        mask_path = self.masks[idx]
        image = np.asarray(Image.open(image_path))
        if self.band_reversal:
            image = image[:, :, ::-1]
        image = image.transpose((2, 0, 1)).astype(np.float32)
        mask = np.asarray(Image.open(mask_path)).astype(np.int64)
        # mask = np.expand_dims(mask, 0)
        return image, mask

    def plot(self, idx=None, save=False, cmap='gray'): # viridis
        # assert 0<=idx < len(self), "Index out of range"
        if idx is None:
            idx = np.random.randint(0, len(self))
        else:
            if not (0 <= idx < len(self)):
                raise IndexError("Index out of range")

        img, mask = self.__getitem__(idx)
        fig, axis = plt.subplots(1, 2)
        axis[0].set_title('image')
        axis[0].imshow(img.transpose((1,2,0)))
        axis[1].set_title('mask')
        axis[1].imshow(mask.squeeze(), cmap=cmap)
        plt.tight_layout()
        try:
            if save:
                plt.savefig(f'dataset_{idx}.png')
            plt.show()
        finally:
            plt.close()


class BaseClassificationDataset:
    def __init__(self, data_dir, transform, batch_size):
        self.data_dir = data_dir
        self.transform = transform
        self.batch_size = batch_size

    def get_dataloader(self):
        dataset = ImageFolder(root=self.data_dir,
                                       transform=self.transform, )
        # dataloader = DataLoader(dataset,
        #                           batch_size=self.batch_size,
        #                           shuffle=True)
        return dataset

