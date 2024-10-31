import os
from abc import ABC
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


class BaseSegmentationDataset(Dataset, ABC):
    def __init__(self, root=None, img_folder='image',  label_folder='label',
                 transform=None,
                 suffix='.jpg',
                 band_reversal=False, 
                 to_numpy=False,
                 normalize=True,
                 **kwargs):
        super(BaseSegmentationDataset).__init__()
        self.root = root
        self.transform = transform
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.band_reversal = band_reversal
        self.to_numpy = to_numpy
        self.normalize = normalize
        self.masks = glob(os.path.join(self.root, label_folder, f'*{suffix}'))
        self.images = [i.replace(label_folder, img_folder) for i in self.masks]

    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):
        image_path = self.images[idx]
        mask_path = self.masks[idx]
        image = np.array(Image.open(image_path))
        if self.band_reversal:
            image = image[:, :, ::-1]
        mask = np.array(Image.open(mask_path))
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        else:
            image = image.transpose((2,0,1)) # HWC -> CHW
        if self.to_numpy:
            image = image.numpy()
            mask = mask.numpy()
        # image = image.astype(np.float32)
        # mask = mask.astype(np.int64)
        # mask = np.expand_dims(mask, 0)
        return image, mask

    def plot(self, idx=None, save=True, plot=False, cmap='gray', save_dir='./'): # viridis
        # assert 0<=idx < len(self), "Index out of range"
        if idx is None:
            idx = np.random.randint(0, len(self))
        else:
            if not (0 <= idx < len(self)):
                raise IndexError("Index out of range")
        img, mask = self.__getitem__(idx)
        
        fig, axis = plt.subplots(1, 2)
        
        axis[0].set_title('image')
        if self.to_numpy:
            img_show = img.transpose((1,2,0))
            if self.normalize:
                img_show = img_show * self.std / self.mean
        else:
            img_show = img.permute(1,2,0)
            if self.normalize:
                img_show = img_show * torch.Tensor(self.std) / torch.Tensor(self.mean)        
        axis[0].imshow(img_show)
        
        axis[1].set_title('mask')
        axis[1].imshow(mask.squeeze(), cmap=cmap)
        
        plt.tight_layout()

        if save:
            plt.savefig(os.path.join(save_dir, f'dataset_{idx}.png'))
        if plot:
            try:
                plt.show()
            except Exception as e:
                print(e)       
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

