import os
from glob import glob
from typing import List, Annotated, Any, Callable, Union, cast
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.utils.data import Dataset
from torchvision.transforms import transforms

# from .base import BaseSegmentationDataset
from . import BaseSegmentationDataset


class NAIPDataset(BaseSegmentationDataset):
    def __init__(self, root, img_folder='imgs',
                 label_folder='labels',
                 transform=None,
                 **kwargs):
        super().__init__(root, **kwargs)
        self.root = root
        self.transform = transform
        self.images = glob(os.path.join(self.root, img_folder, '*.tif'))
        self.masks = glob(os.path.join(self.root, label_folder, '*.tif'))
        self.images.sort()
        self.masks.sort()
        # self.num_classes = 7

    @property
    def num_classes(self) -> int:
        """
        Get the number of classes.
        class is is 1-6 in original NAIP mask

        :return: The number of NAIP classes (6).
        """
        return 6

    @property
    def dict_classes(self) -> list[str]:
        """
        Get the text for classes id.

        :return: The list of WHDLD classes (6).
        """
        return ['baresoil', 'building', 'pavement', 'road', 'vegetation', 'water']

    # def __getitem__(self, idx):
    #     image_path = self.images[idx]
    #     mask_path = self.masks[idx]
    #     image = np.asarray(Image.open(image_path))
    #     if self.band_reversal:
    #         image = image[:, :, ::-1]
    #     image = image.transpose((2, 0, 1)).astype(np.float32)
    #     mask = np.array(Image.open(mask_path)).astype(np.int64)
    #     # mask = np.expand_dims(mask, 0)
    #     return image, mask

    # def plot(self, idx=None, save=True, cmap='gray'): # viridis
    #     # assert 0<=idx < len(self), "Index out of range"
    #     if idx is None:
    #         idx = np.random.randint(0, len(self))
    #     else:
    #         if not (0 <= idx < len(self)):
    #             raise IndexError("Index out of range")

    #     img, mask = self.__getitem__(idx)
    #     img = img.astype(np.uint8)

    #     fig, axis = plt.subplots(1, 2)

    #     axis[0].set_title('image')
    #     axis[0].imshow(img.transpose((1,2,0)))
    #     axis[1].set_title('mask')
    #     axis[1].imshow(mask.squeeze())
    #     plt.tight_layout()
    #     try:
    #         if save:
    #             plt.savefig(f'dataset_{idx}.png')
    #         plt.close()
    #     finally:
    #         plt.show()
        
    #     plt.close()


class WHDLDDataset(Dataset):
    """
    https://sites.google.com/view/zhouwx/dataset#h.p_hQS2jYeaFpV0
    """
    def __init__(self, root,
                 img_folder='Images',
                 label_folder='ImagesPNG',
                 transform=None, **kwargs):
        super().__init__()
        self._num_classes = None
        self.root = root
        self.dataset_name = 'WHDLD'
        self.transform = transform
        self.images = glob(os.path.join(self.root, img_folder, '*.jpg'))
        self.masks = glob(os.path.join(self.root, label_folder, '*.png'))
        self.images.sort()
        self.masks.sort()
        self.classes = ['building', 'road', 'pavement', 'vegetation', 'bare soil', 'water']
        # self.num_classes = len(self.classes) # 1-6 in original data

    def __len__(self):
        return len(self.masks)

    @property
    def num_classes(self) -> int:
        """
        Get the number of classes.
        class is 1-6 in original WHDLD mask

        :return: The number of WHDLD classes (6).
        """
        return self.num_classes

    @property
    def dict_classes(self) -> List[str]:
        """
        Get the text for classes id.

        :return: The list of WHDLD classes (6).
        """
        return self.classes

    def __getitem__(self, idx):
        mask_arr = np.array(Image.open(self.masks[idx]))
        mask_arr = mask_arr-1 # no 0-pixle in mask
        image_arr = np.array(Image.open(self.images[idx])).transpose((2,0,1))
        # mask_arr = mask_arr.astype(np.int64)
        return image_arr, mask_arr

    def plot(self, idx=None, save=False, cmap='gray'): # viridis
        # assert 0<=idx < len(self), "Index out of range"
        if idx is None:
            idx = np.random.randint(0, len(self))
        else:
            if not (0 <= idx < len(self)):
                raise IndexError("Index out of range")

        img, mask = self.__getitem__(idx)
        fig, axis = plt.subplots(1, 2)

        # 创建图例
        cmap = plt.get_cmap('viridis', 6)
        legend_labels = [f'{i}' for i in self.classes]
        legend_handles = [mpatches.Patch(color=cmap(i), label=legend_labels[i]) for i in range(6)]

        axis[0].set_title('image')
        axis[0].imshow(img.transpose((1,2,0)))
        axis[1].set_title('mask')
        axis[1].imshow(mask.squeeze(), cmap=cmap, vmin=0, vmax=5)
        plt.legend(handles=legend_handles, title='class', loc='best')
        # plt.legend(handles=legend_handles, title='class', bbox_to_anchor=(1.0, 1.0), loc=1, borderaxespad=0.1)
        plt.tight_layout()
        try:
            if save:
                plt.savefig(f'dataset_{idx}.png')
            plt.show()
        finally:
            plt.close()

    @num_classes.setter
    def num_classes(self, value):
        self._num_classes = value


class JiageDataset(Dataset):
    def __init__(self, root, transform=None, **kwargs):
        self.root = root
        self.dataset_name = 'Jiage_Building'
        self.transform = transform
        # self.num_classes = 2
        # self.classes = ['background', 'building']
        self.masks = glob(os.path.join(self.root, 'mask','*.tif'))
        # self.images = glob(os.path.join(self.root, 'image','*.tif'))
        self.images = [i.replace('mask', 'image') for i in self.masks]

    def __len__(self):
        return len(self.masks)

    @property
    def num_classes(self) -> int:
        """
        Get the number of classes.

        :return: The number of Jiage classes (2).
        """
        return 2

    @property
    def dict_classes(self) -> list[str]:
        """
        Get the text for classes id.

        :return: The list of Jiage classes (2).
        """
        return ['background', 'building']

    def __getitem__(self, idx):
        mask_arr = np.array(Image.open(self.masks[idx]))
        image_arr = np.array(Image.open(self.images[idx])).transpose((2,0,1))
        # mask_arr = mask_arr.astype(np.int64)
        # image_arr = mask_arr.astype(np.float32)
        return image_arr, mask_arr
    
    def plot(self, idx=None, save=False, cmap='gray'):
        # assert 0<=idx < len(self), "Index out of range"
        if idx is None:
            idx = np.random.randint(0, len(self))
        else:
            if not (0 <= idx < len(self)):
                raise IndexError("Index out of range")

        img, mask = self.__getitem__(idx)
        fig, axis = plt.subplots(1, 2)

        axis[0].set_title('image')
        axis[0].imshow(img.transpose((1,2,0))) # 
        axis[1].set_title('mask')
        axis[1].imshow(mask.squeeze())
        plt.tight_layout()
        try:
            if save:
                plt.savefig(f'dataset_{idx}.png')
            # plt.show()
        finally:
            plt.close()


class TianchiDataset(Dataset):
    """
    https://sites.google.com/view/zhouwx/dataset#h.p_hQS2jYeaFpV0
    """
    def __init__(self, root, img_folder='image', label_folder='label', transform=None, **kwargs):
        super().__init__(**kwargs)
        self.root = root
        self.dataset_name = 'Tianchi_Building'
        self.band_reversal = False
        self.transform = transform
        self.masks = glob(os.path.join(self.root, label_folder, '*.jpg'))
        self.images = [i.replace('label', 'image') for i in self.masks]
        # self.images = glob(os.path.join(self.root, img_folder, '*.jpg'))
        # self.images.sort()
        # self.masks.sort()
        # self.classes = ['background', 'building']
        # self.num_classes = 2

    @property
    def num_classes(self) -> int:
        """
        Get the number of classes.

        :return: The number of Tianchi classes (2).
        """
        return 2

    @property
    def dict_classes(self) -> list[str]:
        """
        Get the text for classes id.

        :return: The list of Tianchi classes (2).
        """
        return ['background', 'building']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        mask_path = self.masks[idx]
        image = np.array(Image.open(image_path))
        if self.band_reversal:
            image = image[:, :, ::-1]
        image = image.transpose((2, 0, 1))#.astype(np.float32)
        mask = np.array(Image.open(mask_path))#.astype(np.int64)
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
        axis[1].imshow(mask.squeeze())
        plt.tight_layout()
        try:
            if save:
                plt.savefig(f'dataset_{idx}.png')
            plt.show()
        finally:
            plt.close()


class UTKFace(Dataset):
    def __init__(self, data_dir):
        self.transform = transforms.Compose([transforms.Resize((32, 32)),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406],
                                                                  [0.229, 0.224, 0.225])])
        self.image_paths = glob(os.path.join(data_dir, '*.jpg'))
        self.images = []
        self.ages = []
        self.genders = []
        self.races = []
        for fn_path in self.image_paths:
            basename = os.path.basename(fn_path)
            filename = basename.split("_")
            if len(filename)==4:
                self.images.append(fn_path)
                self.ages.append(int(filename[0]))
                self.genders.append(int(filename[1]))
                self.races.append(int(filename[2]))
    def __len__(self):
        return len(self.images)
    def __getitem__(self, index)->dict[str, Any]:
        img = Image.open(self.images[index]).convert('RGB')
        img = self.transform(img)
        age = self.ages[index]
        gender = self.genders[index]
        eth = self.races[index]
        sample = {'image':img,
                  'age': age,
                  'gender': gender,
                  'ethnicity':eth}
        return sample


if __name__ == '__main__':
    NAIP_dir = '/data/tbc/segmentation/naip'

    ds = NAIPDataset(root=NAIP_dir)
    x, y = next(iter(ds))
    print(x.shape, y.shape)
    print(len(ds))
    ds.plot(plot=True)
