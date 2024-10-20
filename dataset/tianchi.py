import os
from glob import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from dataset import BaseDataset


class TianchiDataset(BaseDataset):
    """
    https://sites.google.com/view/zhouwx/dataset#h.p_hQS2jYeaFpV0
    """
    def __init__(self, root, img_folder='image', label_folder='label', transform=None, **kwargs):
        super().__init__(root, **kwargs)
        self.root = root
        self.transform = transform
        self.images = glob(os.path.join(self.root, img_folder, '*.jpg'))
        self.masks = glob(os.path.join(self.root, label_folder, '*.jpg'))
        self.images.sort()
        self.masks.sort()
        self.classes = ['background', 'building']


    def __getitem__(self, idx):
        image_path = self.images[idx]
        mask_path = self.masks[idx]
        image = np.asarray(Image.open(image_path))
        if self.band_reversal:
            image = image[:, :, ::-1]
        image = image.transpose((2, 0, 1)).astype(np.float32)
        mask = np.array(Image.open(mask_path))
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


if __name__ == '__main__':
    Tianchi_dir = 'D:\\myspace\\dataset\\segemnt\\tianchi\\train'

    ds = TianchiDataset(root=Tianchi_dir, img_folder='image', label_folder='label')
    ds.plot()
