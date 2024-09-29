import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from dataset import BaseDataset


class WHDLDDataset(BaseDataset):
    """
    https://sites.google.com/view/zhouwx/dataset#h.p_hQS2jYeaFpV0
    """
    def __init__(self, root, img_folder='Images', label_folder='ImagesPNG', transform=None, **kwargs):
        super().__init__(root, **kwargs)
        self.root = root
        self.transform = transform
        self.images = glob(os.path.join(self.root, img_folder, '*.jpg'))
        self.masks = glob(os.path.join(self.root, label_folder, '*.png'))
        self.images.sort()
        self.masks.sort()
        self.classes = ['building', 'road', 'pavement', 'vegetation', 'bare soil', 'water']

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


if __name__ == '__main__':
    WHDLD_dir = 'D:\\myspace\\dataset\\segemnt\\WHDLD'

    ds = WHDLDDataset(root=WHDLD_dir, img_folder='Images', label_folder='ImagesPNG')
    ds.plot(cmap='viridis')
