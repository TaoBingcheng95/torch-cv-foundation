import numpy as np
import matplotlib.pyplot as plt
import torch

from dataset.components import NAIPDataset
from dataset import TianchiDataModule
from transforms import transforms as T
from transforms.transforms_albu import train_transform


def demo_ds():
    NAIP_dir = '/data/tbc/segmentation/naip'
    ds = NAIPDataset(root=NAIP_dir)
    x, y = next(iter(ds))
    print(x.shape, y.shape)
    print(len(ds))
    ds.plot(plot=True)


def demo_dm():
    data_module = TianchiDataModule(root='/data/tbc/segmentation/tianchi', batch_size=8)
    data_module.prepare_data()  # 准备数据，如果需要下载的话
    data_module.setup()  # 设置数据，为训练做准备
    # train_loader = data_module.train_dataloader()
    # for i, (x, y) in enumerate(train_loader):
    #     print(x.shape, y.shape)
    #     if i > 5:
    #         break
    data_module.plot(plot=True)


def check_transform():
    NAIP_dir = '/data/tbc/segmentation/naip'
    
    ds = NAIPDataset(root=NAIP_dir)
    x, y = next(iter(ds))
    print(x.shape, y.shape)

    ds_anth = NAIPDataset(root=NAIP_dir) # , transform=train_transform
    ds_anth.transform = train_transform
    x_a, y_a = next(iter(ds_anth))
    print(x_a.shape, y_a.shape)

    fig, axis = plt.subplots(2,2)
    axis[0,0].imshow(x.transpose((1,2,0)))
    axis[0,1].imshow(y)

    tmp = x_a.permute((1,2,0))
    mean = torch.Tensor((0.485, 0.456, 0.406))
    std = torch.Tensor((0.229, 0.224, 0.225))
    # tmp = tmp * std + mean
    axis[1,0].imshow(tmp)
    axis[1,1].imshow(y_a)
    
    plt.savefig('output/test.png')



if __name__ == '__main__':
    demo_dm()
