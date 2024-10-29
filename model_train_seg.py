import sys
import numpy as np
from torch.utils.data import DataLoader, random_split
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

from dataset.my_dl import TianchiDataset, NAIPDataset
from models.components import SimpleUNet, UNet
from trainers import BaseTrainer
from transforms.transforms import train_transform



if __name__ == '__main__':

    Tianchi_dir = "/data/tbc/segmentation/tianchi" # 'D:\\myspace\\dataset\\segemnt\\tianchi'
    # WHDLD_dir = '/data/tbc/seg/WHDLD'
    # data_dir = '/data/tbc/segmentation/naip'

    val_ratio = 0.4
    test_ratio = 0.2
    num_classes = 2 # naip=7 whdld=6+1
    batch_size = 4
    if sys.platform.startswith('win'):
        num_workers = 0
    else:
        num_workers = 0 # 4
    ds = TianchiDataset(root=Tianchi_dir)
    data_count = len(ds)
    val_count = int(data_count * val_ratio)
    test_count = int(data_count * test_ratio)

    train_ds, val_ds, test_ds = random_split(ds, [data_count - val_count - test_count, val_count, test_count])
    # train_ds.dataset.transform = train_transform
    train_dl = DataLoader(train_ds,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=num_workers,
                          drop_last=True,
                          pin_memory=True)
    val_dl = DataLoader(val_ds,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_workers,
                        drop_last=True,
                        pin_memory=True)
    test_dl = DataLoader(test_ds,
                         batch_size=batch_size,
                         shuffle=False,
                         num_workers=num_workers,
                         drop_last=True,
                         pin_memory=True)

    # model = smp.Unet('resnet34', classes=num_classes, activation='softmax')

    model = UNet(in_channels=3, out_channels=num_classes, use_attention=False)

    # optimizer = torch.optim.Adam(model.parameters(),
    #                              lr=1e-3,
    #                              weight_decay=0.0)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
    #                                            mode='min',
    #                                            patience=10,
    #                                            factor=0.1)

    tt = BaseTrainer(model=model,
                     device='cuda:0',
                     train_dataloader=train_dl,
                     val_dataloader=val_dl,
                     test_dataloader=test_dl,
                     # resume='output/20241028_144830/epoch_60_valacc_0.4985.pth',
                     num_classes=num_classes,
                     optimizer_type='sgd',
                     epochs=100,
                     compile=False # compile model for faster training with pytorch 2.0
                     )
    tt.fit()

    x, y = next(iter(test_dl))
    pred = tt.predict(x)
    # print(x.shape, y.shape)
    # print(pred.shape)
    # print(np.unique(y[0,:,:]))
    # print(np.unique(pred.cpu().numpy()[0,:,:]))

    fig, axis = plt.subplots(1,3)
    axis[0].imshow(x[0,:,:,:].permute(1,2,0).numpy().astype(np.uint8))
    axis[1].imshow(y[0,:,:])
    axis[2].imshow(pred.cpu().numpy()[0,:,:])
    plt.savefig('test.png')
