import os
import sys
from glob import glob
from torch.utils.data import DataLoader, random_split

from dataset.tianchi import TianchiDataset
from dataset.whdld import WHDLDDataset
from models.mynet.Unet import UNet
from trainers import BaseTrainer

if __name__ == '__main__':

    Tianchi_dir = 'D:\\myspace\\dataset\\segemnt\\tianchi'
    # WHDLD_dir = '/data/tbc/seg/WHDLD'

    # img_list = glob(os.path.join(Tianchi_dir, 'image', '*.jpg'))
    # label_list = glob(os.path.join(Tianchi_dir, 'label', '*.jpg'))
    # print(len(img_list), len(label_list))

    val_ratio = 0.4
    test_ratio = 0.2
    num_classes = 2
    batch_size = 8
    if sys.platform.startswith('win'):
        num_workers = 0
    else:
        num_workers = 4
    ds = TianchiDataset(root=Tianchi_dir, img_folder='image', label_folder='label')
    # ds = WHDLDDataset(root='/data/tbc/seg/WHDLD')

    train_ds, val_ds, test_ds = random_split(ds,
                                             [len(ds) - int(len(ds) * val_ratio) - int(len(ds) * test_ratio),
                                              int(len(ds) * val_ratio),
                                              int(len(ds) * test_ratio)])
    train_dl = DataLoader(train_ds,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0,
                          drop_last=True,
                          pin_memory=True)
    val_dl = DataLoader(val_ds,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=0,
                        drop_last=True,
                        pin_memory=True)
    test_dl = DataLoader(test_ds,
                         batch_size=batch_size,
                         shuffle=False,
                         num_workers=0,
                         drop_last=True,
                         pin_memory=True)

    # print(len(train_ds))
    # print(len(val_ds))
    # print(len(test_ds))
    # print(len(train_dl))
    # print(len(train_dl.dataset))

    model = UNet(n_channels=3, n_classes=2)
    # optimizer = torch.optim.Adam(model.parameters(),
    #                              lr=1e-3,
    #                              weight_decay=0.0)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
    #                                            mode='min',
    #                                            patience=10,
    #                                            factor=0.1)

    # compile model for faster training with pytorch 2.0
    compile = False

    tt = BaseTrainer(model=model,
                     device='cuda:0',
                     train_dataloader=train_dl,
                     val_dataloader=val_dl,
                     test_dataloader=test_dl,
                     resume='output/20241023_171706/epoch_1_valacc_1.3577.pth',
                     num_classes=num_classes,
                     optimizer_type='sgd',
                     epochs=2,
                     compile=compile
                     )
    # tt.fit()

    x, y = next(iter(test_dl))
    pred = tt.predict(x)
    print(pred.shape)
    # print(y)
    # print(pred)
