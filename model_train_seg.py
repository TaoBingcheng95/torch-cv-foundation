import os
import sys
from torch.utils.data import DataLoader, random_split

from dataset.tianchi import TianchiDataset
from models.mynet.Unet import UNet
from trainers import BaseTrainer




if __name__ == '__main__':

    Tianchi_dir = 'D:\\myspace\\dataset\\segemnt\\tianchi\\train'
    val_ratio = 0.4
    test_ratio = 0.2
    num_classes = 2
    batch_size = 8
    if sys.platform.startswith('win'):
        num_workers = 0
    else:
        num_workers = 4
    ds = TianchiDataset(root=Tianchi_dir, img_folder='image', label_folder='label')
    train_ds, val_ds, test_ds = random_split(ds,
                                             [len(ds) - int(len(ds) * val_ratio) - int(len(ds) * test_ratio),
                                              int(len(ds) * val_ratio),
                                              int(len(ds) * test_ratio)])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)

    model = UNet(n_channels=3, n_classes=2)
    # optimizer = torch.optim.Adam(model.parameters(),
    #                              lr=1e-3,
    #                              weight_decay=0.0)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
    #                                            mode='min',
    #                                            patience=10,
    #                                            factor=0.1)

    # compile model for faster training with pytorch 2.0
    compile= True

    tt = BaseTrainer(model=model,
                     device='cuda:0',
                     train_dataloader=train_dl,
                     # resume='checkpoints/MNIST_SimpleDenseNet/epoch_10_valacc_0.9760.pth',
                     val_dataloader=val_dl,
                     test_dataloader=test_dl,
                     num_classes=10,
                     optimizer_type='sgd',
                     epochs=2,
                     compile=compile
                     )
    tt.fit()
    #
    # x, y = next(iter(test_dl))
    # pred = tt.predict(x)
    # print(y)
    # print(pred)
