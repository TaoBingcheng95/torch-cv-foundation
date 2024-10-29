import sys
from pprint import pprint
import numpy as np
from torch.utils.data import DataLoader, random_split
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

from dataset.my_dl import TianchiDataset, NAIPDataset
from models.components import SimpleUNet, UNet
from trainers import BaseTrainer
from transforms.transforms import train_transform
from metrics import Metrics


if __name__ == '__main__':

    Tianchi_dir = 'D:\\myspace\\dataset\\segemnt\\tianchi' # "/data/tbc/segmentation/tianchi" #
    # WHDLD_dir = '/data/tbc/seg/WHDLD'
    # data_dir = '/data/tbc/segmentation/naip'

    val_ratio = 0.4
    test_ratio = 0.2
    num_classes = 2 # naip=7 whdld=6+1
    batch_size = 8
    if sys.platform.startswith('win'):
        num_workers = 0
    else:
        num_workers = 4
    ds = TianchiDataset(root=Tianchi_dir)
    data_count = len(ds)
    val_count = int(data_count * val_ratio)
    test_count = int(data_count * test_ratio)

    train_ds, val_ds, test_ds = random_split(ds, [data_count - val_count - test_count, val_count, test_count])
    # train_ds.dataset.transform = train_transform
    train_dl = DataLoader(train_ds,
                          batch_size=batch_size,
                          shuffle=False,
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
    model = smp.Unet(
        encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=2,  # model output channels (number of classes in your dataset)
    )
    # model = UNet(in_channels=3, out_channels=num_classes, use_attention=False)

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
                     resume='output/20240920_094528/epoch_19_acc_0.9623_miou_0.9274.pth',
                     num_classes=num_classes,
                     optimizer_type='sgd',
                     epochs=1,
                     compile=False # compile model for faster training with pytorch 2.0
                     )
    # tt.fit()


    x, y = next(iter(test_dl))
    preds = tt.predict(x)
    print(x.shape, y.shape)
    print(preds.shape)
    # print(np.unique(y[0,:,:]))
    # print(np.unique(pred.cpu().numpy()[0,:,:]))

    metrics = Metrics(2,'cuda:0')
    # 初始化混淆矩阵
    cnf_matrix = np.zeros((2, 2))
    metrics.sample_add(y.to('cuda:0'), preds.to('cuda:0'))
    results = metrics.compute()
    pprint(results)

    # fig, axis = plt.subplots(1,3)
    # axis[0].imshow(x[0,:,:,:].permute(1,2,0).numpy().astype(np.uint8))
    # axis[1].imshow(y[0,:,:])
    # axis[2].imshow(pred.cpu().numpy()[0,:,:])
    # plt.savefig('output/test.png')
