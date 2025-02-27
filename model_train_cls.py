# https://mp.weixin.qq.com/s/ZsKwD-Cb1ynqvCdBIWlZgw
import os
from dataset.mnist_datamodule import MNISTDataModule
from models.components import SimpleDenseNet
# from metrics import Metrics
from trainers import BaseTrainer


if __name__ == '__main__':

    dm = MNISTDataModule(data_dir='./data',
                         batch_size=8,
                         # num_worker=4 * num_GPU,
                         pin_memory=True)
    dm.prepare_data()
    dm.setup()
    train_dl = dm.train_dataloader()
    val_dl = dm.val_dataloader()
    test_dl = dm.test_dataloader()
    num_calsses = dm.num_classes
    
    # MNIST_SimpleDenseNet
    model = SimpleDenseNet(output_size=dm.num_classes)

    # optimizer = torch.optim.Adam(model.parameters(),
    #                              lr=1e-3,
    #                              weight_decay=0.0)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
    #                                            mode='min',
    #                                            patience=10,
    #                                            factor=0.1)
    # compile model for faster training with pytorch 2.0
    compile= False
    resume='checkpoints/MNIST_SimpleDenseNet/epoch_10_valacc_0.9760.pth'
    tt = BaseTrainer(model=model,
                     device='cuda:0',
                     resume=resume,
                     train_dataloader=train_dl,
                     val_dataloader=val_dl,
                     test_dataloader=test_dl,
                     optimizer_type='sgd',
                     epochs=10,
                     compile=compile,
                     num_classes=num_calsses
                     )
    print(tt.save_dir)
    # tt.fit()

    x, y = next(iter(test_dl))
    pred = tt.predict(x)
    print(y)
    print(pred)
