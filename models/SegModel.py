import os
import numpy as np
import segmentation_models_pytorch as smp
# from holocron import optim as hoptim
import torch.optim as optim
import lightning.pytorch as pl
from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader
from loss.loss_func import DiceLoss, CE_DiceLoss
from metrics import SegmentationMetric
# from . import losses, metrics
# from . import config
# from . import SegDataSet

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def cacl_score2(y_pr, y_gt, classes):
    y_pr = torch.argmax(y_pr, dim=1).cpu().numpy().astype(np.uint8)
    y_gt = y_gt.cpu().numpy().astype(np.uint8)
    # print("classes_set length : ", len(config.classes_set))
    e = SegmentationMetric(classes, y_pr, y_gt)
    mAP = e.meanPixelAccuracy()
    miou = e.meanIntersectionOverUnion()
    # fwiou = e.Frequency_Weighted_Intersection_over_Union()
    return torch.tensor(mAP), torch.tensor(miou)

class SegModel(pl.LightningModule):

    def __init__(self, classes=2,lr=1e-3, weight_decay=1e-5):
        super().__init__()
        self.net = smp.DeepLabV3Plus(encoder_name="resnet101",
                                     encoder_weights="imagenet",
                                     classes=classes)
        self.classes = classes
        self.lr = lr
        self.weight_decay = weight_decay
        # self.hparams = hparams

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        # optimizer = hoptim.RAdam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        # optimizer = hoptim.wrapper.Lookahead(optimizer)
        optimizer = optim.Adam(self.parameters(),
                               lr=self.lr,
                               weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)
        return [optimizer], [scheduler]


    def training_step(self, batch, batch_nb):
        img, mask = batch
        out = self(img)
        loss_val = CE_DiceLoss()(out, mask)
        mAP, fwiou = cacl_score2(self.classes, out, mask)
        log_dict = {'train_loss': loss_val, "train_mAP": mAP, "train_miou": fwiou}
        return {'loss': loss_val, 'log': log_dict, 'progress_bar': log_dict}


    def validation_step(self, batch, batch_idx):
        img, mask = batch
        out = self(img)
        loss_val = CE_DiceLoss()(out, mask)
        mAP, fwiou = cacl_score2(self.classes, out, mask)
        return {'val_loss': loss_val, "val_mAP": mAP, "val_miou": fwiou}

    def validation_epoch_end(self, outputs):
        tqdm_dict = {}
        for metric_name in ["val_loss", "val_mAP", "val_miou"]:
            metric_total = 0
            for output in outputs:
                metric_value = output[metric_name]
                metric_total += metric_value
            tqdm_dict[metric_name] = metric_total / len(outputs)
        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'val_loss': tqdm_dict["val_loss"]}
        return result
