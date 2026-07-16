# https://mp.weixin.qq.com/s/uMxruIyMIpY1BpaaZdLKjA
# https://mp.weixin.qq.com/s/vMTXMrvJbjQuYVhruVoTcg

import os
import random
from pathlib import Path
import torch
import torchvision
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from models import LeNetLitModule
from dataset import MNISTDataModule

torch.set_float32_matmul_precision('medium') # 'medium' | 'high'


_EARLY_STOPPING_PATIENCE = 50 # epochs


def save_results(
    img_tensors: list[torch.Tensor], output_tensors: list[torch.Tensor], out_dir: Path, max_number_of_imgs: int = 10
):
    """
    Save test results as images in the provided output directory.
    Args:
        img_tensors: List of the tensors containing the input images.
        output_tensors: List of softmax activation from the trained model.
        out_dir: Path to output directory.
        max_number_of_imgs: Maximum number of images to output from the provided images. The images will be selected randomly.
    """
    selected_img_indices = random.sample(range(len(img_tensors)), min(max_number_of_imgs, len(img_tensors)))
    for img_indice in selected_img_indices:
        # Take the first instance of the batch (index 0)
        img_filepath = out_dir / f"{img_indice}_predicted_{torch.argmax(output_tensors[img_indice], dim=1)[0]}.png"
        torchvision.utils.save_image(img_tensors[img_indice][0], fp=img_filepath)


if __name__ == "__main__":

    out_dir = None
    early_stopping = True
    max_epoch = 5

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    if out_dir is None:
        out_dir = Path(__file__).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    data_module = MNISTDataModule(data_dir='./data/classification',
                                  resize_32=True)

    # Select architecture
    model = LeNetLitModule(in_channels=1, out_channels=10)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        save_last=True,
        filename='best-{epoch:02d}_{val_acc:03f}',
        auto_insert_metric_name = False,
    )
    early_stopping = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=_EARLY_STOPPING_PATIENCE,
        verbose=True,
        mode="min",
    )
    model_summary = pl.callbacks.RichModelSummary(max_depth=-1)
    rich_progress_bar = pl.callbacks.RichProgressBar()

    callbacks = (
        [
            early_stopping,
            checkpoint_callback,
            # model_summary,
            # rich_progress_bar
        ]
    )

    # If your machine has GPUs, it will use the GPU Accelerator for training.
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices='auto',
        strategy="auto",
        max_epochs=max_epoch,
        callbacks=callbacks,
        default_root_dir=out_dir,
    )

    # Train the model ⚡
    # data_module.setup(stage="fit")  # Is called by trainer.fit().
    # Call training_step + validation_step for all the epochs.
    trainer.fit(model, datamodule=data_module)
    # # Validate
    # trainer.validate(datamodule=data_module, ckpt_path='best')
    #
    # # Automatically auto-loads the best weights from the previous run.
    # # data_module.setup(stage="test")  # Is called by trainer.test().
    # # The checkpoint path is logged on the terminal.
    # trainer.test(datamodule=data_module,ckpt_path='best')

    # Run the prediction on the test set and save a subset of the resulting prediction along with the
    # original image.

    # output_preds = trainer.predict(datamodule=data_module, ckpt_path="best")
    # img_tensors, softmax_preds = zip(*output_preds)
    # out_dir_imgs = out_dir / "test_images"
    # out_dir_imgs.mkdir(exist_ok=True, parents=True)
    # save_results(
    #     img_tensors=img_tensors,
    #     output_tensors=softmax_preds,
    #     out_dir=out_dir_imgs,
    # )

