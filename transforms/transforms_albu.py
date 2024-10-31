import albumentations as A
from albumentations.pytorch import ToTensorV2


train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.GaussianBlur(p=0.1),
    A.Rotate(limit=90, p=0.5),
    # A.Resize(height=256, width=256, p=1.0),
    A.Normalize(),
    ToTensorV2()])

val_transform = A.Compose([
    # A.Resize(height=512, width=512, p=1.0),
    A.Normalize(),
    ToTensorV2()])


test_transform = A.Compose([
    # A.Resize(height=512, width=512, p=1.0),
    A.Normalize(),
    ToTensorV2()])
