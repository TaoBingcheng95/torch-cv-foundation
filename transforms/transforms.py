import albumentations as A
from albumentations.pytorch import ToTensorV2


train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.GaussianBlur(p=0.1),
    A.Resize(height=512, width=512, p=1.0),
    A.Normalize(),
    ToTensorV2()])


train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.GaussianBlur(p=0.1),
    A.Resize(height=512, width=512, p=1.0),
    A.Normalize(),
    ToTensorV2()])

val_transform = A.Compose([
    A.Resize(height=512, width=512, p=1.0),
    A.Normalize(),
    ToTensorV2()])


test_transform = A.Compose([
    A.Resize(height=512, width=512, p=1.0),
    A.Normalize(),
    ToTensorV2()])


# transform = {
#         'train':A.Compose([
#             A.HorizontalFlip(p=0.5),
#             A.RandomBrightnessContrast(p=0.2),
#             A.GaussianBlur(p=0.1),
#             A.Resize(height=512, width=512, p=1.0),
#             A.Normalize(),
#             ToTensorV2()
#         ]),
#         'val':A.Compose([
#             A.Resize(height=512, width=512),
#             A.Normalize(),
#             ToTensorV2()
#         ]),
#         'test': A.Compose([
#             A.Resize(height=512, width=512),
#             A.Normalize(),
#             ToTensorV2()
#         ])
# }
