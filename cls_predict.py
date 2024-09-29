from PIL import features
from torch.utils.data import DataLoader

import torch
import torchvision
from torchvision import datasets
from torch.utils.data import random_split

from torchvision.transforms import transforms

import matplotlib.pyplot as plt
from models.mynet import AlexNet

# 对图像进行尺寸变换，因为网络要求的输入是64*64，并且是tensor类型
custom_transform = transforms.Compose([transforms.Resize([224, 224]),
                                       transforms.ToTensor()])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = torchvision.models.vgg16().to(device)
model = AlexNet(in_channels=3, num_classes=4).to(device)

# map_location:指定设备，cpu或者GPU
model.load_state_dict(torch.load(f="checkpoints/satellite-image-classification/train_model_4_72.38.pth",
                                 weights_only=True,
                                 map_location="cpu"))

# 使用自己的数据集
csl_dataset = datasets.ImageFolder(root='D:\\myspace\\dataset\\Classification\\satellite-image-classification',
                                   transform=custom_transform)
# 获取数据集类别数量
classes = csl_dataset.classes
print(classes)

train_ds, val_ds, test_ds = random_split(csl_dataset, [0.3, 0.3, 0.4])

val_loader = DataLoader(dataset=val_ds,
                        batch_size=16,
                        shuffle=True)

# 预测
model.eval()  # 设置为评估模式
with torch.no_grad():
    for features, targets in val_loader:
        predictions = model.forward(features.to(device))
        predictions = torch.argmax(predictions, dim=1)

        plt.figure(figsize=(15, 15))  # 设置窗口大小

        for i in range(len(features)):
            plt.subplot(4, 4, i + 1)
            plt.title("Prediction:{}\nTarget:{}".format(classes[predictions[i]], classes[targets[i]]))
            # 解决报错：Invalid shape (3, 224, 224) for image data
            # 问题产生的原因是由于matplotlib.pyplot 使用时传入的数组型或Tensor型参数应为 img=（224，224，3）这种类型。
            # 其中img[0],img[1]为数组或张量的长与宽,img[2]为维度，如‘RPG’为3
            img = features[i].swapaxes(0, 1)
            img = img.swapaxes(1, 2)
            plt.imshow(img)
            # 关闭坐标轴
            plt.axis('off')
            plt.tight_layout()

        plt.show()
        break