# https://www.global-wheat.com/
# https://www.global-wheat.com/
# https://mp.weixin.qq.com/s/57IqCF_ZPx5U1azeVb0YdA

from PIL import Image
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import ast

import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
import torch.nn.functional as F

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import datasets,transforms

import albumentations as A   # pip install albumentations==1.1.0  
from albumentations.pytorch import ToTensorV2  


# 训练与验证数据增强，利用albumentations  随机翻转转换，随机图片处理
# 对象检测的增强与正常增强不同，因为在这里需要确保 bbox 在转换后仍然正确与对象对齐
train_transform = A.Compose([A.Flip(0.5),ToTensorV2(p=1.0)],
                            bbox_params = {'format':"pascal_voc",'label_fields': ['labels']}
                            )

val_transform = A.Compose([ToTensorV2(p=1.0)],
                          bbox_params = {'format':"pascal_voc","label_fields":['labels']}
                          )

def train_test_split(dataFrame,split):  
    len_tot = len(dataFrame)  
    val_len = int(split*len_tot)  
    train_len = len_tot-val_len  
    train_data,val_data = dataFrame.iloc[:train_len][:],dataFrame.iloc[train_len:][:]  
    return train_data,val_data 


def collate_fn(batch):
    """
    collate_fn默认是对数据（图片）通过torch.stack()进行简单的拼接。对于分类网络来说，默认方法是可以的（因为传入的就是数据的图片），
    但是对于目标检测来说，train_dataset返回的是一个tuple，即(image, target)。
    如果我们还是采用默认的合并方法，那么就会出错。
    所以我们需要自定义一个方法，即collate_fn=train_dataset.collate_fn
    """
    return tuple(zip(*batch)) 


class WheatDataset(Dataset):
    """
    定义WheatDataset 返回 图片，标签
    """
    def __init__(self,data,root_dir,transform=None,train=True):  
        self.data = data  
        self.root_dir = root_dir  
        self.image_names = self.data.image_id.values  
        self.bboxes = self.data.bboxes.values  
        self.transform = transform  
        self.isTrain = train

    def __len__(self):  
        return len(self.data)

    def __getitem__(self,index):  
        # print(self.image_names)
        # print(self.bboxes)
        img_path = os.path.join(self.root_dir,self.image_names[index]+".jpg")  # 拼接路径  
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)   # 读取图片  
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)  # BGR2RGB  
        image /= 255.0    # 归一化  
        bboxes = torch.tensor(self.bboxes[index],dtype=torch.float64)  
        """  
        As per the docs of torchvision  
        we need bboxes in format (xmin,ymin,xmax,ymax)  
        Currently we have them in format (xmin,ymin,width,height)  
        """
        # 格式转换 (xmin,ymin,width,height)-----> (xmin,ymin,xmax,ymax)
        bboxes[:,2] = bboxes[:,0]+bboxes[:,2]
        bboxes[:,3] = bboxes[:,1]+bboxes[:,3]  
        """  
        we need to return image and a target dictionary  
        target:  
            boxes,labels,image_id,area,iscrowd  
        """  
        area = (bboxes[:,3]-bboxes[:,1])*(bboxes[:,2]-bboxes[:,0])   # 计算面积  
        area = torch.as_tensor(area,dtype=torch.float32)  
          
        # there is only one class  
        labels = torch.ones((len(bboxes),),dtype=torch.int64)   # 标签  
          
        # suppose all instances are not crowded  
        iscrowd = torch.zeros((len(bboxes),),dtype=torch.int64)
        # target是个字典 里面 包括 boxes,labels,image_id,area,iscrowd
        target = {'boxes': bboxes, 'labels': labels, 'image_id': torch.tensor([index]), "area": area,
                  'iscrowd': iscrowd}

        if self.transform is not None:  
            sample = {  
                'image': image,  
                'bboxes': target['boxes'],  
                'labels': labels  
            }  
            sample = self.transform(**sample)  
            image = sample['image']  
              
            # 沿着一个新维度对输入张量序列进行连接。 序列中所有的张量都应该为相同形状,
            # 把多个2维的张量凑成一个3维的张量；多个3维的凑成一个4维的张量…以此类推，也就是在增加新的维度进行堆叠
            target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)  
              
        return image,target  


# 这一个类来保存对应的loss  
class Averager:  
    def __init__(self):  
        self.current_total = 0.0  
        self.iterations = 0.0  
  
    def send(self, value):  
        self.current_total += value  
        self.iterations += 1  
  
    @property  
    def value(self):  
        if self.iterations == 0:  
            return 0  
        else:  
            return 1.0 * self.current_total / self.iterations  
  
    def reset(self):  
        self.current_total = 0.0  
        self.iterations = 0.0 


if __name__ == '__main__':
    SPLIT = 0.2  
    DATAPATH = '../global-wheat-detection'
    BATCH_SIZE = 4
    DEVICE = "cuda" if  torch.cuda.is_available()  else  "cpu"
    LR = 1e-4
    EPOCHS = 10

    # 读取 train.csv文件
    df = pd.read_csv(DATAPATH + '/train.csv')  
    df.bbox = df.bbox.apply(ast.literal_eval)  # 将string of list 转成list数据
      
    # 利用groupby 将同一个image_id的数据进行聚合，方式为list进行，并且用reset_index直接转变成dataframe
    df = df.groupby("image_id")["bbox"].apply(list).reset_index(name="bboxes")
    train_data_df, val_data_df = train_test_split(df, SPLIT) #  划分 train val 8:2

    train_data = WheatDataset(train_data_df, DATAPATH + "/train", transform=train_transform)  
    valid_data = WheatDataset(val_data_df, DATAPATH + "/train", transform=val_transform) 

    #  构建训练和测试 dataloader  
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)  
    val_dataloader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weight=True)
    num_classes = 2
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    #  定义模型， 优化器，损失， 迭代，以及 学习率  
    train_loss = []  
    #  val_loss = []  
    model = model.to(DEVICE)  
    params =[p for  p  in  model.parameters()  if  p.requires_grad]  
    optimizer = optim.Adam(params, lr=LR)  
    loss_hist = Averager()  
    itr = 1  
    lr_scheduler = None  
      
    loss_hist = Averager()  
    itr = 1

    for  epoch  in  range(EPOCHS):
        loss_hist.reset()
        for  images,  targets  in  train_dataloader:
              #  print(images)
              #  print(targets)
              #  for image in images:
              #      print(image.dtype)  # torch.float32
              #  for t in targets:
              #      for k, v in t.items():
              #          print(k ,v.dtype)
            images = list(image.to(DEVICE) for  image  in  images)
            targets = [{k: v.to(DEVICE) for  k,  v  in  t.items()}  for  t  in  targets]
            loss_dict = model(images, targets)
            # for loss in loss_dict.values():
            #    print(loss.dtype)
            losses = sum(loss for  loss  in  loss_dict.values())
            loss_value = losses.item()
            loss_hist.send(loss_value)
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            if itr % 50 == 0:
                print(f"Iteration #{itr} loss: {loss_value}")
            itr  +=  1
        #  update the learning rate
        if lr_scheduler  is   not  None:
            lr_scheduler.step()
        print(f"Epoch #{epoch} loss: {loss_hist.value}")
    torch.save(model.state_dict(), 'fasterrcnn_resnet50_fpn.pth')

    images, targets = next(iter(val_dataloader))  
    images = list(img.to(DEVICE) for  img  in  images)  
    #  print(images[0].shape)  
    targets = [{k: v.to(DEVICE) for  k,  v  in  t.items()}  for  t  in  targets]  
    boxes = targets[1]['boxes'].cpu().numpy().astype(np.int32)  
    sample = images[1].permute(1, 2, 0).cpu().numpy()  
      
    model.eval()  
    cpu_device = torch.device("cpu")  
    #  print(images[0].shape)  
      
      
    outputs = model(images)  
    outputs = [{k: v.to(cpu_device) for  k,  v  in  t.items()}  for  t  in  outputs]  
    #  print(outputs[1]['boxes'].detach().numpy().astype(np.int32))  
      
    pred_boxes = outputs[1]['boxes'].detach().numpy().astype(np.int32)  
      
    fig, ax = plt.subplots(1, 1, figsize = (16, 8))  
      
    for  b,  box  in  zip(boxes,  pred_boxes):  
          #  绘制预测边框 红色表示  
        cv2.rectangle(sample,  
                      (box[0], box[1]),  
                      (box[2], box[3]),  
                      (220, 0,  0), 3)  
          #  绘制实际边框  绿色表示  
        cv2.rectangle(sample,  
                      (b[0], b[1]),  
                      (b[2], b[3]),  
                      (0, 220,  0), 3)  
      
    ax.set_axis_off()  
    ax.imshow(sample)  
    plt.show()  
