# -*- coding: utf-8 -*-
# @Time    : 2024/8/31 23:55
# @Author  : xuxing
# @Site    : 
# @File    : load_config.py
# @Software: PyCharm

import yaml
import torch
import torch.optim as optim
import torch.nn as nn

# 加载配置文件
from torch.utils.data import DataLoader

from datasets.transform import transform
from datasets.whdld import WhdldDataset
from models.unet import UNet


class TrainConfig:
    def __init__(self, yaml_file):
        self.config = self.load_config(yaml_file)
        self.device = self.config['training']['device']
        self.model = None
        self.transform = None
        self.optimizer = None
        self.loss_function = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        
        self.setup()


    def load_config(self, yaml_file):
        with open(yaml_file, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        return config

    def setup(self):
        self.train_loader = self.create_data_loader(mode='train')
        self.val_loader = self.create_data_loader(mode='val')
        self.create_model()
        self.create_optimizer()
        self.create_loss_function()
        self.create_scheduler()

    def summary(self):
        return {
            'train_data': self.train_loader,
            'val_data': self.val_loader,
            'model': self.model,
            'optimizer': self.optimizer,
            'loss': self.loss_function,
            'scheduler': self.scheduler
        }
    
    def create_data_loader(self, mode='train'):
        shuffle = True if mode == 'train' else False
        dataset = WhdldDataset(self.config['dataset'][mode], transform=transform['train'])
        data_loader = DataLoader(dataset, batch_size=self.config['training']['batch_size'],
                                  num_workers=self.config['dataset']['num_workers'], shuffle=shuffle)
        return data_loader
        
    def create_model(self):
        model_config = self.config
        name = model_config['model']['name']
        num_classes = model_config['dataset']['num_classes']
        input_channels = model_config['dataset']['in_channels']

        if name == "unet":
            self.model = UNet(in_channels=input_channels, out_channels=num_classes).to(self.device)
        else:
            raise ValueError(f"Unknown model name: {name}")
        
        return self.model

    def create_optimizer(self):
        training_config = self.config['training']
        optimizer_name = training_config['optimizer']
        learning_rate = training_config['learning_rate']

        if optimizer_name == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_name == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    def create_loss_function(self):
        loss_function_name = self.config['training']['loss_function']

        if loss_function_name == "cross_entropy":
            self.loss_function = nn.CrossEntropyLoss()
        elif loss_function_name == "mse":
            self.loss_function = nn.MSELoss()
        else:
            raise ValueError(f"Unknown loss function: {loss_function_name}")

    def create_scheduler(self):
        scheduler_config = self.config['training']['scheduler']
        scheduler_type = scheduler_config.get("type")

        if scheduler_type == "step_lr":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config["step_size"],
                gamma=scheduler_config["gamma"]
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")


class DatasetConfig:
    def __init__(self, yaml_file):
        self.config = self.load_config(yaml_file)
        self.cls_dict = self.config.get('cls_dict', {})
    
    def load_config(self, yaml_file):
        with open(yaml_file, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        return config
    
    def get_class_info(self):
        return self.cls_dict
    
    def get_class_names(self):
        return list(self.cls_dict.keys())
    
    def get_color_mapping(self):
        return {info['cls']: tuple(info['color']) for info in self.cls_dict.values()}
    
    def get_class_ids(self):
        return {cls_info['color']: cls_info['cls'] for cls_info in self.cls_dict.values()}
    
    def print_summary(self):
        print("Class Dictionary:")
        for class_name, class_info in self.cls_dict.items():
            print(f"{class_name}: Class ID = {class_info['cls']}, Color = {class_info['color']}")



def get_train_config(config_file = 'model.yaml'):
    return TrainConfig(config_file)

def get_dataset_config(config_file = 'config_dataset.yaml'):
    return DatasetConfig(config_file)
    


if __name__ == "__main__":
    config_file = 'model/model.yaml'  # 替换为你的 YAML 文件路径
    train_config = TrainConfig(config_file)

    # 使用配置文件 'config_dataset.yaml'
    yaml_file = 'data/config_dataset.yaml'
    dataset_config = DatasetConfig(yaml_file)

    # 打印所有类别信息
    print("Testing DatasetConfig:")
    dataset_config.print_summary()

    # 获取类别信息
    class_info = dataset_config.get_class_info()
    print("\nClass Info:", class_info)

    # 获取类别名称
    class_names = dataset_config.get_class_names()
    print("\nClass Names:", class_names)

    # 获取颜色到类别的映射
    class_colors = dataset_config.get_color_mapping()
    print("\nClass Colors:", class_colors)

    # 获取颜色到类别ID的映射
    class_ids = dataset_config.get_class_ids()
    print("\nClass IDs from Colors:", class_ids)


