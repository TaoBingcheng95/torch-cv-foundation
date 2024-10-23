# -*- coding: utf-8 -*-
# @Time    : 2024/8/31 22:17
# @Author  : comet
# @Site    :
# @File    : predictor.py
# @Software: PyCharm

import os
from PIL import Image
import torch

from config.load_config import get_train_config
# from datasets.transform import transform
from tqdm import tqdm
import numpy as np


class Predictor:
    def __init__(self,
                 weight,
                 yaml_fp='./config/model.yaml',
                 transform=None,
                 device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = get_train_config(yaml_fp).model
        self.model.eval()
        self.weight = weight
        self.transform = transform

        self.load_weight()

    def load_weight(self):
        self.model.load_state_dict(torch.load(self.weight))
        print('load model weight from ', self.weight)

    def predict(self, image_path):
        image = np.array(Image.open(image_path))
        if self.transform:
            image = self.transform(image=image)['image'].unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(image)
            _, pred = torch.max(output, 1)

        return pred.squeeze().cpu().numpy()

    def predict_folder(self, input_dir, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("start predict ...")
        for image_name in tqdm(os.listdir(input_dir)):
            image_path = os.path.join(input_dir, image_name)
            pred_array = self.predict(image_path)

            # 保存预测结果为 .tif 文件
            pred_image = Image.fromarray(pred_array.astype('uint8'))
            save_path = os.path.join(output_dir, f"{image_name.split('.')[0]}.tif")
            pred_image.save(save_path)
        print(f"results save to {output_dir}")


if __name__ == '__main__':
    test_path = './data/WHDLD/outputs/test/images'
    pred_path = './data/WHDLD/outputs/test/preds'
    model_path = './outputs/20240902_102844/epoch_2_acc_0.7091_miou_0.5493.pth'

    predictor = Predictor(weight=model_path, transform=None)
    predictor.predict_folder(test_path, pred_path)
