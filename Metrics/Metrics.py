import os
import torch


class Metrics:
    def __init__(self, class_num, device='cpu'):
        self.class_num = class_num
        self.device = device
        self.cfm = self.cfm_init(self.class_num)

    def cfm_init(self, class_num):
        return torch.zeros(size=(class_num, class_num), dtype=torch.int).to(self.device)

    def update(self):
        self.cfm = self.cfm_init(self.class_num)

    def sample_add(self, true_vector, pre_vector):
        true_vector = true_vector.flatten()
        pre_vector = pre_vector.flatten()

        mask = (true_vector >= 0) & (true_vector < self.class_num)
        self.cfm += torch.bincount(self.class_num * true_vector[mask] + pre_vector[mask],
                                   minlength=self.class_num ** 2).reshape(self.class_num, self.class_num).to(
            self.device)

    def acc(self):
        per_class_acc = torch.diag(self.cfm).float() / torch.sum(self.cfm, dim=1).float()
        total_acc = torch.diag(self.cfm).sum().float() / self.cfm.sum().float()
        return per_class_acc, total_acc

    def iou(self):
        per_class_iou = torch.diag(self.cfm).float() / (
                torch.sum(self.cfm, dim=1).float() + torch.sum(self.cfm, dim=0).float() - torch.diag(
            self.cfm).float())
        total_iou = torch.diag(self.cfm).sum().float() / (
                torch.sum(self.cfm).float() + torch.sum(self.cfm).float() - torch.diag(self.cfm).sum().float())
        return per_class_iou, total_iou

    def precision(self):
        per_class_precision = torch.diag(self.cfm).float() / torch.sum(self.cfm, dim=0).float()
        total_precision = torch.diag(self.cfm).sum().float() / torch.sum(self.cfm, dim=0).sum().float()
        return per_class_precision, total_precision

    def recall(self):
        per_class_recall = torch.diag(self.cfm).float() / torch.sum(self.cfm, dim=1).float()
        total_recall = torch.diag(self.cfm).sum().float() / torch.sum(self.cfm, dim=1).sum().float()
        return per_class_recall, total_recall

    def compute(self):
        acc = self.acc()
        iou = self.iou()
        precision = self.precision()
        recall = self.recall()

        results = {
            'acc': acc[0].tolist(),  # 每类的准确率
            'total_acc': acc[1].item(),  # 总体准确率
            'iou': iou[0].tolist(),  # 每类的 IoU
            'total_iou': iou[1].item(),  # 总体 IoU
            'precision': precision[0].tolist(),  # 每类的精确率
            'total_precision': precision[1].item(),  # 总体精确率
            'recall': recall[0].tolist(),  # 每类的召回率
            'total_recall': recall[1].item()  # 总体召回率
        }

        return results
