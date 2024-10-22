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
        denominator = torch.sum(self.cfm, dim=0).float()
        per_class_precision = torch.where(denominator != 0, torch.diag(self.cfm).float() / denominator,
                                          torch.zeros_like(denominator))
        total_precision = torch.diag(self.cfm).sum().float() / denominator.sum().float()
        return per_class_precision, total_precision


    def recall(self):
        denominator = torch.sum(self.cfm, dim=1).float()
        per_class_recall = torch.where(denominator != 0, torch.diag(self.cfm).float() / denominator,
                                       torch.zeros_like(denominator))
        total_recall = torch.diag(self.cfm).sum().float() / denominator.sum().float()
        return per_class_recall, total_recall

    def kappa(self):
        total = self.cfm.sum().float()

        # 观察到的准确率
        p_o = torch.diag(self.cfm).sum() / total

        # 期望的准确率
        sum_over_rows = torch.sum(self.cfm, dim=1)
        sum_over_cols = torch.sum(self.cfm, dim=0)
        p_e = (sum_over_rows * sum_over_cols).sum() / (total * total)

        # 计算Kappa系数
        kappa = (p_o - p_e) / (1 - p_e)
        return kappa

    def compute(self):
        acc = self.acc()
        iou = self.iou()
        precision = self.precision()
        recall = self.recall()
        kappa = self.kappa()
        results = {
            'acc': acc[0].tolist(),  # 每类的准确率
            'total_acc': acc[1].item(),  # 总体准确率
            'iou': iou[0].tolist(),  # 每类的 IoU
            'total_iou': iou[1].item(),  # 总体 IoU
            'precision': precision[0].tolist(),  # 每类的精确率
            'total_precision': precision[1].item(),  # 总体精确率
            'recall': recall[0].tolist(),  # 每类的召回率
            'total_recall': recall[1].item(),  # 总体召回率
            'kappa': kappa.item() # Kappa系数
        }

        return results


if __name__ == '__main__':

    # 假设我们有一个3类的语义分割任务
    class_num = 3
    metrics = Metrics(class_num)

    # 模拟一些样本的真实标签和预测标签
    true_vectors = torch.tensor([0, 1, 2, 1, 0, 2, 2, 1, 0, 1])
    pred_vectors = torch.tensor([0, 2, 2, 1, 0, 2, 0, 1, 1, 1])

    # 将这些样本添加到混淆矩阵中
    metrics.sample_add(true_vectors, pred_vectors)

    # 计算所有的指标
    results = metrics.compute()

    # 输出结果
    print("评估指标:")
    for key, value in results.items():
        print(f"{key}: {value}")
