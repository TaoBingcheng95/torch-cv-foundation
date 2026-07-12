# https://mp.weixin.qq.com/s/tiqqFdjTip8IaPx6AMXVpQ
# import os
import numpy as np
import torch

from torchmetrics.classification import (
    MulticlassConfusionMatrix,
    MulticlassAccuracy,
    MulticlassJaccardIndex,  # IoU
    MulticlassPrecision,
    MulticlassF1Score,
    MulticlassRecall,
    CohenKappa,
)



class TorchMetricsWrapper:
    """
    使用 torchmetrics 复现自定义 Metrics 类的功能
    """
    def __init__(self, num_classes: int, device: str = 'cpu', average: str = 'none'):
        """
        :param num_classes: 分类类别数
        :param device: 计算设备
        :param average: 
            - 'none': 返回每类指标 + 总体指标
            - 'macro': 返回宏平均
            - 'weighted': 返回加权平均
        """
        self.num_classes = num_classes
        self.device = torch.device(device)
        self.average = average
        
        # 初始化指标（自动处理设备）
        self.confmat = MulticlassConfusionMatrix(num_classes=num_classes, normalize=None).to(self.device)
        self.accuracy = MulticlassAccuracy(num_classes=num_classes, average='none' if average == 'none' else average).to(self.device)
        self.iou = MulticlassJaccardIndex(  # IoU = Jaccard Index
            num_classes=num_classes, average='none' if average == 'none' else average).to(self.device)
        self.precision = MulticlassPrecision(num_classes=num_classes, average='none' if average == 'none' else average).to(self.device)
        self.recall = MulticlassRecall(num_classes=num_classes, average='none' if average == 'none' else average).to(self.device)
        self.kappa = CohenKappa(task="multiclass", num_classes=num_classes).to(self.device)
    
    def reset(self):
        """重置所有指标状态（每个 epoch 开始前调用）"""
        self.confmat.reset()
        self.accuracy.reset()
        self.iou.reset()
        self.precision.reset()
        self.recall.reset()
        self.kappa.reset()
    
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        累积一个 batch 的预测结果
        :param preds: 模型输出的 logits 或 prob，shape=(B, num_classes) 或 (B,)
        :param target: 真实标签，shape=(B,)
        """
        # 如果传入的是 logits，先取 argmax
        if preds.dim() > 1 and preds.shape[1] == self.num_classes:
            preds = torch.argmax(preds, dim=1)
        
        # 确保标签在有效范围内（自动过滤 -100 等忽略索引）
        valid_mask = (target >= 0) & (target < self.num_classes)
        if not valid_mask.any():
            return
        
        preds_valid = preds[valid_mask]
        target_valid = target[valid_mask]
        
        # 批量更新所有指标
        # print(preds_valid.device)
        # print(target_valid.device)
        # print(self.confmat.device)
        self.confmat.update(preds_valid, target_valid)
        self.accuracy.update(preds_valid, target_valid)
        self.iou.update(preds_valid, target_valid)
        self.precision.update(preds_valid, target_valid)
        self.recall.update(preds_valid, target_valid)
        self.kappa.update(preds_valid, target_valid)
    
    def compute(self) -> dict:
        """
        计算并返回所有指标
        :return: 与自定义 Metrics 类相同格式的字典
        """
        # 获取每类指标（如果 average='none'）
        acc_per_class = self.accuracy.compute()  # shape: (num_classes,) or scalar
        iou_per_class = self.iou.compute()
        precision_per_class = self.precision.compute()
        recall_per_class = self.recall.compute()
        
        # 总体指标
        total_acc = acc_per_class.mean() if acc_per_class.dim() > 0 else acc_per_class
        total_iou = iou_per_class.mean() if iou_per_class.dim() > 0 else iou_per_class
        total_precision = precision_per_class.mean() if precision_per_class.dim() > 0 else precision_per_class
        total_recall = recall_per_class.mean() if recall_per_class.dim() > 0 else recall_per_class
        kappa = self.kappa.compute()
        
        # 混淆矩阵
        cfm = self.confmat.compute()  # shape: (num_classes, num_classes)
        
        # 构建返回字典（保持与自定义类兼容）
        results = {
            'confmat': cfm.cpu().numpy(),  # 可选：返回混淆矩阵
            'acc': acc_per_class.cpu().tolist() if acc_per_class.dim() > 0 else [acc_per_class.item()],
            'total_acc': total_acc.item(),
            'iou': iou_per_class.cpu().tolist() if iou_per_class.dim() > 0 else [iou_per_class.item()],
            'total_iou': total_iou.item(),
            'precision': precision_per_class.cpu().tolist() if precision_per_class.dim() > 0 else [precision_per_class.item()],
            'total_precision': total_precision.item(),
            'recall': recall_per_class.cpu().tolist() if recall_per_class.dim() > 0 else [recall_per_class.item()],
            'total_recall': total_recall.item(),
            'kappa': kappa.item(),
        }
        return results


class Metrics:
    def __init__(self, class_num, device='cpu'):
        self.class_num = class_num
        self.device = device
        self.cfm = self.cfm_init(self.class_num)

    def cfm_init(self, class_num, dtype=torch.int64):
        return torch.zeros(size=(class_num, class_num), dtype=dtype).to(self.device)

    def reset(self):
        # self.cfm = self.cfm_init(self.class_num)
        self.cfm.zero_()

    def sample_add(self, true_vector, pre_vector):
        true_vector = true_vector.flatten()
        pre_vector = pre_vector.flatten()
        valid_mask = (true_vector >= 0) & (true_vector < self.class_num)
        if not valid_mask.any():
            return  # ✅ 无有效样本时直接返回
        # self.cfm += torch.bincount(self.class_num * true_vector[mask] + pre_vector[mask],
        #                            minlength=self.class_num ** 2).reshape(self.class_num, self.class_num).to(
        #     self.device)
        true_valid = true_vector[valid_mask]
        pre_valid = pre_vector[valid_mask]
        
        # bincount 计算混淆矩阵
        indices = self.class_num * true_valid + pre_valid
        cfm_batch = torch.bincount(indices, minlength=self.class_num ** 2)
        self.cfm += cfm_batch.reshape(self.class_num, self.class_num).to(self.device)

    def acc(self):
        per_class_acc = torch.diag(self.cfm).float() / torch.sum(self.cfm, dim=1).float()
        total_acc = torch.diag(self.cfm).sum().float() / self.cfm.sum().float()
        return per_class_acc, total_acc

    # def iou(self):
    #     per_class_iou = torch.diag(self.cfm).float() / (
    #             torch.sum(self.cfm, dim=1).float() + torch.sum(self.cfm, dim=0).float() - torch.diag(
    #         self.cfm).float())
    #     total_iou = torch.diag(self.cfm).sum().float() / (
    #             torch.sum(self.cfm).float() + torch.sum(self.cfm).float() - torch.diag(self.cfm).sum().float())
    #     return per_class_iou, total_iou
    def iou(self, eps=1e-6):
        intersection = torch.diag(self.cfm).float()
        union = torch.sum(self.cfm, dim=1).float() + torch.sum(self.cfm, dim=0).float() - intersection
        per_class_iou = torch.where(union > eps, intersection / union, torch.zeros_like(intersection))
        total_iou = intersection.sum() / (union.sum() + eps)
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


class SegmentationMetric(object):
    def __init__(self, numClass, imgPredict, imgLabel):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)  # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0
        return meanAcc  # 返回单个值，如：np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96） / 3 =  0.89

    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)  # 取对角元素的值，返回列表
        # union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
        #     self.confusionMatrix)  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - intersection
        IoU = intersection / union  # 返回列表，其值为各个类别的IoU
        mIoU = np.nanmean(IoU)  # 求各类别IoU的平均
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel):  # 同FCN中score.py的fast_hist()函数
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        # print(mask.shape)
        # print(label.shape)
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusionMatrix, axis=1) / np.sum(self.confusionMatrix)
        iu = np.diag(self.confusionMatrix) / (
                np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) -
                np.diag(self.confusionMatrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))



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

    """
    img1 = torch.Tensor([[0, 1, 2], [0, 1, 0]])
    img2 = torch.Tensor([[0, 1, 2], [2, 1, 1]])
    e = SegmentationMetric(3, img1, img2)
    print(e.Frequency_Weighted_Intersection_over_Union())
    print(e.pixelAccuracy())
    print(e.confusionMatrix)
    """
