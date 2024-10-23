import numpy as np
from sklearn.metrics import confusion_matrix, cohen_kappa_score, jaccard_score
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.metrics import f1_score, fbeta_score


class MetricsSKLML:
    def __init__(self, truth, pred):
        self.truth = truth.flatten()
        self.pred = pred.flatten()
        self.cm = None
        self.num_classes = None

        self.class_metrics = {}
        self.overall_metrics = {}

        self.get_confusion_matrix()

        self.calculate_metrics()

    def get_confusion_matrix(self):
        """计算混淆矩阵"""
        self.cm = confusion_matrix(self.truth, self.pred)
        self.num_classes = self.cm.shape[0]

    def calculate_metrics(self):
        """计算每个类别和全局的TP、TN、FP、FN"""
        total_TP = 0
        total_FP = 0
        total_FN = 0
        total_TN = 0
        for i in range(self.num_classes):
            TP = self.cm[i, i]
            FP = np.sum(self.cm[:, i]) - TP
            FN = np.sum(self.cm[i, :]) - TP
            TN = np.sum(self.cm) - (TP + FP + FN)
            total_TP += TP
            total_FP += FP
            total_FN += FN
            total_TN += TN
            self.class_metrics[i] = {"TP": TP, "TN": TN, "FP": FP, "FN": FN}  # modify base your data
            # print(f"Class {i}: TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
        # 全局TP、FP、FN、TN
        # print(f"Total TP: {total_TP}, Total TN: {total_TN}, Total FP: {total_FP}, Total FN: {total_FN}")
        self.overall_metrics = {"TP": total_TP, "TN": total_TN, "FP": total_FP, "FN": total_FN}

    def calculate_accuracy(self):
        """
        计算每个类别和全局的准确率Accuracy
        在整个模型中，所有判断正确的结果占总样本数量的比重
        (TP+TN)/(TP+FP+FN+TN)
        """
        class_accuracies = [] # {}
        for i, metrics in self.class_metrics.items():
            # TP = metrics["TP"]
            # TN = metrics["TN"]
            # FP = metrics["FP"]
            # FN = metrics["FN"]
            # accuracy = (TP + TN) / (TP + TN + FP + FN)
            class_indices = [index for index, value in enumerate(self.truth) if value == i]
            class_truth = [self.truth[index] for index in class_indices]
            class_pred = [self.pred[index] for index in class_indices]
            # class_accuracies[i] = accuracy_score(class_truth, class_pred)
            class_accuracies.append(accuracy_score(class_truth, class_pred))

        # total_TP = self.overall_metrics["TP"]
        # total_TN = self.overall_metrics["TN"]
        # total_FP = self.overall_metrics["FP"]
        # total_FN = self.overall_metrics["FN"]
        # overall_accuracy = (total_TP + total_TN) / (total_TP + total_TN + total_FP + total_FN)
        overall_accuracy = accuracy_score(self.truth, self.pred)
        return class_accuracies, overall_accuracy

    def calculate_precision(self, average='micro'):
        """
        计算每个类别和全局的精确率
        在模型预测为正的结果中，模型预测对的数量所占的比重，
        TP/(TP+FP)

        micro平均
            计算方式：先计算全局的 TP、FP、FN，然后基于这些总数来计算指标。
            适用场景：micro 平均在类别不平衡时非常有用，因为它会考虑所有样本的贡献。
            影响：强调频率较高的类别，与样本数量成正比。
        macro 平均
            计算方式：对每个类别分别计算指标，然后取这些指标的算术平均。
            适用场景：macro 平均在关注所有类别对指标的贡献时有用，无论类别的样本大小。
            影响：对每个类别一视同仁，可能会忽略样本较少类别的影响。
        总结
            micro 强调全局表现，适合处理类别不平衡问题。
            macro 强调每个类别的独立表现，适合分析每个类别的单独表现。
        """
        overall_precision = precision_score(self.truth, self.pred, average=average)
        class_precisions = precision_score(self.truth, self.pred, average=None)
        # class_precision_dict = {i: class_precisions[i] for i in range(len(class_precisions))}
        return class_precisions, overall_precision

    def calculate_recall(self, average='micro'):
        """
        计算每个类别和全局的召回率recall/灵敏度Sensitivity
        在所有真实值是正的结果中，模型预测对的数量所占比重
        TP/(TP+FN)
        """
        overall_recall = recall_score(self.truth, self.pred, average=average)
        class_recalls = recall_score(self.truth, self.pred, average=None)
        # class_recall_dict = {i: class_recall[i] for i in range(len(class_recall))}
        return class_recalls, overall_recall

    def calculate_specificity(self, average='micro'):
        """
        计算每个类别和全局的特异度Specificity
        在所有真实值是负的结果中，模型预测对的数量所占比重
        TN/(FP+TN)
        """
        class_specificities = [] #{}
        for i, metrics in self.class_metrics.items():
            # TP = metrics["TP"]
            TN = metrics["TN"]
            FP = metrics["FP"]
            # FN = metrics["FN"]
            # class_specificities[i] = TN / (TN + FP)
            class_specificities.append(TN / (TN + FP))
        # total_TP = self.overall_metrics["TP"]
        total_TN = self.overall_metrics["TN"]
        total_FP = self.overall_metrics["FP"]
        # total_FN = self.overall_metrics["FN"]
        overall_specificity = total_TN / (total_TN + total_FP)
        return class_specificities, overall_specificity

    def calculate_f1_score(self, average='micro'):
        """
        计算每个类别和全局的F1分数

        Precision表示指预测的准确性, 所以研究区域为预测为正类的样本, 对应查准率。
        Recall侧重原始样本的完整性, 所以研究区域为原始类别为正类的样本，对应查全率。

        查准率和查全率是一对矛盾的度量。
        一般情况下，查准率高时查全率往往偏低；而查全率高时查准率往往偏低。
        在Precision和Recall的基础上提出了F1值的概念, 来对Precision和Recall进行整体评价，用于综合反映整体的指标。
        """
        overall_f1 = f1_score(self.truth, self.pred, average=average)
        class_f1_scores = f1_score(self.truth, self.pred, average=None)
        # class_f1_dict = {i: class_f1_scores[i] for i in range(len(class_f1_scores))}
        return class_f1_scores, overall_f1

    def calculate_jaccard_score(self, average='micro'):
        """
        计算每个类别和全局的Jaccard系数
        (TP)/(TP+FP+FN)
        """
        overall_jaccard = jaccard_score(self.truth, self.pred, average=average)
        class_jaccards = jaccard_score(self.truth, self.pred, average=None)
        # class_jaccard_dict = {i: class_jaccards[i] for i in range(len(class_jaccards))}
        return class_jaccards, overall_jaccard

    def calculate_cohen_kappa_score(self):
        return cohen_kappa_score(self.truth, self.pred)

    def calculate_falpha_score(self, alpha=1.0, average='micro'):
        """
        计算每个类别和全局的Fβ分数
        当 alpha = 1.0 时, Fβ分数实际上就是 F1 分数。
        这意味着精确率和召回率的权重是相等的。
        F1 分数是精确率和召回率的调和平均数，在这种情况下，它们的权重相同。
        通过调整β，修正Precision的权重
        """
        overall_falpha = fbeta_score(self.truth, self.pred, beta=alpha, average=average)
        class_falpha_scores = fbeta_score(self.truth, self.pred, beta=alpha, average=None)

        class_falpha_dict = {i: class_falpha_scores[i] for i in range(len(class_falpha_scores))}

        return class_falpha_dict, overall_falpha

    def get_class_metrics(self):
        """返回每个类别的指标"""
        return self.class_metrics

    def get_overall_metrics(self):
        """返回全局的指标"""
        return self.overall_metrics


class MetricsSKL:
    def __init__(self, num_classes=2, average='micro'):
        self.num_classes = num_classes
        self.average = average
        self.truth = None
        self.pred = None
        self.cm = None
        self.num_classes = None
        self.class_metrics = {}
        self.overall_metrics = {}
        self.reset()

    def reset(self):
        """重置数据"""
        self.truth = None
        self.pred = None
        self.cm = None
        self.num_classes = None
        self.class_metrics = {}
        self.overall_metrics = {}

    def cfm_init(self, class_num):
        return np.zeros(shape=(class_num, class_num), dtype=np.int32)

    def sample_add(self, truth, pred):
        """批量添加数据"""
        # self.truth.extend(truth.flatten())
        # self.pred.extend(pred.flatten())
        if self.truth is None:
            self.truth = truth.flatten()
            self.pred = pred.flatten()
        else:
            self.truth = np.concatenate((self.truth, truth.flatten()))
            self.pred = np.concatenate((self.pred, pred.flatten()))

    def compute(self):
        """计算所有指标"""
        compute_result = {}
        self.get_confusion_matrix()
        self.statistics_metrics()

        class_res, overall_res = self.calculate_accuracy()
        compute_result["class_accuracies"] = class_res
        compute_result["overall_accuracy"] = overall_res

        class_res, overall_res = self.calculate_precision()
        compute_result["class_precisions"] = class_res
        compute_result["overall_precision"] = overall_res

        class_res, overall_res = self.calculate_recall()
        compute_result["class_recalls"] = class_res
        compute_result["overall_recall"] = overall_res

        class_res, overall_res = self.calculate_f1_score()
        compute_result["class_f1_scores"] = class_res
        compute_result["overall_f1_score"] = overall_res

        class_res, overall_res = self.calculate_iou()
        compute_result["class_ious"] = class_res
        compute_result["overall_iou"] = overall_res

        compute_result["cohen_kappa_score"] = self.calculate_cohen_kappa_score()
        return compute_result

    def get_confusion_matrix(self):
        """计算混淆矩阵"""
        self.cm = confusion_matrix(self.truth, self.pred)
        self.num_classes = self.cm.shape[0]

    def statistics_metrics(self):  # calculate_metrics
        """计算每个类别和全局的TP、TN、FP、FN"""
        total_TP = 0
        total_FP = 0
        total_FN = 0
        total_TN = 0
        for i in range(self.num_classes):
            TP = self.cm[i, i]
            FP = np.sum(self.cm[:, i]) - TP
            FN = np.sum(self.cm[i, :]) - TP
            TN = np.sum(self.cm) - (TP + FP + FN)
            total_TP += TP
            total_FP += FP
            total_FN += FN
            total_TN += TN
            self.class_metrics[i] = {"TP": TP, "TN": TN, "FP": FP, "FN": FN}  # modify base your data
            # print(f"Class {i}: TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
        # 全局TP、FP、FN、TN
        # print(f"Total TP: {total_TP}, Total TN: {total_TN}, Total FP: {total_FP}, Total FN: {total_FN}")
        self.overall_metrics = {"TP": total_TP, "TN": total_TN, "FP": total_FP, "FN": total_FN}

    def calculate_accuracy(self):
        """
        计算每个类别和全局的准确率Accuracy
        在整个模型中，所有判断正确的结果占总样本数量的比重
        (TP+TN)/(TP+FP+FN+TN)
        """
        class_accuracies = [] # {}
        for i, metrics in self.class_metrics.items():
            # TP = metrics["TP"]
            # TN = metrics["TN"]
            # FP = metrics["FP"]
            # FN = metrics["FN"]
            # accuracy = (TP + TN) / (TP + TN + FP + FN)
            class_indices = [index for index, value in enumerate(self.truth) if value == i]
            class_truth = [self.truth[index] for index in class_indices]
            class_pred = [self.pred[index] for index in class_indices]
            # class_accuracies[i] = accuracy_score(class_truth, class_pred)
            class_accuracies.append(accuracy_score(class_truth, class_pred))

        # total_TP = self.overall_metrics["TP"]
        # total_TN = self.overall_metrics["TN"]
        # total_FP = self.overall_metrics["FP"]
        # total_FN = self.overall_metrics["FN"]
        # overall_accuracy = (total_TP + total_TN) / (total_TP + total_TN + total_FP + total_FN)
        overall_accuracy = accuracy_score(self.truth, self.pred)
        return class_accuracies, overall_accuracy

    def calculate_precision(self, average='micro'):
        """
        计算每个类别和全局的精确率
        在模型预测为正的结果中，模型预测对的数量所占的比重，
        TP/(TP+FP)

        micro平均
            计算方式：先计算全局的 TP、FP、FN，然后基于这些总数来计算指标。
            适用场景：micro 平均在类别不平衡时非常有用，因为它会考虑所有样本的贡献。
            影响：强调频率较高的类别，与样本数量成正比。
        macro 平均
            计算方式：对每个类别分别计算指标，然后取这些指标的算术平均。
            适用场景：macro 平均在关注所有类别对指标的贡献时有用，无论类别的样本大小。
            影响：对每个类别一视同仁，可能会忽略样本较少类别的影响。
        总结
            micro 强调全局表现，适合处理类别不平衡问题。
            macro 强调每个类别的独立表现，适合分析每个类别的单独表现。
        """
        overall_precision = precision_score(self.truth, self.pred, average=average)
        class_precisions = precision_score(self.truth, self.pred, average=None)
        # class_precision_dict = {i: class_precisions[i] for i in range(len(class_precisions))}
        return class_precisions, overall_precision

    def calculate_recall(self, average='micro'):
        """
        计算每个类别和全局的召回率recall/灵敏度Sensitivity
        在所有真实值是正的结果中，模型预测对的数量所占比重
        TP/(TP+FN)
        """
        overall_recall = recall_score(self.truth, self.pred, average=average)
        class_recalls = recall_score(self.truth, self.pred, average=None)
        # class_recall_dict = {i: class_recalls[i] for i in range(len(class_recalls))}
        return class_recalls, overall_recall

    def calculate_specificity(self, average='micro'):
        """
        计算每个类别和全局的特异度Specificity
        在所有真实值是负的结果中，模型预测对的数量所占比重
        TN/(FP+TN)
        """
        class_specificities = [] # {}
        for i, metrics in self.class_metrics.items():
            TN = metrics["TN"]
            FP = metrics["FP"]
            # class_specificities[i] = TN / (TN + FP)
            class_specificities.append(TN / (FP + TN))
        total_TN = self.overall_metrics["TN"]
        total_FP = self.overall_metrics["FP"]
        overall_specificity = total_TN / (total_TN + total_FP)
        return class_specificities, overall_specificity

    def calculate_f1_score(self, average='micro'):
        """
        计算每个类别和全局的F1分数

        Precision表示指预测的准确性, 所以研究区域为预测为正类的样本, 对应查准率。
        Recall侧重原始样本的完整性, 所以研究区域为原始类别为正类的样本，对应查全率。

        查准率和查全率是一对矛盾的度量。
        一般情况下，查准率高时查全率往往偏低；而查全率高时查准率往往偏低。
        在Precision和Recall的基础上提出了F1值的概念, 来对Precision和Recall进行整体评价，用于综合反映整体的指标。
        """
        overall_f1 = f1_score(self.truth, self.pred, average=average)
        class_f1_scores = f1_score(self.truth, self.pred, average=None)
        # class_f1_dict = {i: class_f1_scores[i] for i in range(len(class_f1_scores))}
        return class_f1_scores, overall_f1

    def calculate_jaccard_score(self, average='micro'):
        """
        计算每个类别和全局的Jaccard系数
        (TP)/(TP+FP+FN)
        """
        overall_jaccard = jaccard_score(self.truth, self.pred, average=average)
        class_jaccards = jaccard_score(self.truth, self.pred, average=None)
        # class_jaccard_dict = {i: class_jaccards[i] for i in range(len(class_jaccards))}
        return class_jaccards, overall_jaccard

    def calculate_iou(self, average='micro'):
        overall_iou = jaccard_score(self.truth, self.pred, average=average)
        class_ious = jaccard_score(self.truth, self.pred, average=None)
        # class_iou_dict = {i: class_ious[i] for i in range(len(class_ious))}
        return class_ious, overall_iou


    def calculate_cohen_kappa_score(self):
        return cohen_kappa_score(self.truth, self.pred)

    def calculate_falpha_score(self, alpha=1.0, average='micro'):
        """
        计算每个类别和全局的Fβ分数
        当 alpha = 1.0 时, Fβ分数实际上就是 F1 分数。
        这意味着精确率和召回率的权重是相等的。
        F1 分数是精确率和召回率的调和平均数，在这种情况下，它们的权重相同。
        通过调整β，修正Precision的权重
        """
        overall_falpha = fbeta_score(self.truth, self.pred, beta=alpha, average=average)
        class_falpha_scores = fbeta_score(self.truth, self.pred, beta=alpha, average=None)
        # class_falpha_dict = {i: class_falpha_scores[i] for i in range(len(class_falpha_scores))}
        return class_falpha_scores, overall_falpha

    def get_class_metrics(self):
        """返回每个类别的指标"""
        return self.class_metrics

    def get_overall_metrics(self):
        """返回全局的指标"""
        return self.overall_metrics


def prepare_data():
    truth_1 = [1] * 100 + [2] * 11 + [3] * 14 + [4] * 2 + [5] * 3
    predict_1 = [1] * 130
    truth_2 = [1] * 13 + [2] * 92 + [3] * 15 + [4] * 6 + [5] * 4
    predict_2 = [2] * 130
    truth_3 = [1] * 17 + [2] * 13 + [3] * 94 + [4] * 4 + [5] * 2
    predict_3 = [3] * 130
    truth_4 = [1] * 5 + [2] * 9 + [3] * 6 + [4] * 107 + [5] * 3
    predict_4 = [4] * 130
    truth_5 = [1] * 2 + [2] * 5 + [3] * 5 + [4] * 4 + [5] * 114
    predict_5 = [5] * 130

    truth = np.asarray(truth_1 + truth_2 + truth_3 + truth_4 + truth_5)
    pred = np.asarray(predict_1 + predict_2 + predict_3 + predict_4 + predict_5)
    truth = truth - 1
    pred = pred - 1
    return truth, pred


if __name__ == '__main__':
    # 假设我们有一个3类的语义分割任务
    class_num = 3
    metrics = MetricsSKL(num_classes=class_num)

    # 模拟一些样本的真实标签和预测标签
    # true_vectors = np.asarray([0, 1, 2, 1, 0, 2, 2, 1, 0, 1])
    # pred_vectors = np.asarray([0, 2, 2, 1, 0, 2, 0, 1, 1, 1])
    true_vectors, pred_vectors = prepare_data()
    metrics.sample_add(true_vectors, pred_vectors)

    # 计算所有的指标
    results = metrics.compute()

    # 输出结果
    print("评估指标:")
    for key, value in results.items():
        print(f"{key}: {value}")



