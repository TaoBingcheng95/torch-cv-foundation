"""
https://captain-whu.github.io/SCD/

This the official implementation version of Separated Kappa (SeK),
which requires prediction maps with size of H x W x Num_class as the input and calculates the SeK.

Put the model predictions in INFER_DIR and extract the label files in LABEL_DIR.
The the evaluation results can be obtained by running the Metric.py.

为了减轻标签不平衡的影响，我们利用平均交并比 （mIOU） 来评估 BCD 结果，并提出分离 Kappa （SeK） 系数来评估 SCD 结果。
"""

from PIL import Image
import numpy as np
import math
import os


# num_class = 37
# IMAGE_FORMAT = '.png'
# INFER_DIR = './prediction_dir/'
# LABEL_DIR = './label_dir/'


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def get_hist(image, label, num_class):
    hist = np.zeros((num_class, num_class))
    hist += fast_hist(image.flatten(), label.flatten(), num_class)
    return hist


def cal_kappa(hist):
    if hist.sum() == 0:
        po = 0
        pe = 1
        kappa = 0
    else:
        po = np.diag(hist).sum() / hist.sum()
        pe = np.matmul(hist.sum(1), hist.sum(0).T) / hist.sum() ** 2
        if pe == 1:
            kappa = 0
        else:
            kappa = (po - pe) / (1 - pe)
    return kappa


def Eval(label_array, infer_array, num_class=5):
    # hist = np.zeros((num_class, num_class))
    # name_list = sorted(os.listdir(INFER_DIR))
    # for idx in range(len(name_list)):
    #     name = name_list[idx].split('.')[0]
    #     infer_file = INFER_DIR + '/' + str(name) + IMAGE_FORMAT
    #     label_file = LABEL_DIR + '/' + str(name) + IMAGE_FORMAT
    #     infer = Image.open(infer_file)
    #     label = Image.open(label_file)
    #     infer_array = np.array(infer)
    #     label_array = np.array(label)
    #     hist += get_hist(infer_array, label_array)

    hist = get_hist(infer_array, label_array, num_class)

    hist_fg = hist[1:, 1:]
    c2hist = np.zeros((2, 2))
    c2hist[0][0] = hist[0][0]
    c2hist[0][1] = hist.sum(1)[0] - hist[0][0]
    c2hist[1][0] = hist.sum(0)[0] - hist[0][0]
    c2hist[1][1] = hist_fg.sum()
    hist_n0 = hist.copy()
    hist_n0[0][0] = 0
    kappa_n0 = cal_kappa(hist_n0)
    iu = np.diag(c2hist) / (c2hist.sum(1) + c2hist.sum(0) - np.diag(c2hist))
    IoU_fg = iu[1]
    IoU_mean = (iu[0] + iu[1]) / 2
    Sek = (kappa_n0 * math.exp(IoU_fg)) / math.e

    print('Mean IoU = %.5f' % IoU_mean)
    print('Sek = %.5f' % Sek)


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
    truth, pred = prepare_data()

    Eval(truth, pred)

    # overall_iou: 0.639344262295082
