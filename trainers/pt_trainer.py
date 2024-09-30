# https://blog.itpub.net/18841117/viewspace-3015295/

import os
import sys
import time
import datetime
import copy

from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
# import logging

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader


from Metrics import Metrics

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

def setup_logging(output_dir):
    """
    设置loguru的日志记录，按照当前运行时间作为文件名保存

    :param output_dir: 日志文件保存的目录
    """
    os.makedirs(output_dir, exist_ok=True)
    log_file_path = os.path.join(output_dir, "training.log")
    logger.remove()
    logger.add(log_file_path, rotation="500 MB", retention="10 days", level="INFO")
    logger.add(lambda msg: print(msg, end=""), level="INFO")
    logger.info(f"Logging is set up. Logs are being saved to {log_file_path}.")


class Trainer_bak:
    def __init__(self, model: nn.Module,
                 train_dl: DataLoader = None,
                 val_dl: DataLoader = None,
                 test_dl: DataLoader = None,
                 device: str = 'cuda:0',
                 num_classes: int = 10,
                 criterion: nn.Module = nn.CrossEntropyLoss(),
                 lr: float = 0.001,
                 num_epoch: int = 5,
                 optims: str = 'sgd',
                 init: bool = True,
                 resume: str = False,
                 compile: bool = False,
                 save_dir: str = 'logs', **kwargs):
        super().__init__()

        def init_xavier(m):  # 参数初始化
            # if type(m) == nn.Linear or type(m) == nn.Conv2d:
            if type(m) == nn.Linear:
                nn.init.xavier_normal_(m.weight)

        if init:
            model.apply(init_xavier)

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.model = model
        self.resume = resume
        self.model_name = None
        self.num_classes = num_classes
        self.save_dir = save_dir
        self.best_model_wts = None
        self.compile = compile

        self.train_dataloader = train_dl
        self.val_dataloader = val_dl
        self.test_dataloader = test_dl

        # 定义损失函数
        # CrossEntropyLoss 是用于多分类问题的损失函数，特别适用于分类任务。
        self.criterion = criterion
        self.epochs = num_epoch
        self.optims = optims
        self.optimizer = None
        self.lr = lr
        self.lr_scheduler = None
        # self.optim_scheduler()

        # 用于记录训练和验证过程中的损失值
        self.train_loss_all = []
        self.val_loss_all = []
        self.train_acc_all = []
        self.val_acc_all = []
        self.cnf_matrix = None  # 分类问题

        self.init_settings()

    def init_settings(self):
        """
        加载之前训练的模型（如有指定恢复路径）。
        初始化输出目录，用于存储训练日志和模型文件。
        设置日志记录系统，用于记录训练过程中的关键信息。
        初始化度量工具Metrics，用于评估模型在验证集上的表现。
        """
        if self.resume:
            self.model.load_state_dict(torch.load(self.resume, weights_only=True))
            self.model_name = os.path.basename(self.resume)
        self.model = self.model.to(self.device)
        # 保存模型的最佳权重, 使用 copy.deepcopy 复制模型的当前权重，以便在后面保存最优模型时使用。
        self.best_model_wts = copy.deepcopy(self.model.state_dict())

        # logger.info('init output dirs ... ')
        os.makedirs(os.path.join(self.save_dir, datetime.datetime.now().strftime("%Y%m%d_%H%M%S")),
                    exist_ok=True)

        # logger.info('init loggings ... ')
        # setup_logging(self.save_dir)

        # logger.info('init Metrics ... ')
        self.metrics = Metrics(self.num_classes, self.device)

        # logger.info('init optimizer ... ')
        self.optim_scheduler()
        # 初始化混淆矩阵
        self.cnf_matrix = np.zeros((self.num_classes, self.num_classes))

        if self.compile:
            # 定义使用的loss和optimizer，这里支持自定义
            self.model.compile(
                loss=nn.CrossEntropyLoss(),
                optimizer=optim.Adam(self.model.parameters(), lr=2e-5),
                scheduler=None,
                # metrics=['accuracy']
            )

    def optim_scheduler(self):
        """
        定义优化器
        """
        if self.optims == 'sgd':
            optimizer = torch.optim.SGD((param for param in self.model.parameters() if param.requires_grad),
                                        lr=self.lr,
                                        weight_decay=0)
        elif self.optims == 'adam':
            optimizer = torch.optim.Adam((param for param in self.model.parameters() if param.requires_grad),
                                         lr=self.lr,
                                         weight_decay=0)
        elif self.optims == 'adamW':
            optimizer = torch.optim.AdamW((param for param in self.model.parameters() if param.requires_grad),
                                          lr=self.lr,
                                          weight_decay=0)
        else:
            optimizer = torch.optim.SGD((param for param in self.model.parameters() if param.requires_grad),
                                        lr=self.lr,
                                        weight_decay=0)
        self.optimizer = optimizer
        self.lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    def fit(self):
        # 初始化变量以跟踪最佳准确率
        best_acc = 0.0
        best_ckpt = None
        # 训练数据的处理、损失计算、反向传播、参数更新
        for epoch in range(self.epochs):
            self.train_step(epoch)
            self.val_step(epoch)
            if self.val_acc_all[-1] > best_acc:  # 如果当前epoch的验证准确率大于历史最佳准确率
                best_acc = self.val_acc_all[-1]  # 更新最佳验证准确率
                # 深拷贝当前模型的参数权重，保存最佳模型
                self.best_model_wts = copy.deepcopy(self.model.state_dict())
                # 加载最佳模型权重，将模型参数恢复为训练期间保存的最佳权重
                self.model.load_state_dict(self.best_model_wts)

                # 保存最佳模型的权重到指定路径
                # 注意：torch.save() 的参数应该是模型的 `state_dict()`，而不是 `load_state_dict()` 的返回值
                try:
                    if best_ckpt is not None and os.path.exists(best_ckpt):
                        os.remove(best_ckpt)
                except Exception as e:
                    print(f"删除旧模型时发生错误：{e}")

                best_ckpt = os.path.join(self.save_dir,
                                         f'epoch_{epoch}_valacc_{best_acc:.3}.pth')
                try:
                    torch.save(self.model.state_dict(), best_ckpt)
                    print(f"已保存最佳模型到 {best_ckpt}")
                except Exception as e:
                    print(f"保存模型时发生错误：{e}")

    def train_step(self, epoch):
        # 初始化每个epoch的累积损失和正确预测数
        train_loss = 0.0  # 记录训练损失
        total_correct = 0.0
        # 初始化样本数量
        total_samples = 0
        self.model.train()
        train_count = len(self.train_dataloader)
        try:
            for i, batch in tqdm(enumerate(self.train_dataloader), total=train_count,
                                 desc=f'Train {epoch}/{self.epochs}'):
                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)
                bs = labels.size(0)

                # 反向传播前将梯度清零
                self.optimizer.zero_grad()
                # 前向传播
                outputs = self.model(images)
                # 计算损失
                loss = self.criterion(outputs, labels)
                # 反向传播，计算梯度
                loss.backward()
                # 更新模型参数
                self.optimizer.step()
                # 累加本批次的损失
                train_loss += loss.item() * bs

                # 计算当前批次的准确率
                preds = torch.argmax(outputs, dim=1)
                correct = torch.sum(preds == labels.data)
                total_correct += correct.item()
                # 累加训练样本总数
                total_samples += bs

            # 学习率调整
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            # 平均训练损失：总损失除以样本总数
            train_loss /= total_samples
            # 平均训练准确率：正确预测数除以样本总数
            train_acc = total_correct / total_samples
            self.train_loss_all.append(train_loss)
            self.train_acc_all.append(train_acc)
            # 打印当前 epoch 的训练损失和准确率，打印列表的最后一个值（即当前epoch的结果）
            print(f'Epoch {epoch}/{self.epochs} Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f}')
        except Exception as e:
            print(f"An error occurred during training: {e}")
            # logging.error(f"An error occurred during training: {e}", exc_info=True)

    def val_step(self, epoch):
        val_loss = 0.0
        total_correct = 0
        total_samples = 0

        if len(self.val_dataloader) == 0:
            print("Warning: Validation dataloader is empty.")
            return
        val_count = len(self.val_dataloader)

        try:
            self.model.eval()
            with torch.no_grad():
                for step, (b_x, b_y) in tqdm(enumerate(self.val_dataloader), total=val_count,
                                             desc=f'Val {epoch}/{self.epochs}'):
                    b_x = b_x.to(self.device)
                    b_y = b_y.to(self.device)
                    bs = b_y.size(0)

                    # 前向传播，获取模型对验证集数据的预测结果
                    outputs = self.model(b_x)
                    # 计算损失，衡量模型预测与真实标签的差异
                    loss = self.criterion(outputs, b_y)
                    # 累加本批次的损失值，并乘以当前批次的样本数，便于后续计算平均损失
                    val_loss += loss.item() * bs

                    # 获取每个样本预测的最大值对应的类别标签
                    preds = torch.argmax(outputs, dim=1)
                    # 计算本批次预测正确的样本数，并累加
                    correct = torch.sum(preds == b_y.data)
                    total_correct += correct.item()
                    total_samples += bs

                    # 更新混淆矩阵数据
                    if len(b_y.shape) == 1:  # 分类问题
                        print("混淆矩阵")
                        for idx in range(len(b_y)):
                            self.cnf_matrix[b_y[idx]][preds[idx]] += 1

                # 计算并保存验证集的平均损失和准确率
                val_loss /= total_samples
                val_acc = total_correct / total_samples
                self.val_loss_all.append(val_loss)  # 平均验证损失：总损失除以验证集样本总数
                self.val_acc_all.append(val_acc)  # 平均验证准确率：正确预测数除以验证集样本总数

                # 打印当前 epoch 的验证损失和准确率
                print(f'Epoch {epoch} Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}')

        except Exception as e:
            self.model.train()
            logger.error(f"Error during validation: {e}")
            raise e
        finally:
            self.model.train()

    def plot(self):
        # 使用pandas将训练过程的损失和准确率保存到DataFrame
        # 方便后续分析和可视化
        train_process = pd.DataFrame(
            data={'epoch': range(self.epochs),  # epoch的序号
                  'train_loss_all': self.train_loss_all,  # 每个epoch的训练损失
                  "val_loss_all": self.val_loss_all,  # 每个epoch的验证损失
                  "train_acc_all": self.train_acc_all,  # 每个epoch的训练准确率
                  "val_acc_all": self.val_acc_all,  # 每个epoch的验证准确率
                  }
        )


class Trainer(nn.Module):
    """
    __init__ : 输入并定义dataloader、model、optimizer、loss function、lr scheduler
    init_settings : 创建日志模型保存目录、初始化评价指标类
    train : 模型训练
    evaluate : 模型验证
    save_model : 保存模型参数
    load_model : 加载模型参数
    """

    def __init__(self,
                 model: nn.Module=None,
                 num_classes: int = 2,
                 train_dataloader: DataLoader = None,
                 val_dataloader: DataLoader = None,
                 test_dataloader: DataLoader = None,
                 optims: str = 'sgd',
                 criterion=nn.CrossEntropyLoss(),
                 # scheduler:None = None,
                 lr: float = 0.001,
                 epochs: int = 5,
                 device='cuda',
                 resume=None,
                 output_dir='./output',
                 cls=True,
                 compile = False, **kwargs):
        """
        创建 Trainer 对象时，完成训练和验证过程的所有必要初始化工作
        model : 传入待训练的深度学习模型，本文暂时使用unet。
        train_loader 和 val_loader : 分别是训练集和验证集的数据加载器，用于批量加载数据。
        criterion : 损失函数，用于计算模型预测输出与真实标签之间的误差。
        optimizer : 优化器，用于在训练过程中调整模型的参数，目标是最小化损失。
        scheduler : 学习率调度器，用于动态调整训练过程中的学习率，提升模型收敛效果。
        device : 指定使用的设备（CPU或GPU），用于加速计算。
        num_classes : 任务中的类别数，用于度量计算，比如混淆矩阵。
        resume : 模型恢复路径，用于从先前保存的检查点加载模型参数（如有需要）。
        output_dir : 输出目录，用于存储训练日志、模型检查点等。
        """
        super().__init__(**kwargs)

        self.device = device
        self.save_dir = os.path.join(output_dir, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))

        self.model = model.to(self.device)
        self.resume = resume
        self.model_name = None
        self.num_classes = num_classes

        self.criterion = criterion
        self.optim = optims
        self.lr = lr
        self.epochs = epochs
        self.metrics = None
        self.optimizer = None
        self.lr_scheduler = None

        self.train_loader = train_dataloader
        self.val_loader = val_dataloader
        if test_dataloader:
            self.test_loader = test_dataloader
        else:
            self.test_loader = val_dataloader

        # 用于记录训练和验证过程中的损失值
        self.train_loss_all = []
        self.val_loss_all = []
        self.train_acc_all = []
        self.val_acc_all = []
        self.cnf_matrix = None  # 分类问题
        self.cls = cls
        self.writer = None
        self.compile = compile

        self.init_settings()

    def init_settings(self):
        """
        加载之前训练的模型（如有指定恢复路径）。
        初始化输出目录，用于存储训练日志和模型文件。
        设置日志记录系统，用于记录训练过程中的关键信息。
        初始化度量工具Metrics，用于评估模型在验证集上的表现。
        """

        logger.info('init device ... ')
        try:
            self.device = torch.device(self.device)
        except Exception as e:
            print(e)
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

        if self.resume:
            self.load_model(self.resume)
            self.model_name = os.path.basename(self.resume)

        logger.info('init output dirs ... ')
        os.makedirs(self.save_dir, exist_ok=True)

        logger.info('init loggings ... ')
        setup_logging(self.save_dir)

        logger.info('init Metrics ... ')
        self.metrics = Metrics(self.num_classes, self.device)
        # 初始化混淆矩阵
        self.cnf_matrix = np.zeros((self.num_classes, self.num_classes))

        logger.info('init optimizer ... ')
        self.optim_scheduler()

        logger.info('model compile... ')
        if self.compile:
            # 定义使用的loss和optimizer，这里支持自定义
            self.model.compile(
                loss=self.criterion,
                optimizer=self.optimizer,
                scheduler=self.lr_scheduler, # None
                # metrics=['accuracy']
            )

    def optim_scheduler(self):
        """
        定义优化器
        """
        if self.optim == 'sgd':
            optimizer = optim.SGD((param for param in self.model.parameters() if param.requires_grad),
                                  lr=self.lr,
                                  weight_decay=0)
        elif self.optim == 'adam':
            optimizer = optim.Adam((param for param in self.model.parameters() if param.requires_grad),
                                   lr=self.lr,
                                   weight_decay=0)
        elif self.optim == 'adamW':
            optimizer = optim.AdamW((param for param in self.model.parameters() if param.requires_grad),
                                    lr=self.lr,
                                    weight_decay=0)
        else:
            optimizer = optim.SGD((param for param in self.model.parameters() if param.requires_grad),
                                  lr=self.lr,
                                  weight_decay=0)
        self.optimizer = optimizer
        self.lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    def fit(self):
        """
        通过指定的 epoch 数，逐步优化模型，保存最佳表现的模型。
        训练循环: 对于每个 epoch：
                加载数据: 从 train_loader 中加载一个批次的图像和对应的标签，将其移动到指定设备上（如 GPU）。
                清零梯度: 在每次反向传播之前，调用 self.optimizer.zero_grad() 清除上一次计算的梯度，避免累积。
                前向传播: 将输入图像通过模型，得到预测输出。
                计算损失: 使用损失函数 criterion 计算预测输出和真实标签之间的误差。
                反向传播: 调用 loss.backward() 计算损失对模型参数的梯度。
                优化模型参数: 调用 self.optimizer.step() 更新模型参数，最小化损失。
                累积损失: 记录当前批次的损失值，以便后续计算平均训练损失。
                设置模型为训练模式: 调用 self.model.train()，确保模型在训练过程中正确处理 dropout 和 batch normalization 等操作。
                批次循环: 对每个数据批次：
                调整学习率: 在每个 epoch 完成后，调用 self.scheduler.step() 依据预设的策略调整学习率。
                计算平均训练损失: 通过累积的损失计算该 epoch 的平均训练损失，并记录。
                验证模型: 调用 evaluate 方法，使用验证集评估模型的表现。
                保存模型: 如果当前 epoch 的验证精度超过历史最佳精度，则保存该 epoch 的模型，并更新最佳精度记录。
        """

        best_val_acc = 0.0
        train_count = len(self.train_loader)
        for epoch in range(self.epochs):
            logger.info(f"Epoch {epoch + 1}/{self.epochs}")
            self.model.train()
            train_loss = 0.0
            # train_acc = 0.0
            total_correct = 0
            total_samples = 0

            for batch_idx, batch in tqdm(enumerate(self.train_loader), total=train_count, desc='Training'):
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                bs = targets.size(0)

                # 清零梯度
                self.optimizer.zero_grad()
                # 前向传播
                outputs = self.model(inputs)
                # 计算损失
                loss = self.criterion(outputs, targets)
                # 反向传播
                loss.backward()
                # 更新参数
                self.optimizer.step()
                train_loss += loss.item() * bs

                # 获取每个样本预测的最大值对应的类别标签
                preds = torch.argmax(outputs, dim=1)
                # 更新混淆矩阵
                self.metrics.sample_add(targets, preds)

                # # 计算本批次预测正确的样本数，并累加
                # correct = torch.sum(preds == target.data)
                # total_correct += correct.item()
                total_samples += bs

            self.lr_scheduler.step()

            train_loss /= total_samples
            self.train_loss_all.append(train_loss)
            # train_acc = total_correct / total_samples
            _, train_acc = self.metrics.acc()
            self.train_acc_all.append(train_acc.cpu())
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Train Acc: {train_acc:.4f}")

            evaluate_results = self.evaluate()

            val_acc = evaluate_results['total_acc']
            # val_miou = evaluate_results['total_iou']
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model(f"epoch_{epoch + 1}_valacc_{val_acc:.4f}.pth")
        self.test()
        self.plot_acc_loss(save_path=os.path.join(self.save_dir, 'acc_loss.png'))
        # if self.cls:
        #     self.plot_confusion_matrix(cm =self.cnf_matrix,
        #                                classes= [i for i in range(self.num_classes)],)

    def evaluate(self):
        """
        评估模型在验证集上的性能，以确保模型不仅在训练集上表现良好，也能泛化到未见过的数据
        评估循环: 对验证集中的每个数据批次：
            加载数据: 从 val_loader 中加载一个批次的图像和对应的标签，将其移动到指定设备上（如 GPU）。
            前向传播: 将输入图像通过模型，得到预测输出。
            计算预测: 使用 torch.max 获取每个像素/样本的预测类别。
            更新度量工具: 调用 self.metrics.sample_add(labels, preds) 将当前批次的预测结果与真实标签传递给度量工具，更新混淆矩阵或其他度量信息。
            计算损失: 使用损失函数 criterion 计算预测输出和真实标签之间的误差，并累积损失值。
        计算平均验证损失: 通过累积的损失计算验证集的平均损失，并记录。
        计算并记录评估指标: 调用 self.metrics.compute() 计算如准确率、交并比等关键指标，并将其记录到日志中。
        返回评估结果: 返回包含所有评估指标的字典，便于进一步处理。
        """
        # 在验证过程中不执行 dropout 和 batch normalization 的更新
        self.model.eval()

        self.metrics.update()
        val_loss = 0.0
        # val_acc = 0.0
        total_correct = 0
        total_samples = 0

        val_count = len(self.val_loader)
        with torch.no_grad():
            # logger.info("start evaluating ...")
            for batch in tqdm(self.val_loader, total=val_count, desc='Evaluating'):
                # images, labels = batch
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                bs = targets.size(0)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item() * bs

                # 获取每个样本预测的最大值对应的类别标签
                # _, preds = torch.max(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                # 更新混淆矩阵
                self.metrics.sample_add(targets, preds)

                # # 计算本批次预测正确的样本数，并累加
                # correct = torch.sum(preds == targets.data)
                # total_correct += correct.item()
                total_samples += bs

        val_loss /= total_samples
        self.val_loss_all.append(val_loss)
        # val_acc = total_correct / total_samples
        _, val_acc = self.metrics.acc()
        self.val_acc_all.append(val_acc.cpu())
        logger.info(f"Validation Loss: {val_loss:.4f}")
        logger.info(f"Validation Acc: {val_acc:.4f}")

        # 计算并记录指标
        results = self.metrics.compute()
        logger.info("Validation Metrics:")
        for key, value in results.items():
            logger.info(f"{key}: {value}")

        return results

    def test(self):
        """
        测试模型。
        """
        self.model.eval()
        # test_loss_all = []
        test_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                bs = targets.size(0)

                # 前向传播
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                # test_loss_all.append(loss.item())
                test_loss += loss.item() * bs
                total_samples += bs

                targets = targets.cpu().numpy()
                outputs = torch.argmax(outputs, dim=1).cpu().numpy()
                # 更新混淆矩阵数据
                if self.cls:  # 分类问题
                    for idx in range(len(targets)):
                        self.cnf_matrix[targets[idx]][outputs[idx]] += 1

        test_loss /= total_samples
        logger.info(f"Final Test Loss: {test_loss:.4f}")

    def predict(self, im_data):
        """
        目标: 使用训练好的模型对给定的图像进行预测。
        加载模型参数: 从指定的检查点文件中加载模型的参数，并将其移动到指定的设备上。
        加载图像: 使用 cv2.imread() 函数加载图像，并将其转换为 PyTorch 张量。
        预处理图像: 将图像转换为模型所需的输入格式，例如归一化、裁剪等。
        前向传播: 将预处理后的图像输入模型，得到预测输出。
        获取预测结果: 从预测输出中获取最大值对应的类别标签。
        返回预测结果: 返回预测结果，即预测的类别标签。
        """
        im_data = torch.tensor(im_data).unsqueeze(0).to(self.device)
        # im_data = torch.tensor(Image.open(img_path))
        self.model.eval()
        with torch.no_grad():
            pred = self.model(im_data)
            pred = torch.argmax(pred, dim=1)
            return pred

    def save_model(self, filename):
        """
        目标: 在验证集上取得最佳表现时，将当前的模型参数保存为一个检查点文件。
        移除旧模型文件: 如果当前已经存在一个模型文件，会先删除旧的模型文件。
        保存新模型: 调用 torch.save(self.model.state_dict(), model_path) 将当前模型的参数保存到指定路径，并更新记录的模型文件名。
        记录模型保存: 记录模型保存的路径和文件名，便于后续恢复或推理使用。
        """
        if self.model_name and os.path.exists(os.path.join(self.save_dir, self.model_name)):
            os.remove(os.path.join(self.save_dir, self.model_name))
        model_path = os.path.join(self.save_dir, filename)
        torch.save(self.model.state_dict(), model_path)
        self.model_name = filename
        logger.info(f"Model saved to {model_path}")

    def load_model(self, checkpoint):
        """
        目标: 从指定的检查点文件加载模型参数，用于恢复训练或进行推理。
        加载模型参数: 调用 torch.load(checkpoint) 从检查点文件中加载模型的状态字典，并将其加载到当前模型中。
        设置设备: 将加载后的模型移动到指定的设备上（如 GPU），以便进行后续的训练或验证。
        记录模型加载: 记录模型加载的路径，便于追踪和调试。
        """
        self.model.load_state_dict(torch.load(checkpoint, weights_only=True))
        self.model.to(self.device)
        logger.info(f"Model loaded from {checkpoint}")

    def plot_acc_loss(self, save_path=None):
        # 检查数据有效性
        if not (self.train_loss_all and self.val_loss_all and self.train_acc_all and self.val_acc_all):
            raise ValueError("One or more of the data lists is empty or None.")

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(self.train_loss_all, 'ro-', label='Train Loss')
        plt.plot(self.val_loss_all, 'bs-', label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Train and Validation Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(self.train_acc_all, 'ro-', label='Train Acc')
        plt.plot(self.val_acc_all, 'bs-', label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Train and Validation Accuracy')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
        plt.close()

    def plot_confusion_matrix(self, cm, classes,
                              normalize=False,
                              title='Confusion matrix'):
        """
        绘制混淆矩阵。

        :param cm: 混淆矩阵
        :param classes: 类别标签列表
        :param normalize: 是否归一化混淆矩阵，默认为 False
        :param title: 图表标题
        :param cmap: 颜色映射，默认为 Blues
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        plt.imshow(cm, cmap='viridis')  # , interpolation='nearest'
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        # fmt = '.2f' if normalize else 'd'
        # thresh = cm.max() / 2.
        # for i in range(cm.shape[0]):
        #     for j in range(cm.shape[1]):
        #         plt.text(j, i, format(cm[i, j], fmt),
        #                  horizontalalignment="center",
        #                  color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
        plt.close()


# if __name__ == '__main__':
#
#     device_flag = 'cuda:0'
#     # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#     FashionMNIST_dir = './data'
#
#     # 超参数
#     batch_size = 32
#     # 对于Windows用户，这里应设置为0，否则会出现多线程错误
#     if sys.platform.startswith('win'):
#         num_workers = 0
#     else:
#         num_workers = 4
#     lr = 1e-4
#     epochs = 3
#
#     LeNet = LeNetV1()
#
#     # 加载并处理训练和验证数据
#     train_dl, val_dl, test_dl = FashionMNIST_loader(root=FashionMNIST_dir, batch_size=batch_size,
#                                                     num_workers=num_workers)
#
#     x, y = next(iter(test_dl))
#     print(len(y.shape))
#     print(x.shape)
#
#     # criterion = nn.CrossEntropyLoss() # torch.nn.modules.loss.CrossEntropyLoss
#
#     aa = TrainerV1(LeNet,
#                    num_classes=10,
#                    train_dataloader=train_dl,
#                    val_dataloader=val_dl,
#                    test_dataloader=test_dl,
#                    device=device_flag,
#                    num_epoch=epochs,
#                    optims='adam',
#                    resume='output/20240930_105131/epoch_3_acc_0.7721.pth',
#                    cls=True)
#     # aa.fit()
#
#     res = aa.predict(x[0, :, :, :])
#     print(res, y[0])
