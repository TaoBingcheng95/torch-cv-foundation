# https://blog.itpub.net/18841117/viewspace-3015295/

import os
from pathlib import Path
import datetime
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

from metrics import Metrics
from .logger_utils import setup_logging



class BaseTrainer(nn.Module):
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
                 device='cuda',
                 resume=None,
                 num_classes: int = 2,
                 train_dataloader: DataLoader = None,
                 val_dataloader: DataLoader = None,
                 test_dataloader: DataLoader = None,
                 optimizer_type: str = 'adam',
                 scheduler_type: str = 'steplr',
                 lr: float = 0.001, # learning_rate
                 step_size:int=10,
                 gamma:float=0.1,
                 criterion=nn.CrossEntropyLoss(),
                 epochs: int = 5,
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
        self.save_dir = Path(os.path.join(output_dir, datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))
        self.logger = None

        self.model = model
        self.resume = resume
        self.model_name = None
        self.num_classes = num_classes

        self.optimizer_type:str = optimizer_type
        self.scheduler_type:str = scheduler_type
        self.lr:float = lr
        self.step_size:int=step_size
        self.gamma:float=gamma
        self.optimizer= None # optim.Optimizer
        self.scheduler = None # lr_scheduler.LRScheduler

        self.criterion = criterion

        self.epochs = epochs
        self.metrics = None

        self.train_loader = train_dataloader
        self.val_loader = val_dataloader
        if test_dataloader:
            self.test_loader = test_dataloader
        else:
            self.test_loader = val_dataloader
        self.train_count = len(self.train_loader)
        self.val_count = len(self.val_loader)
        self.test_count = len(self.test_loader)

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

        self.logger = setup_logging(self.save_dir)
        self.logger.info('init loggings ... ')

        self.logger.info('init device ... ')
        self.device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.logger.info('init optimizer ... ')
        self.init_optim_scheduler()

        if self.resume:
            self.logger.info(f'resume from {self.resume} ')
            self.load_model(self.resume)
            self.model_name = os.path.basename(self.resume)

        self.logger.info('init output dirs ... ')
        os.makedirs(self.save_dir, exist_ok=True)

        self.logger.info('init metrics ... ')
        self.metrics = Metrics(self.num_classes, self.device)
        # 初始化混淆矩阵
        self.cnf_matrix = np.zeros((self.num_classes, self.num_classes))

        if self.compile:
            pass
            # self.logger.info('model compile... ')
            # # # 定义使用的loss和optimizer，这里支持自定义
            # # self.model.compile(
            # #     # loss=self.criterion,
            # #     optimizer=self.optimizer,
            # #     scheduler=self.scheduler, # None
            # #     # metrics=['accuracy']
            # # )
            # try:
            #     self.model = torch.compile(self.model)
            # except RuntimeError as e:
            #     self.logger.error(f"Model compilation failed: {e}")

    def init_optim_scheduler(self):
        # Initialize optimizer
        if self.optimizer_type.lower() == 'adam':
            # self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            self.optimizer = optim.Adam((param for param in self.model.parameters() if param.requires_grad),
                                        lr=self.lr,
                                        weight_decay=0)
        elif self.optimizer_type.lower() == 'sgd':
            self.optimizer = optim.SGD((param for param in self.model.parameters() if param.requires_grad),
                                       lr=self.lr, # momentum = 0.9,
                                       weight_decay=0)
        elif self.optimizer_type.lower() == 'adamW':
            self.optimizer = optim.AdamW((param for param in self.model.parameters() if param.requires_grad),
                                         lr=self.lr,
                                         weight_decay=0)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")

        # Initialize scheduler
        if self.scheduler_type.lower() == 'steplr':
            self.scheduler = lr_scheduler.StepLR(self.optimizer,
                                                 step_size=self.step_size,
                                                 gamma=self.gamma)
        elif self.scheduler_type.lower() == 'exponentiallr':
            self.scheduler = lr_scheduler.ExponentialLR(self.optimizer,
                                                        gamma=self.gamma)
        else:
            raise ValueError(f"Unsupported scheduler type: {self.scheduler_type}")


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
            self.logger.info(f"Epoch {epoch + 1}/{self.epochs}")
            # self.model.train()

            # train_loss = 0.0
            # for batch_idx, batch in tqdm(enumerate(self.train_loader), total=train_count, desc='Training'):
            #     inputs, targets = batch
            #     inputs = inputs.to(self.device)
            #     targets = targets.to(self.device)
            #
            #     self.optimizer.zero_grad() # 清零梯度
            #     outputs = self.model(inputs) # 前向传播
            #     loss = self.criterion(outputs, targets) # 计算损失
            #     loss.backward()# 反向传播
            #     self.optimizer.step()# 更新参数
            #
            #     train_loss += loss.item() * targets.size(0)
            #
            #     preds = torch.argmax(outputs, dim=1).detach() # 获取每个样本预测的最大值对应的类别标签
            #     self.metrics.sample_add(targets, preds) # 更新混淆矩阵
            #     # total_samples += targets.size(0)
            # self.scheduler.step()
            # train_loss /= len(self.train_loader.dataset) # total_samples
            # self.train_loss_all.append(train_loss)
            # _, train_acc = self.metrics.acc()
            # self.train_acc_all.append(train_acc.cpu())
            # self.logger.info(f"Train Loss: {train_loss:.4f}")
            # self.logger.info(f"Train Acc: {train_acc:.4f}")

            train_results = self.train_step()
            self.logger.info(f"Train Loss: {train_results['loss']:.4f}")
            self.logger.info(f"Train Acc: {train_results['acc']:.4f}")

            # evaluate_results = self.evaluate()
            val_results = self.evaluate()
            self.logger.info(f"Validation Loss: {val_results['loss']:.4f}")
            self.logger.info(f"Validation Acc: {val_results['acc']:.4f}")

            # # 定期测试
            # if (epoch + 1) % 5 == 0:  # 每5个epoch进行一次测试
            #     test_results = self.test()
            #     self.logger.info(f"Test Loss: {test_results['loss']:.4f}")
            #     self.logger.info(f"Test Acc: {test_results['acc']:.4f}")

            val_acc = val_results['acc']
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                checkpoint = {
                    'val_acc': best_val_acc,
                    'epoch': epoch,
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'lr_schedule': self.scheduler.state_dict()}
                self.save_model(f"epoch_{epoch + 1}_valacc_{val_acc:.4f}.pth",
                                checkpoint=checkpoint)

        test_results = self.test()
        self.logger.info(f"Test Loss: {test_results['loss']:.4f}")
        self.logger.info(f"Test Acc: {test_results['acc']:.4f}")

        self.plot_acc_loss(save_path=os.path.join(self.save_dir, 'acc_loss.png'))
        # if self.cls:
        #     self.plot_confusion_matrix(cm =self.cnf_matrix,
        #                                classes= [i for i in range(self.num_classes)],)

    def train_step(self):
        total_loss = 0.0
        self.model.train()
        self.metrics.update()
        self.logger.info("start training ...")
        for batch in tqdm(self.train_loader, total=self.train_count, desc='Training'):
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * targets.size(0)
            preds = torch.argmax(outputs, dim=1).detach()
            self.metrics.sample_add(targets, preds)

        self.scheduler.step()
        avg_loss = total_loss / len(self.train_loader.dataset)
        results = self.metrics.compute()
        train_acc = results['total_acc']
        self.train_loss_all.append(avg_loss)
        self.train_acc_all.append(train_acc)

        return {'loss': avg_loss, 'acc': train_acc}

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
        total_loss = 0.0
        self.model.eval()
        self.metrics.update()
        with torch.no_grad():
            self.logger.info("start evaluating ...")
            for batch in tqdm(self.val_loader, total=self.val_count, desc='Evaluating'):
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item() * targets.size(0)

                preds = torch.argmax(outputs, dim=1).detach() # 获取每个样本预测的最大值对应的类别标签
                self.metrics.sample_add(targets, preds) # 更新混淆矩阵

        avg_loss = total_loss / len(self.train_loader.dataset) # total_samples
        # 计算并记录指标
        results = self.metrics.compute()
        self.logger.info("Validation metrics:")
        for key, value in results.items():
            self.logger.info(f"{key}: {value}")
        val_acc = results['total_acc']
        self.val_acc_all.append(val_acc)
        self.val_loss_all.append(avg_loss)
        return {'loss': avg_loss, 'acc': val_acc} # results

    def test(self):
        total_loss = 0.0
        self.model.eval()
        self.metrics.update()

        with torch.no_grad():
            for batch in tqdm(self.test_loader, total=self.test_count, desc='Testing'):
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item() * targets.size(0)

                preds = torch.argmax(outputs, dim=1).detach()#.cpu().numpy()
                # # 更新混淆矩阵数据
                # if self.cls:  # 分类问题
                #     for idx in range(len(targets)):
                #         self.cnf_matrix[targets[idx]][outputs[idx]] += 1
                # 更新混淆矩阵
                self.metrics.sample_add(targets, preds)

        avg_loss = total_loss / len(self.test_loader.dataset) #total_samples
        _, test_acc = self.metrics.acc()
        # self.logger.info(f"Final Test Loss: {test_loss:.4f}")
        # self.logger.info(f"Final Test Acc: {test_acc:.4f}")
        # self.val_loss_all.append(test_loss)
        # # val_acc = total_correct / total_samples

        # self.val_acc_all.append(test_acc.cpu())
        # self.logger.info(f"Test Loss: {test_loss:.4f}")
        # self.logger.info(f"Test Acc: {test_acc:.4f}")
        #
        # # 计算并记录指标
        # results = self.metrics.compute()
        # self.logger.info("Test metrics:")
        # for key, value in results.items():
        #     self.logger.info(f"{key}: {value}")
        return {'loss': avg_loss, 'acc': test_acc}

    @torch.no_grad()
    def predict(self, inputs):
        """
        目标: 使用训练好的模型对给定的图像进行预测。
        加载模型参数: 从指定的检查点文件中加载模型的参数，并将其移动到指定的设备上。
        加载图像: 使用 cv2.imread() 函数加载图像，并将其转换为 PyTorch 张量。
        预处理图像: 将图像转换为模型所需的输入格式，例如归一化、裁剪等。
        前向传播: 将预处理后的图像输入模型，得到预测输出。
        获取预测结果: 从预测输出中获取最大值对应的类别标签。
        返回预测结果: 返回预测结果，即预测的类别标签。
        """
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.tensor(inputs)
        inputs = inputs.to(self.device)
        self.model.eval()
        pred = self.model(inputs)
        pred = torch.argmax(pred, dim=1)
        return pred

    def save_model(self, filename,checkpoint=None):
        """
        目标: 在验证集上取得最佳表现时，将当前的模型参数保存为一个检查点文件。
        移除旧模型文件: 如果当前已经存在一个模型文件，会先删除旧的模型文件。
        保存新模型: 调用 torch.save(self.model.state_dict(), model_path) 将当前模型的参数保存到指定路径，并更新记录的模型文件名。
        记录模型保存: 记录模型保存的路径和文件名，便于后续恢复或推理使用。
        """
        if self.model_name and os.path.exists(os.path.join(self.save_dir, self.model_name)):
            os.remove(os.path.join(self.save_dir, self.model_name))
        model_path = os.path.join(self.save_dir, filename)
        if checkpoint:
            torch.save(checkpoint, model_path)
        else:
            torch.save(self.model.state_dict(), model_path)
        self.model_name = filename
        self.logger.info(f"Model saved to {model_path}")

    def load_model(self, checkpoint):
        """
        目标: 从指定的检查点文件加载模型参数，用于恢复训练或进行推理。
        加载模型参数: 调用 torch.load(checkpoint) 从检查点文件中加载模型的状态字典，并将其加载到当前模型中。
        设置设备: 将加载后的模型移动到指定的设备上（如 GPU），以便进行后续的训练或验证。
        记录模型加载: 记录模型加载的路径，便于追踪和调试。
        """
        if not os.path.exists(checkpoint):
            self.logger.error(f"Loading model from {checkpoint}")
            raise FileNotFoundError(f"Checkpoint file {checkpoint} not found.")

        try:
            checkpoint = torch.load(checkpoint, weights_only=False)
            # start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['model'], strict=False)
            self.model.to(self.device)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['lr_schedule'])
        except Exception as e:
            self.logger.error(f"Warning: Error loading model from {checkpoint}: {e}")
            raise e
        # self.logger.info(f"Model loaded from {checkpoint}")

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

    @staticmethod
    def plot_confusion_matrix(cm, classes,
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
