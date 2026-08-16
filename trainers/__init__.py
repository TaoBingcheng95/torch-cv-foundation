
"""trainers."""

from .basetrainer import BaseTrainer
from .dpp_trainer import DDPTrainer
from .visualizer import Visualizer, TrainingVisualizer
from .report import print_test_report, fmt_value
from .utils import TrainConfig
from .tb_logger import TensorBoardLogger, TBLogger

# from .logger_utils import setup_logging

__all__ = (
    'BaseTrainer',
    'DDPTrainer',
    'Visualizer',
    'TrainingVisualizer',  # 向后兼容别名
    'print_test_report',
    'fmt_value',
    'TensorBoardLogger',
    'TBLogger',  # 向后兼容别名
    'TrainConfig',
)
