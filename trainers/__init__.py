# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""trainers."""

# from trainers.depredate.api import get_model, get_model_weights, get_weight, list_models

# from .base import BaseTask
# from .byol import BYOLTask
# from .classification import ClassificationTask, MultiLabelClassificationTask
# from .detection import ObjectDetectionTask
# from .iobench import IOBenchTask
# from .moco import MoCoTask
# from .regression import PixelwiseRegressionTask, RegressionTask
# from .segmentation import SemanticSegmentationTask
# from .simclr import SimCLRTask
from .basetrainer import BaseTrainer

from .logger_utils import setup_logging

__all__ = (
    # Supervised
    # 'ClassificationTask',
    # 'MultiLabelClassificationTask',
    # 'ObjectDetectionTask',
    # 'PixelwiseRegressionTask',
    # 'RegressionTask',
    # 'SemanticSegmentationTask',
    BaseTrainer,
    # Self-supervised
    # 'BYOLTask',
    # 'MoCoTask',
    # 'SimCLRTask',
    # Base classes
    # 'BaseTask',
    # Other
    # 'IOBenchTask',
    # utilities
    'setup_logging',
    # 'get_model',
    # 'get_model_weights',
    # 'get_weight',
    # 'list_models',
)
