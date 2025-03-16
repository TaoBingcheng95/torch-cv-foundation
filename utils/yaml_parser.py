import os
from typing import Any, Dict, List, Optional, Tuple
import yaml
import hydra
# import rootutils
# import torch
import omegaconf
from omegaconf import DictConfig
from pprint import pprint
import logging
import rootutils
rootutils.setup_root(__file__, indicator=".git", pythonpath=True) # indicator=".project-root"

# 配置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_yaml(yaml_file):
    with open(yaml_file, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
        for k, v in config.items():
            print(k, v)
    return config


def load_config(config_path: str, config_name: str) -> DictConfig:
    """
    加载配置文件。

    :param config_path: 配置文件路径。
    :param config_name: 配置文件名称。
    :return: 加载的配置对象。
    """
    with hydra.initialize(version_base="1.3", config_path=config_path):
        cfg = hydra.compose(config_name=config_name)
        return cfg

def main(cfg: DictConfig) -> Optional[float]:
    """主入口点，用于训练模型。

    :param cfg: 由 Hydra 生成的 DictConfig 配置对象。
    :return: 可选的浮点数，表示优化指标值。
    """
    try:
        # 应用额外的实用工具
        # 例如，如果 cfg 中没有提供标签，则询问用户输入标签，打印配置树等
        print(cfg)
        
        # 这里可以添加更多的训练逻辑
        # 例如，加载数据集、定义模型、训练模型等
        
        # 示例返回值
        return 0.95  # 假设优化指标值为 0.95
    except Exception as e:
        logger.error(f"发生错误: {e}")
        return None




if __name__ == '__main__':

    print(os.environ['PROJECT_ROOT'])

    config_path = "../configs"
    config_name = "train.yaml"
    # cfg = load_yaml(os.path.join(config_path, configs_fn))

    cfg = load_config(config_path, config_name)
    logger.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule = hydra.utils.instantiate(cfg.data)
    print(type(datamodule))
    