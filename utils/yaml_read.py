from typing import Any, Dict, List, Optional, Tuple
import yaml
import rootutils
import hydra
from pprint import pprint
# import lightning as L

import lightning.pytorch as L
import rootutils
import torch
from lightning.pytorch import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger

from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".git", pythonpath=True) # indicator=".project-root"

from utils import (
    RankedLogger,
    extras,
    get_metric_value,
    # instantiate_callbacks,
    # instantiate_loggers,
    # log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)



def load_yaml(config_file):
    """
    Load a YAML configuration file.

    :param config_file: Path to the YAML configuration file.
    :return: Parsed configuration as a dictionary.
    """
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config


def load_trainer_config(trainer_config_path):
    # 加载 trainer.yaml
    trainer_config = load_yaml(trainer_config_path)

    # 获取数据和模型配置文件路径
    data_config_path = trainer_config['data_config']
    model_config_path = trainer_config['model_config']

    # 加载数据和模型配置文件
    data_config = load_yaml(data_config_path)
    model_config = load_yaml(model_config_path)

    # 合并配置
    full_config = {
        'data': data_config,
        'model': model_config
    }

    return full_config


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    # if cfg.get("seed"):
    #     L.seed_everything(cfg.seed, workers=True)

    # log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    # log.info("Instantiating callbacks...")
    # callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    # log.info("Instantiating loggers...")
    # logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer) # , callbacks=callbacks, logger=logger

    # object_dict = {
    #     "cfg": cfg,
    #     "datamodule": datamodule,
    #     "model": model,
    #     "callbacks": callbacks,
    #     "logger": logger,
    #     "trainer": trainer,
    # }

    # if logger:
    #     log.info("Logging hyperparameters!")
    #     log_hyperparameters(object_dict)

    # if cfg.get("train"):
    #     log.info("Starting training!")
    #     trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    # train_metrics = trainer.callback_metrics

    # if cfg.get("test"):
    #     log.info("Starting testing!")
    #     ckpt_path = trainer.checkpoint_callback.best_model_path
    #     if ckpt_path == "":
    #         log.warning("Best ckpt not found! Using current weights for testing...")
    #         ckpt_path = None
    #     trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    #     log.info(f"Best ckpt path: {ckpt_path}")

    # test_metrics = trainer.callback_metrics

    # # merge train and test metrics
    # metric_dict = {**train_metrics, **test_metrics}

    return None, None # metric_dict, object_dict



@hydra.main(version_base="1.3", config_path="./configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    # extras(cfg)
    
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer) # , callbacks=callbacks, logger=logger
    
    # print(cfg.paths.root_dir)
    # print(cfg.paths.data_dir)
    # print(cfg.paths.log_dir)
    # print(cfg.paths.output_dir)
    # print(cfg.paths.work_dir)
    #print(cfg.data.data_dir)
    # pprint(cfg.model)

    # # train the model
    # metric_dict, _ = train(cfg)

    # # safely retrieve metric value for hydra-based hyperparameter optimization
    # metric_value = get_metric_value(
    #     metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    # )

    # return optimized metric
    return None # metric_value



# 示例使用
if __name__ == "__main__":

    main()

    # config_path = 'configs/train.yaml'  # 请将此路径替换为你的配置文件路径
    # config = load_yaml(config_path)
    # pprint(config)

    # tmp = hydra.utils.instantiate(config.data)
    # print(type(tmp))
