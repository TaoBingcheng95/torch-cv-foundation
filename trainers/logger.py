"""通用日志记录器

为本项目提供统一的日志入口，特性：
    - 控制台输出：时间戳 + 级别 + 消息，按级别着色（自动检测终端能力）
    - 文件日志：可选写入 UTF-8 日志文件，便于训练结束后排查
    - DDP 多进程感知：非主进程（RANK 不在 {-1, 0}）自动降级，避免多卡日志重复
    - 幂等/单例获取：get_logger 复用同名 logger，重复配置不会叠加 handler

典型用法：
    >>> from trainers.logger import LOGGER, get_logger, add_file_handler
    >>> LOGGER.info("start training")                 # 直接使用全局 logger
    >>> logger = get_logger("MyModule")               # 获取（或惰性创建）具名 logger
    >>> add_file_handler(logger, "output/train.log")  # 运行期确定目录后追加文件日志
"""

import os
import sys
import platform
import logging
from pathlib import Path


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
# PyTorch 多卡 DDP 环境变量，https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv("RANK", -1))
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))
# 平台布尔量
MACOS, LINUX, WINDOWS = (platform.system() == x for x in ("Darwin", "Linux", "Windows"))

LOGGING_NAME = "torch-cv"
# 全局详细模式，可用环境变量 CV_VERBOSE 覆盖
VERBOSE = str(os.getenv("CV_VERBOSE", True)).lower() == "true"

# 输出格式
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ANSI 颜色（按级别着色 levelname）
_COLORS = {
    "DEBUG": "\033[36m",      # 青色
    "INFO": "\033[32m",       # 绿色
    "WARNING": "\033[33m",    # 黄色
    "ERROR": "\033[31m",      # 红色
    "CRITICAL": "\033[1;31m",  # 加粗红
}
_RESET = "\033[0m"


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------
def _enable_windows_utf8() -> None:
    """在 Windows 上尝试把 stdout 切换为 UTF-8，以正确显示中文/emoji。"""
    if not (WINDOWS and hasattr(sys.stdout, "encoding") and sys.stdout.encoding != "utf-8"):
        return
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        elif hasattr(sys.stdout, "buffer"):
            import io

            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    except Exception:
        # 无法切换时静默降级，不影响主流程
        pass


def _supports_color(stream) -> bool:
    """判断输出流是否支持 ANSI 颜色（连接到交互式终端时才启用）。"""
    return hasattr(stream, "isatty") and stream.isatty()


class ColoredFormatter(logging.Formatter):
    """给 levelname 着色的格式化器。

    仅在格式化期间临时替换 record.levelname 并立即还原，不破坏原始 record，
    因此同一条日志被多个 handler 处理时不会相互污染。
    """

    def __init__(self, fmt=LOG_FORMAT, datefmt=DATE_FORMAT, use_color=True):
        super().__init__(fmt, datefmt)
        self.use_color = use_color

    def format(self, record):
        if not self.use_color or record.levelname not in _COLORS:
            return super().format(record)
        original = record.levelname
        record.levelname = f"{_COLORS[original]}{original}{_RESET}"
        try:
            return super().format(record)
        finally:
            record.levelname = original


# ---------------------------------------------------------------------------
# 公共接口
# ---------------------------------------------------------------------------
def set_logging(name=LOGGING_NAME, verbose=VERBOSE, log_file=None, use_color=None):
    """配置并返回一个 logger（幂等）。

    Args:
        name (str): logger 名称。
        verbose (bool): 为 True 且处于主进程时级别为 INFO，否则为 ERROR。
        log_file (str | os.PathLike | None): 可选日志文件路径，仅主进程写入。
        use_color (bool | None): 是否给控制台着色；None 表示自动检测终端。

    Returns:
        logging.Logger: 配置完成的 logger。
    """
    level = logging.INFO if verbose and RANK in {-1, 0} else logging.ERROR

    _enable_windows_utf8()

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    # 清空旧 handler，保证重复调用不会叠加输出
    logger.handlers.clear()

    # 控制台 handler
    if use_color is None:
        use_color = _supports_color(sys.stdout)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(ColoredFormatter(use_color=use_color))
    logger.addHandler(console)

    # 文件 handler（可选，仅主进程；文件内不着色）
    if log_file and RANK in {-1, 0}:
        add_file_handler(logger, log_file, level=level)

    return logger


def get_logger(name=LOGGING_NAME):
    """获取具名 logger；若尚未配置则惰性完成默认配置。"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        return set_logging(name)
    return logger


def add_file_handler(logger, log_file, level=None):
    """为已有 logger 追加文件 handler（UTF-8、无颜色）。

    适合训练场景：输出目录常在运行期才确定，可在创建目录后再挂载文件日志。
    仅主进程真正写文件；重复添加相同路径会被跳过。
    """
    if RANK not in {-1, 0}:
        return logger

    log_file = Path(log_file)
    # 去重：避免对同一文件重复添加 handler
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler) and Path(getattr(h, "baseFilename", "")) == log_file.resolve():
            return logger

    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level if level is not None else logger.level)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    logger.addHandler(file_handler)
    return logger


# 全局 logger，供各模块直接导入使用
LOGGER = set_logging(LOGGING_NAME, verbose=VERBOSE)
