import os
from loguru import logger
# import logging

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
    # logger.info(f"Logging is set up. Logs are being saved to {log_file_path}.")
    return logger
