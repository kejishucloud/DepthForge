from __future__ import annotations

import logging
from pathlib import Path


def setup_logger(log_path: Path) -> logging.Logger:
    """
    初始化文件与控制台日志器。
    :param log_path: 参数介绍
    :return: 返回介绍
    """
    logger = logging.getLogger("scanner")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger
