from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple


def _resolve_level(level: Optional[str | int]) -> int:
    if level is None:
        return logging.INFO
    if isinstance(level, int):
        return level
    name = str(level).upper()
    return getattr(logging, name, logging.INFO)


def configure_app_logging(log_dir: Path, level: Optional[str | int] = None) -> Tuple[logging.Logger, Path]:
    """
    初始化应用日志（控制台 + 文件）。
    :param log_dir: 日志目录
    :param level: 日志级别
    :return: (logger, log_path)
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"app_{ts}.log"
    logger = setup_logger(log_path, level=level, name="scanner", clear_handlers=True)
    return logger, log_path


def setup_logger(
    log_path: Path,
    *,
    level: Optional[str | int] = None,
    name: str = "scanner",
    clear_handlers: bool = False,
) -> logging.Logger:
    """
    初始化文件与控制台日志器。
    :param log_path: 日志文件
    :param level: 日志级别
    :param name: logger 名称
    :param clear_handlers: 是否清空已有 handlers
    :return: logger
    """
    logger = logging.getLogger(name)
    resolved_level = _resolve_level(level)
    logger.setLevel(resolved_level)
    logger.propagate = False
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(threadName)s %(message)s")

    if clear_handlers:
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(resolved_level)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(resolved_level)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def install_qt_message_handler(logger: logging.Logger) -> None:
    """
    将 Qt 日志重定向到 Python logging。
    """
    try:
        from PySide6.QtCore import QtMsgType, qInstallMessageHandler
    except Exception:
        return

    def _handler(mode, context, message):  # type: ignore[override]
        if mode == QtMsgType.QtDebugMsg:
            logger.debug("Qt: %s", message)
        elif mode == QtMsgType.QtInfoMsg:
            logger.info("Qt: %s", message)
        elif mode == QtMsgType.QtWarningMsg:
            logger.warning("Qt: %s", message)
        elif mode == QtMsgType.QtCriticalMsg:
            logger.error("Qt: %s", message)
        elif mode == QtMsgType.QtFatalMsg:
            logger.critical("Qt: %s", message)
        else:
            logger.info("Qt: %s", message)

    qInstallMessageHandler(_handler)
