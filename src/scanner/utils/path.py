from __future__ import annotations

from pathlib import Path
from typing import Iterable


def ensure_dir(path: Path) -> Path:
    """
    确保目录存在。
    :param path: 目标目录路径
    :return: 创建后的目录路径
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_dirs(paths: Iterable[Path]) -> list[Path]:
    """
    从路径列表中过滤目录。
    :param paths: 路径迭代器
    :return: 仅包含目录的路径列表
    """
    return [p for p in paths if p.is_dir()]


def make_session_dir(root: Path, name: str) -> Path:
    """
    创建指定名称的会话目录。
    :param root: 会话根目录
    :param name: 会话目录名
    :return: 创建后的会话目录路径
    """
    session_dir = root / name
    ensure_dir(session_dir)
    return session_dir
