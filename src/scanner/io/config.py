from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: Path) -> Dict[str, Any]:
    """
    读取 YAML 配置文件。
    :param path: 配置文件路径
    :return: 配置字典
    """
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data


def save_config(path: Path, config: Dict[str, Any]) -> None:
    """
    保存配置到 YAML 文件。
    :param path: 输出文件路径
    :param config: 配置字典
    :return: None
    """
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
