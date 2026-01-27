from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from scanner.io.config import load_config
from scanner.ui.main_window import MainWindow


def main() -> None:
    """
    GUI 入口，加载默认配置并启动 Qt 应用。
    :return: 返回介绍
    """
    root = Path(__file__).resolve().parents[2]
    config_path = root / "configs" / "default.yaml"
    config = load_config(config_path)

    app = QApplication(sys.argv)
    window = MainWindow(config, config_path)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
