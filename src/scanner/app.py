from __future__ import annotations

import faulthandler
import logging
import os
import signal
import sys
import threading
import traceback
from pathlib import Path
from typing import Optional, TextIO

from scanner.io.config import load_config
from scanner.io.logger import configure_app_logging, install_qt_message_handler

_FAULT_LOG: Optional[TextIO] = None
_STOP_EVENT = threading.Event()
_FORCE_EXIT_TIMER: Optional[threading.Timer] = None


def _setup_env() -> None:
    if os.environ.get("DEPTHFORGE_SAFE_MODE", "0") == "1":
        os.environ.setdefault("QT_OPENGL", "software")
        os.environ.setdefault("OPEN3D_CPU_RENDERING", "1")


def _setup_qt_opengl() -> None:
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QSurfaceFormat
        from PySide6.QtWidgets import QApplication
    except Exception:
        return
    QApplication.setAttribute(Qt.AA_UseDesktopOpenGL, True)
    fmt = QSurfaceFormat()
    fmt.setRenderableType(QSurfaceFormat.OpenGL)
    fmt.setVersion(3, 2)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    fmt.setDepthBufferSize(24)
    fmt.setStencilBufferSize(8)
    QSurfaceFormat.setDefaultFormat(fmt)


def _set_logger_level(logger: logging.Logger, level: str | int) -> None:
    if isinstance(level, int):
        resolved = level
    else:
        resolved = getattr(logging, str(level).upper(), logging.INFO)
    logger.setLevel(resolved)
    for handler in logger.handlers:
        handler.setLevel(resolved)


def _setup_crash_logging(log_path: Path, logger: logging.Logger) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "a", buffering=1)
    global _FAULT_LOG
    _FAULT_LOG = log_file
    faulthandler.enable(file=log_file)
    if hasattr(signal, "SIGUSR1"):
        faulthandler.register(signal.SIGUSR1, file=log_file, all_threads=True)

    def _excepthook(exc_type, exc, tb) -> None:
        try:
            logger.error("Unhandled exception", exc_info=(exc_type, exc, tb))
            log_file.write("\n=== Unhandled Exception ===\n")
            log_file.write("".join(traceback.format_exception(exc_type, exc, tb)))
            log_file.write("\n")
            log_file.flush()
        finally:
            sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = _excepthook

    def _thread_excepthook(args: threading.ExceptHookArgs) -> None:
        logger.error(
            "Unhandled thread exception (%s)",
            args.thread.name,
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    threading.excepthook = _thread_excepthook


def _install_signal_handlers(logger: logging.Logger) -> None:
    def _sigint_handler(_signum, _frame) -> None:
        logger.warning("SIGINT received, requesting shutdown")
        _STOP_EVENT.set()
        global _FORCE_EXIT_TIMER
        if _FORCE_EXIT_TIMER is None:
            def _force_exit() -> None:
                logger.error("Force exit due to unresponsive shutdown")
                os._exit(130)
            _FORCE_EXIT_TIMER = threading.Timer(2.0, _force_exit)
            _FORCE_EXIT_TIMER.daemon = True
            _FORCE_EXIT_TIMER.start()

    signal.signal(signal.SIGINT, _sigint_handler)
    try:
        signal.siginterrupt(signal.SIGINT, False)
    except Exception:
        pass


def main() -> None:
    """
    GUI 入口，加载默认配置并启动 Qt 应用。
    :return: 返回介绍
    """
    root = Path(__file__).resolve().parents[2]
    log_level = os.environ.get("DEPTHFORGE_LOG_LEVEL", "INFO")
    logger, log_path = configure_app_logging(root / "logs", level=log_level)

    logger.info("=== DepthForge GUI Starting ===")
    logger.info("Log file: %s", log_path)
    _setup_env()
    _setup_qt_opengl()
    _setup_crash_logging(log_path, logger)
    _install_signal_handlers(logger)

    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication
    from scanner.ui.main_window import MainWindow

    config_path = root / "configs" / "default.yaml"
    logger.info("Loading config: %s", config_path)
    config = load_config(config_path)
    cfg_level = config.get("logging", {}).get("level")
    if cfg_level and "DEPTHFORGE_LOG_LEVEL" not in os.environ:
        _set_logger_level(logger, cfg_level)
        logger.info("Log level set to %s from config", cfg_level)

    logger.info("Creating QApplication")
    app = QApplication(sys.argv)
    try:
        from PySide6.QtGui import QGuiApplication

        platform = QGuiApplication.platformName()
        screen = QGuiApplication.primaryScreen()
        geom = screen.availableGeometry() if screen else None
        logger.info("Qt platform=%s screen=%s", platform, geom)
    except Exception:
        logger.exception("Failed to query Qt platform/screen")
    install_qt_message_handler(logger)

    logger.info("Creating MainWindow")
    window = MainWindow(config, config_path, stop_event=_STOP_EVENT)
    window.show()
    try:
        app.processEvents()
        logger.info(
            "Window immediate visible=%s minimized=%s geom=%s",
            window.isVisible(),
            window.isMinimized(),
            window.geometry(),
        )
        if not window.isVisible() or window.isMinimized():
            window.showNormal()
        window.raise_()
        window.activateWindow()
        app.processEvents()
    except Exception:
        logger.exception("Initial window show failed")
    logger.info("Main window shown")

    tick = QTimer()
    tick.setInterval(200)

    def _tick() -> None:
        if _STOP_EVENT.is_set():
            logger.info("Stop flag set, quitting Qt loop")
            app.quit()

    tick.timeout.connect(_tick)
    tick.start()

    def _ensure_visible() -> None:
        try:
            logger.info(
                "Window check visible=%s minimized=%s geom=%s",
                window.isVisible(),
                window.isMinimized(),
                window.geometry(),
            )
            if not window.isVisible():
                window.show()
            if window.isMinimized():
                window.showNormal()
            window.raise_()
            window.activateWindow()
        except Exception:
            logger.exception("Ensure window visible failed")

    QTimer.singleShot(300, _ensure_visible)
    QTimer.singleShot(1500, _ensure_visible)

    exit_code = app.exec()
    logger.info("Qt loop exited with code %s", exit_code)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
