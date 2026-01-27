from pathlib import Path

from scanner.io.config import load_config


def test_load_config():
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")
    assert "device" in config
    assert "scan" in config
    assert "tracking" in config
