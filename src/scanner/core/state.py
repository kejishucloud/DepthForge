from __future__ import annotations

from enum import Enum


class ScannerState(str, Enum):
    IDLE = "IDLE"
    PREVIEW = "PREVIEW"
    SCANNING = "SCANNING"
    PLAYBACK = "PLAYBACK"
    RECORDING = "RECORDING"
    PAUSED = "PAUSED"
    LOST = "LOST"
    OPTIMIZING = "OPTIMIZING"
    EXPORTING = "EXPORTING"
    STOPPING = "STOPPING"
