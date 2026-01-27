from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


class TrackingStatus(str, Enum):
    OK = "OK"
    WARN = "WARN"
    LOST = "LOST"


class ScanMode(str, Enum):
    REALTIME = "realtime"
    SEMI = "semi"
    OFFLINE = "offline"


@dataclass
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    depth_scale: float


@dataclass
class FrameBundle:
    color_rgb: np.ndarray
    depth_mm: np.ndarray
    timestamp: float
    intrinsics: Intrinsics
    index: int


@dataclass
class FramePacket:
    """兼容旧接口的 Frame 数据结构。"""
    color: np.ndarray
    depth: np.ndarray
    intrinsics: Dict[str, Any]
    timestamp: float
    index: int


@dataclass
class Keyframe:
    index: int
    pose: np.ndarray
    color: np.ndarray
    depth: np.ndarray
    intrinsics: Dict[str, Any]
    quality: float


@dataclass
class SessionInfo:
    session_id: str
    root: Path
    config: Dict[str, Any]
    keyframe_indices: List[int] = field(default_factory=list)
    trajectory: List[np.ndarray] = field(default_factory=list)
    bag_path: Optional[str] = None
    logs_path: Optional[str] = None
    exports: List[Dict[str, Any]] = field(default_factory=list)
    version: str = "0.1"
