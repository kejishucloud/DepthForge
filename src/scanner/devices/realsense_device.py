from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import platform
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_RS = None
_LOGGER = logging.getLogger("scanner")


def _require_rs():
    global _RS
    if _RS is None:
        try:
            import pyrealsense2 as rs  # type: ignore
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError("pyrealsense2 未安装或无法加载，请先安装 RealSense SDK。") from exc
        _RS = rs
    return _RS


def get_rs_module():
    return _RS

from scanner.core.types import FrameBundle, Intrinsics


@dataclass
class DeviceInfo:
    name: str
    serial: str
    firmware: str


class RealSenseDevice:
    def __init__(self, config: Dict[str, Any]) -> None:
        self._config = config
        self._pipeline: Optional["rs.pipeline"] = None
        self._profile: Optional["rs.pipeline_profile"] = None
        self._align: Optional["rs.align"] = None
        self._depth_scale: float = 0.001
        self._intrinsics: Optional[Intrinsics] = None
        self._frame_index = 0
        self._last_frame_time = 0.0
        self._filters: Dict[str, Any] = {}
        self._color_format = self._normalize_color_format(self._config.get("color_format"))

    @staticmethod
    def _normalize_color_format(value: Optional[str]) -> str:
        if value is None:
            value = "auto"
        mode = str(value).strip().lower()
        if mode in ("", "auto", "default"):
            return "rgb8" if platform.system() == "Darwin" else "bgr8"
        if mode in ("rgb", "rgb8"):
            return "rgb8"
        if mode in ("bgr", "bgr8"):
            return "bgr8"
        raise ValueError(f"不支持的 color_format: {value}，仅支持 rgb8 / bgr8 / auto")

    @staticmethod
    def list_devices() -> List[DeviceInfo]:
        """
        枚举当前可用的 RealSense 设备信息。
        :return: 设备信息列表
        """
        rs = _require_rs()
        ctx = rs.context()
        devices = []
        for dev in ctx.query_devices():
            devices.append(
                DeviceInfo(
                    name=dev.get_info(rs.camera_info.name),
                    serial=dev.get_info(rs.camera_info.serial_number),
                    firmware=dev.get_info(rs.camera_info.firmware_version),
                )
            )
        return devices

    @staticmethod
    def has_device() -> bool:
        """
        检查是否存在可用 RealSense 设备。
        :return: 是否检测到设备
        """
        rs = _require_rs()
        ctx = rs.context()
        return len(ctx.query_devices()) > 0

    def connect(self) -> DeviceInfo:
        """
        连接设备并初始化流、对齐与深度尺度。
        :return: 设备信息
        """
        rs = _require_rs()

        self._frame_index = 0
        pipeline = rs.pipeline()
        cfg = rs.config()

        if self._config.get("playback_bag"):
            bag_path = self._resolve_bag_path(self._config.get("bag_path"), mode="playback")
            cfg.enable_device_from_file(str(bag_path), repeat_playback=False)
        else:
            if not self.has_device():
                raise RuntimeError("未检测到 RealSense 设备，请检查连接/驱动/权限")

        if self._config.get("record_bag"):
            bag_path = self._resolve_bag_path(self._config.get("bag_path"), mode="record")
            cfg.enable_record_to_file(str(bag_path))

        if self._config.get("enable_color", True):
            color_format = rs.format.rgb8 if self._color_format == "rgb8" else rs.format.bgr8
            cfg.enable_stream(
                rs.stream.color,
                int(self._config.get("color_width", 640)),
                int(self._config.get("color_height", 480)),
                color_format,
                int(self._config.get("fps", 30)),
            )

        if self._config.get("enable_depth", True):
            cfg.enable_stream(
                rs.stream.depth,
                int(self._config.get("depth_width", 640)),
                int(self._config.get("depth_height", 480)),
                rs.format.z16,
                int(self._config.get("fps", 30)),
            )

        try:
            profile = pipeline.start(cfg)
        except Exception as exc:
            _LOGGER.exception("RealSense pipeline start failed")
            raise RuntimeError(f"RealSense pipeline start 失败: {exc}") from exc
        self._pipeline = pipeline
        self._profile = profile

        if self._config.get("playback_bag"):
            device = profile.get_device()
            playback = device.as_playback()
            playback.set_real_time(self._config.get("playback_real_time", True))

        if self._config.get("align_to_color", True):
            self._align = rs.align(rs.stream.color)

        depth_sensor = profile.get_device().first_depth_sensor()
        self._depth_scale = float(depth_sensor.get_depth_scale())

        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_stream.get_intrinsics()
        self._intrinsics = Intrinsics(
            width=intr.width,
            height=intr.height,
            fx=intr.fx,
            fy=intr.fy,
            cx=intr.ppx,
            cy=intr.ppy,
            depth_scale=self._depth_scale,
        )

        self._init_filters()

        device = profile.get_device()
        name = device.get_info(rs.camera_info.name)
        serial = device.get_info(rs.camera_info.serial_number)
        firmware = device.get_info(rs.camera_info.firmware_version)
        return DeviceInfo(name=name, serial=serial, firmware=firmware)

    def disconnect(self) -> None:
        """
        停止管线并释放设备资源。
        :return: None
        """
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
        self._pipeline = None
        self._profile = None
        self._align = None
        self._filters = {}
        self._intrinsics = None

    def read_frame(self, timeout_ms: int = 1000) -> Optional[FrameBundle]:
        """
        读取一帧对齐后的彩色/深度图并裁剪深度范围。
        :param timeout_ms: 等待超时（毫秒）
        :return: 帧数据，若超时/无有效帧则返回 None
        """
        if self._pipeline is None:
            return None

        try:
            frames = self._pipeline.wait_for_frames(timeout_ms)
        except Exception as exc:
            message = str(exc).lower()
            if "frame didn't arrive" in message or "frame didnt arrive" in message or "timeout" in message:
                _LOGGER.debug("RealSense frame timeout after %sms", timeout_ms)
                return None
            raise
        if self._align is not None:
            frames = self._align.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            return None

        depth_frame = self._apply_filters(depth_frame)

        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())
        if self._color_format == "bgr8":
            color_rgb = color[:, :, ::-1].copy()
        else:
            color_rgb = color.copy()

        depth_mm = (depth.astype(np.float32) * self._depth_scale * 1000.0).astype(np.uint16)

        now = float(color_frame.get_timestamp() / 1000.0)
        self._last_frame_time = now
        if self._intrinsics is None:
            raise RuntimeError("相机内参未初始化")
        bundle = FrameBundle(
            color_rgb=color_rgb,
            depth_mm=depth_mm,
            timestamp=now,
            index=self._frame_index,
            intrinsics=self._intrinsics,
        )
        self._frame_index += 1
        return bundle

    def get_depth_scale(self) -> float:
        """
        获取深度尺度（米/单位）。
        :return: 深度尺度
        """
        return self._depth_scale

    def get_last_frame_time(self) -> float:
        """
        获取上一帧时间戳。
        :return: 上一帧时间戳（秒）
        """
        return self._last_frame_time

    def get_intrinsics(self) -> Optional[Intrinsics]:
        """
        获取当前相机内参。
        :return: 内参对象（若未初始化则为 None）
        """
        return self._intrinsics

    def _init_filters(self) -> None:
        """
        根据配置初始化深度滤波链。
        :return: None
        """
        filters_cfg = self._config.get("filters", {})
        self._filters = {}
        rs = _require_rs()
        if filters_cfg.get("enable_decimation", False):
            decimation = rs.decimation_filter()
            decimation.set_option(rs.option.filter_magnitude, float(filters_cfg.get("decimation_magnitude", 2)))
            self._filters["decimation"] = decimation
        if filters_cfg.get("enable_spatial", True):
            spatial = rs.spatial_filter()
            spatial.set_option(rs.option.filter_magnitude, float(filters_cfg.get("spatial_magnitude", 2)))
            spatial.set_option(rs.option.filter_smooth_alpha, float(filters_cfg.get("spatial_alpha", 0.5)))
            spatial.set_option(rs.option.filter_smooth_delta, float(filters_cfg.get("spatial_delta", 20)))
            self._filters["spatial"] = spatial
        if filters_cfg.get("enable_temporal", True):
            temporal = rs.temporal_filter()
            temporal.set_option(rs.option.filter_smooth_alpha, float(filters_cfg.get("temporal_alpha", 0.4)))
            temporal.set_option(rs.option.filter_smooth_delta, float(filters_cfg.get("temporal_delta", 20)))
            self._filters["temporal"] = temporal
        if filters_cfg.get("enable_hole_filling", True):
            hole = rs.hole_filling_filter(int(filters_cfg.get("hole_filling_mode", 1)))
            self._filters["hole"] = hole
        if filters_cfg.get("enable_threshold", True):
            thresh = rs.threshold_filter(
                float(self._config.get("depth_min_m", 0.2)),
                float(self._config.get("depth_max_m", 1.0)),
            )
            self._filters["threshold"] = thresh

    def _resolve_bag_path(self, bag_path: Optional[str], mode: str) -> Path:
        if not bag_path:
            raise ValueError(f"{mode} 模式需要提供 bag_path")
        path = Path(bag_path).expanduser()
        if path.suffix.lower() != ".bag":
            raise ValueError("bag_path 必须以 .bag 结尾")
        if mode == "record":
            path = path.resolve()
            parent = path.parent
            parent.mkdir(parents=True, exist_ok=True)
            if not parent.exists() or not parent.is_dir():
                raise ValueError(f"bag 路径目录不存在: {parent}")
            if not os.access(parent, os.W_OK):
                raise ValueError(f"bag 路径不可写: {parent}")
        else:
            path = path.resolve()
            if not path.exists():
                raise ValueError(f"bag 文件不存在: {path}")
        self._config["bag_path"] = str(path)
        return path

    def _apply_filters(self, depth_frame: "rs.depth_frame") -> "rs.depth_frame":
        """
        依次应用配置的深度滤波器。
        :param depth_frame: 原始深度帧
        :return: 处理后的深度帧
        """
        frame = depth_frame
        for key in ("decimation", "spatial", "temporal", "hole", "threshold"):
            filt = self._filters.get(key)
            if filt is not None:
                frame = filt.process(frame)
        return frame
