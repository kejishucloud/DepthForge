from __future__ import annotations

import logging
import threading
import time
import traceback
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
from PySide6.QtCore import QThread, Signal

from scanner.core.keyframe import KeyframePolicy, KeyframeSelector
from scanner.core.queues import FrameQueue
from scanner.core.types import FrameBundle, Keyframe, TrackingStatus
from scanner.core.timer import FPSCounter
from scanner.fusion.tsdf import TSDFVolumeManager
from scanner.tracking.odometry import RgbdOdometryTracker
from scanner.tracking.quality import QualityThresholds, evaluate_quality
from scanner.utils.matrix import identity, pose_distance
from scanner.devices import realsense_device as rs_device

if TYPE_CHECKING:  # pragma: no cover
    import open3d as o3d

_LOGGER = logging.getLogger("scanner")


class CaptureThread(QThread):
    frame_signal = Signal(object)
    info_signal = Signal(dict)
    log_signal = Signal(str)
    error_signal = Signal(str)
    device_signal = Signal(object)

    def __init__(self, device, config: Dict[str, Any]) -> None:
        super().__init__()
        self._device = device
        self._config = config
        self._stop_event = threading.Event()
        self._fps = FPSCounter()
        self._last_fps_log = 0.0
        self._frame_timeout_ms = int(self._config.get("device", {}).get("frame_timeout_ms", 500))

    def run(self) -> None:
        """
        采集线程主循环，持续输出 RGB-D 帧。
        :return: 返回介绍
        """
        _LOGGER.info("CaptureThread starting")
        self.log_signal.emit("采集线程启动")
        connect_timeout = float(self._config.get("device", {}).get("connect_timeout_sec", 5))
        connect_result: dict[str, Any] = {}

        def _do_connect() -> None:
            try:
                connect_result["info"] = self._device.connect()
            except Exception as exc:  # pragma: no cover - runtime dependent
                connect_result["error"] = exc

        connect_thread = threading.Thread(target=_do_connect, name="rs-connect", daemon=True)
        connect_thread.start()
        connect_thread.join(connect_timeout)

        if connect_thread.is_alive():
            self.error_signal.emit(f"设备连接超时（>{connect_timeout}s）")
            _LOGGER.error("CaptureThread connect timeout")
            return
        if "error" in connect_result:
            exc = connect_result["error"]
            _LOGGER.exception("CaptureThread connect failed", exc_info=exc)
            self.error_signal.emit(self._format_exception("connect", exc))
            return

        info = connect_result.get("info")
        if info is None:
            self.error_signal.emit("设备连接失败：未知错误")
            return
        self.log_signal.emit(f"设备连接成功: {info.name} {info.serial}")
        self.device_signal.emit(info)

        try:
            while not self._stop_event.is_set():
                try:
                    frame = self._device.read_frame(timeout_ms=self._frame_timeout_ms)
                except Exception as exc:
                    _LOGGER.exception("CaptureThread read_frame failed")
                    self.error_signal.emit(self._format_exception("read_frame", exc))
                    break

                if frame is None:
                    time.sleep(0.01)
                    continue

                fps = self._fps.tick()
                self.info_signal.emit({"fps": fps, "index": frame.index})
                self.frame_signal.emit(frame)

                now = time.time()
                if now - self._last_fps_log > 2.0:
                    self.log_signal.emit(f"FPS: {fps:.1f}")
                    self._last_fps_log = now
        except Exception as exc:
            _LOGGER.exception("CaptureThread crashed")
            self.error_signal.emit(self._format_exception("run", exc))
        finally:
            try:
                self._device.disconnect()
            except Exception:
                _LOGGER.exception("CaptureThread disconnect failed")
            self.log_signal.emit("采集线程已停止")
            _LOGGER.info("CaptureThread stopped")

    def stop(self) -> None:
        """
        请求停止采集线程。
        :return: None
        """
        self._stop_event.set()

    @staticmethod
    def _format_exception(stage: str, exc: Exception) -> str:
        rs_mod = rs_device.get_rs_module()
        rs_error = getattr(rs_mod, "error", None) if rs_mod is not None else None
        if rs_error is not None and isinstance(exc, rs_error):
            return f"RealSense {stage} error: {exc}"
        if isinstance(exc, RuntimeError):
            return f"RuntimeError during {stage}: {exc}"
        return f"{type(exc).__name__} during {stage}: {exc}"


class ReconstructThread(QThread):
    preview_signal = Signal(object)
    preview_mesh_signal = Signal(object)
    status_signal = Signal(str)
    log_signal = Signal(str)
    trajectory_signal = Signal(object)
    error_signal = Signal(str)

    def __init__(self, queue: FrameQueue[FrameBundle], config: Dict[str, Any]) -> None:
        super().__init__()
        self._queue = queue
        self._config = config
        self._stop_event = threading.Event()
        self._paused = False
        self._tracker = RgbdOdometryTracker(config.get("tracking", {}).get("odom", {}))
        self._quality = QualityThresholds(**config.get("tracking", {}).get("quality", {}))
        self._tsdf = TSDFVolumeManager(config.get("fusion", {}))
        policy_cfg = config.get("keyframe", {})
        self._key_selector = KeyframeSelector(
            KeyframePolicy(
                min_translation=float(policy_cfg.get("min_translation", 0.03)),
                min_rotation_deg=float(policy_cfg.get("min_rotation_deg", 5.0)),
                min_quality=float(policy_cfg.get("min_quality", 0.2)),
                min_interval=int(policy_cfg.get("min_interval", 10)),
                max_keyframes=int(policy_cfg.get("max_keyframes", 300)),
                mode_overrides=policy_cfg.get("mode_overrides", {}),
            ),
            mode=config.get("scan", {}).get("mode", "realtime"),
        )
        self._preview_stride = int(config.get("scan", {}).get("preview_stride", 5))
        self._fusion_stride = int(config.get("scan", {}).get("fusion_stride", 2))
        self._stop_on_lost = bool(config.get("scan", {}).get("stop_on_lost", True))
        self._lost_max_frames = int(config.get("scan", {}).get("lost_max_frames", 5))
        self._preview_voxel = float(config.get("fusion", {}).get("preview_voxel", 0.01))
        self._preview_mesh = bool(config.get("fusion", {}).get("preview_mesh", False))

        self._keyframes: List[Keyframe] = []
        self._trajectory: List[np.ndarray] = []
        self._status = TrackingStatus.OK
        self._lost_count = 0

    def run(self) -> None:
        """
        跟踪与融合主循环。
        :return: None
        """
        _LOGGER.info("ReconstructThread starting")
        self.log_signal.emit("重建线程启动")
        prev_frame: Optional[FrameBundle] = None
        current_pose = identity()

        try:
            while not self._stop_event.is_set():
                if self._paused:
                    time.sleep(0.05)
                    continue

                frame = self._queue.get(timeout=0.1)
                if frame is None:
                    continue

                if prev_frame is None:
                    prev_frame = frame
                    self._trajectory.append(current_pose.copy())
                    # 首帧仅初始化，不进行配准
                    if self._key_selector.should_add(frame.index, current_pose, 1.0):
                        self._keyframes.append(
                            Keyframe(
                                index=frame.index,
                                pose=current_pose.copy(),
                                color=frame.color_rgb.copy(),
                                depth=frame.depth_mm.copy(),
                                intrinsics=self._key_selector.intrinsics_to_dict(frame.intrinsics),
                                quality=1.0,
                            )
                        )
                        self._key_selector.update(frame.index, current_pose.copy(), self._keyframes)
                    continue

                # 1) RGB-D 里程计估计相邻帧相对位姿
                odom_result = self._tracker.estimate(prev_frame, frame, np.eye(4))
                fitness = odom_result.fitness
                rmse = odom_result.rmse
                pose_update = odom_result.pose
                delta_translation, delta_rotation = pose_distance(identity(), pose_update)

                # 2) ICP 精配提高稳定性与可解释的质量指标
                icp_cfg = self._config.get("tracking", {}).get("icp_refine", {})
                icp_success = False
                if bool(icp_cfg.get("enable", True)):
                    icp_result = self._tracker.refine_icp(
                        prev_frame,
                        frame,
                        pose_update,
                        float(icp_cfg.get("max_correspondence", 0.03)),
                    )
                    if icp_result.success:
                        icp_success = True
                        pose_update = icp_result.pose
                        fitness = icp_result.fitness
                        rmse = icp_result.rmse
                        delta_translation, delta_rotation = pose_distance(identity(), pose_update)

                # 3) 质量评估并更新状态
                if not odom_result.success and not icp_success:
                    self._status = TrackingStatus.LOST
                else:
                    self._status = evaluate_quality(
                        fitness, rmse, delta_translation, delta_rotation, self._quality
                    )
                self.status_signal.emit(self._status.value)

                if self._status == TrackingStatus.LOST:
                    self._lost_count += 1
                    if self._stop_on_lost and self._lost_count >= self._lost_max_frames:
                        self.log_signal.emit("跟踪丢失，已暂停融合，请调整姿态恢复后继续")
                        self._paused = True
                    prev_frame = frame
                    continue

                self._lost_count = 0
                # 4) 累积全局位姿
                current_pose = current_pose @ pose_update
                self._trajectory.append(current_pose.copy())

                # 5) 关键帧采样
                if self._key_selector.should_add(frame.index, current_pose, float(fitness)):
                    self._keyframes.append(
                        Keyframe(
                            index=frame.index,
                            pose=current_pose.copy(),
                            color=frame.color_rgb.copy(),
                            depth=frame.depth_mm.copy(),
                            intrinsics=self._key_selector.intrinsics_to_dict(frame.intrinsics),
                            quality=float(fitness),
                        )
                    )
                    self._key_selector.update(frame.index, current_pose.copy(), self._keyframes)

                mode = self._config.get("scan", {}).get("mode", "realtime")
                if mode in ("realtime", "offline") or (
                    mode == "semi" and frame.index % self._fusion_stride == 0
                ):
                    # 6) TSDF 融合
                    self._tsdf.integrate_frame(frame, current_pose)

                if frame.index % self._preview_stride == 0:
                    pcd = self._tsdf.extract_preview_point_cloud(self._preview_voxel)
                    self.preview_signal.emit(pcd)
                    if self._preview_mesh:
                        mesh = self._tsdf.extract_preview_mesh()
                        self.preview_mesh_signal.emit(mesh)

                prev_frame = frame
        except Exception as exc:
            _LOGGER.exception("ReconstructThread crashed")
            detail = traceback.format_exc()
            self.error_signal.emit(f"ReconstructThread error: {exc}\n{detail}")
        finally:
            self.log_signal.emit("重建线程已停止")
            _LOGGER.info("ReconstructThread stopped")

    def stop(self) -> None:
        """
        请求停止重建线程。
        :return: None
        """
        self._stop_event.set()

    def pause(self) -> None:
        """
        暂停重建（停止融合与输出）。
        :return: None
        """
        self._paused = True

    def resume(self) -> None:
        """
        恢复重建并清空丢失计数。
        :return: None
        """
        self._paused = False
        self._lost_count = 0
        self.status_signal.emit(TrackingStatus.OK.value)

    def get_keyframes(self) -> List[Keyframe]:
        """
        获取当前关键帧列表的副本。
        :return: 关键帧列表
        """
        return list(self._keyframes)

    def get_trajectory(self) -> List[np.ndarray]:
        """
        获取轨迹列表的副本。
        :return: 位姿序列
        """
        return list(self._trajectory)

    def get_tsdf(self) -> TSDFVolumeManager:
        """
        获取当前 TSDF 管理器实例。
        :return: TSDF 管理器
        """
        return self._tsdf


class OptimizeThread(QThread):
    mesh_signal = Signal(object)
    log_signal = Signal(str)
    error_signal = Signal(str)

    def __init__(self, keyframes: List[Keyframe], config: Dict[str, Any]) -> None:
        super().__init__()
        self._keyframes = keyframes
        self._config = config
        self._result_mesh: Optional["o3d.geometry.TriangleMesh"] = None
        self._stop_event = threading.Event()

    def run(self) -> None:
        """
        执行回环检测、全局优化与重融合。
        :return: None
        """
        _LOGGER.info("OptimizeThread starting")
        self.log_signal.emit("优化线程启动")
        try:
            if not self._keyframes:
                self.log_signal.emit("没有关键帧，跳过优化")
                return
            if self._stop_event.is_set():
                return
            from scanner.loop_closure.pose_graph import PoseGraphOptimizer
            from scanner.fusion.tsdf import TSDFVolumeManager

            optimizer = PoseGraphOptimizer(self._config.get("loop_closure", {}))
            result = optimizer.build(self._keyframes)
            self.log_signal.emit(
                f"位姿图构建完成: odom={result.odom_edges} loop={result.loop_edges}"
            )
            if self._stop_event.is_set():
                return
            optimizer.optimize(result.pose_graph)

            if self._stop_event.is_set():
                return
            optimized_poses = [node.pose for node in result.pose_graph.nodes]
            tsdf = TSDFVolumeManager(self._config.get("fusion", {}))
            tsdf.integrate_keyframes(self._keyframes, optimized_poses)
            mesh = tsdf.extract_mesh()
            self._result_mesh = mesh
            self.mesh_signal.emit(mesh)
            self.log_signal.emit("全局优化与重融合完成")
        except Exception as exc:
            _LOGGER.exception("OptimizeThread crashed")
            self.error_signal.emit(f"OptimizeThread error: {exc}")
        finally:
            self.log_signal.emit("优化线程已停止")
            _LOGGER.info("OptimizeThread stopped")

    def get_result_mesh(self) -> Optional[o3d.geometry.TriangleMesh]:
        """
        获取优化线程生成的网格结果。
        :return: 网格或 None
        """
        return self._result_mesh

    def stop(self) -> None:
        self._stop_event.set()
