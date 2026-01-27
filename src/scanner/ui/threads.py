from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np
import open3d as o3d
from PySide6.QtCore import QThread, Signal

from scanner.core.keyframe import KeyframePolicy, KeyframeSelector
from scanner.core.queues import FrameQueue
from scanner.core.types import FrameBundle, Keyframe, TrackingStatus
from scanner.core.timer import FPSCounter
from scanner.fusion.tsdf import TSDFVolumeManager
from scanner.tracking.odometry import RgbdOdometryTracker
from scanner.tracking.quality import QualityThresholds, evaluate_quality
from scanner.utils.matrix import identity, pose_distance


class CaptureThread(QThread):
    frame_signal = Signal(object)
    info_signal = Signal(dict)
    log_signal = Signal(str)
    error_signal = Signal(str)

    def __init__(self, device, config: Dict[str, Any]) -> None:
        super().__init__()
        self._device = device
        self._config = config
        self._running = False
        self._fps = FPSCounter()

    def run(self) -> None:
        """
        采集线程主循环，持续输出 RGB-D 帧。
        :return: 返回介绍
        """
        try:
            info = self._device.connect()
            self.log_signal.emit(f"设备连接成功: {info.name} {info.serial}")
        except Exception as exc:
            self.error_signal.emit(str(exc))
            return

        self._running = True
        while self._running:
            try:
                frame = self._device.read_frame()
            except Exception as exc:
                self.error_signal.emit(str(exc))
                break

            if frame is None:
                time.sleep(0.01)
                continue

            fps = self._fps.tick()
            self.info_signal.emit({"fps": fps, "index": frame.index})
            self.frame_signal.emit(frame)

        self._device.disconnect()
        self.log_signal.emit("采集线程已停止")

    def stop(self) -> None:
        """
        函数介绍。
        :return: 返回介绍
        """
        self._running = False


class ReconstructThread(QThread):
    preview_signal = Signal(object)
    preview_mesh_signal = Signal(object)
    status_signal = Signal(str)
    log_signal = Signal(str)
    trajectory_signal = Signal(object)

    def __init__(self, queue: FrameQueue[FrameBundle], config: Dict[str, Any]) -> None:
        super().__init__()
        self._queue = queue
        self._config = config
        self._running = False
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
        :return: 返回介绍
        """
        self._running = True
        prev_frame: Optional[FrameBundle] = None
        current_pose = identity()

        while self._running:
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
            if mode in ("realtime", "offline") or (mode == "semi" and frame.index % self._fusion_stride == 0):
                # 6) TSDF 融合
                self._tsdf.integrate_frame(frame, current_pose)

            if frame.index % self._preview_stride == 0:
                pcd = self._tsdf.extract_preview_point_cloud(self._preview_voxel)
                self.preview_signal.emit(pcd)
                if self._preview_mesh:
                    mesh = self._tsdf.extract_preview_mesh()
                    self.preview_mesh_signal.emit(mesh)

            prev_frame = frame

        self.log_signal.emit("重建线程已停止")

    def stop(self) -> None:
        """
        函数介绍。
        :return: 返回介绍
        """
        self._running = False

    def pause(self) -> None:
        """
        函数介绍。
        :return: 返回介绍
        """
        self._paused = True

    def resume(self) -> None:
        """
        函数介绍。
        :return: 返回介绍
        """
        self._paused = False
        self._lost_count = 0
        self.status_signal.emit(TrackingStatus.OK.value)

    def get_keyframes(self) -> List[Keyframe]:
        """
        函数介绍。
        :return: 返回介绍
        """
        return list(self._keyframes)

    def get_trajectory(self) -> List[np.ndarray]:
        """
        函数介绍。
        :return: 返回介绍
        """
        return list(self._trajectory)

    def get_tsdf(self) -> TSDFVolumeManager:
        """
        函数介绍。
        :return: 返回介绍
        """
        return self._tsdf


class OptimizeThread(QThread):
    mesh_signal = Signal(object)
    log_signal = Signal(str)

    def __init__(self, keyframes: List[Keyframe], config: Dict[str, Any]) -> None:
        super().__init__()
        self._keyframes = keyframes
        self._config = config
        self._result_mesh: Optional[o3d.geometry.TriangleMesh] = None

    def run(self) -> None:
        """
        执行回环检测、全局优化与重融合。
        :return: 返回介绍
        """
        if not self._keyframes:
            self.log_signal.emit("没有关键帧，跳过优化")
            return
        from scanner.loop_closure.pose_graph import PoseGraphOptimizer
        from scanner.fusion.tsdf import TSDFVolumeManager

        optimizer = PoseGraphOptimizer(self._config.get("loop_closure", {}))
        result = optimizer.build(self._keyframes)
        self.log_signal.emit(
            f"位姿图构建完成: odom={result.odom_edges} loop={result.loop_edges}"
        )
        optimizer.optimize(result.pose_graph)

        optimized_poses = [node.pose for node in result.pose_graph.nodes]
        tsdf = TSDFVolumeManager(self._config.get("fusion", {}))
        tsdf.integrate_keyframes(self._keyframes, optimized_poses)
        mesh = tsdf.extract_mesh()
        self._result_mesh = mesh
        self.mesh_signal.emit(mesh)
        self.log_signal.emit("全局优化与重融合完成")

    def get_result_mesh(self) -> Optional[o3d.geometry.TriangleMesh]:
        """
        函数介绍。
        :return: 返回介绍
        """
        return self._result_mesh
