from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import open3d as o3d
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QLabel,
    QMainWindow,
    QPushButton,
    QCheckBox,
    QDoubleSpinBox,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from scanner.core.queues import FrameQueue
from scanner.core.state import ScannerState
from scanner.core.types import FrameBundle, SessionInfo
from scanner.devices.realsense_device import RealSenseDevice
from scanner.geometry.exporter import export_mesh, export_point_cloud
from scanner.geometry.postprocess import postprocess_mesh
from scanner.io.session import SessionManager
from scanner.io.logger import setup_logger
from scanner.ui.qt_viz import Qt3DViewer
from scanner.ui.threads import CaptureThread, OptimizeThread, ReconstructThread
from scanner.utils.path import ensure_dir

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


class MainWindow(QMainWindow):
    def __init__(self, config: Dict[str, Any], config_path: Path) -> None:
        super().__init__()
        self._config = config
        self._config_path = config_path
        self._session: Optional[SessionInfo] = None
        self._session_manager = SessionManager(Path(config.get("session_root", "sessions")))
        self._logger = None

        self._capture_thread: Optional[CaptureThread] = None
        self._reconstruct_thread: Optional[ReconstructThread] = None
        self._optimize_thread: Optional[OptimizeThread] = None
        self._queue: Optional[FrameQueue[FrameBundle]] = None
        self._state = ScannerState.IDLE

        self._current_mesh: Optional[o3d.geometry.TriangleMesh] = None
        self._current_pcd: Optional[o3d.geometry.PointCloud] = None

        self.setWindowTitle("RealSense D435 手持 3D 扫描建模")
        self.resize(1400, 820)
        self._build_ui()
        self._set_state(ScannerState.IDLE)

    def _build_ui(self) -> None:
        central = QWidget(self)
        layout = QHBoxLayout(central)

        left_panel = QWidget(self)
        left_layout = QVBoxLayout(left_panel)

        control_group = QGroupBox("扫描控制")
        control_layout = QGridLayout(control_group)

        self.start_btn = QPushButton("开始")
        self.pause_btn = QPushButton("暂停")
        self.resume_btn = QPushButton("继续")
        self.stop_btn = QPushButton("停止")
        self.optimize_btn = QPushButton("全局优化")
        self.export_btn = QPushButton("导出")
        self.save_session_btn = QPushButton("保存会话")
        self.load_session_btn = QPushButton("加载会话")

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["realtime", "semi", "offline"])
        self.mode_combo.setCurrentText(self._config.get("scan", {}).get("mode", "realtime"))

        self.bag_path_edit = QLineEdit()
        self.bag_path_edit.setPlaceholderText("bag 路径（回放或录制）")
        self.bag_path_edit.setText(self._config.get("device", {}).get("bag_path", ""))
        self.record_checkbox = QCheckBox("录制 .bag")
        self.record_checkbox.setChecked(bool(self._config.get("device", {}).get("record_bag", False)))

        self.export_combo = QComboBox()
        self.export_combo.addItems(["ply", "obj", "stl"])
        self.export_combo.setCurrentText(self._config.get("export", {}).get("default_mesh_format", "ply"))
        self.cloud_export_combo = QComboBox()
        self.cloud_export_combo.addItems(["ply", "pcd"])
        self.cloud_export_combo.setCurrentText(self._config.get("export", {}).get("default_cloud_format", "ply"))

        control_layout.addWidget(QLabel("模式"), 0, 0)
        control_layout.addWidget(self.mode_combo, 0, 1)
        control_layout.addWidget(QLabel("bag 路径"), 1, 0)
        control_layout.addWidget(self.bag_path_edit, 1, 1)
        control_layout.addWidget(self.record_checkbox, 2, 0, 1, 2)
        control_layout.addWidget(self.start_btn, 3, 0)
        control_layout.addWidget(self.pause_btn, 3, 1)
        control_layout.addWidget(self.resume_btn, 4, 0)
        control_layout.addWidget(self.stop_btn, 4, 1)
        control_layout.addWidget(self.optimize_btn, 5, 0)
        control_layout.addWidget(self.export_btn, 5, 1)
        control_layout.addWidget(self.save_session_btn, 6, 0)
        control_layout.addWidget(self.load_session_btn, 6, 1)
        control_layout.addWidget(QLabel("网格导出"), 7, 0)
        control_layout.addWidget(self.export_combo, 7, 1)
        control_layout.addWidget(QLabel("点云导出"), 8, 0)
        control_layout.addWidget(self.cloud_export_combo, 8, 1)

        params_group = QGroupBox("参数")
        params_layout = QFormLayout(params_group)
        self.voxel_spin = QDoubleSpinBox()
        self.voxel_spin.setDecimals(4)
        self.voxel_spin.setRange(0.001, 0.02)
        self.voxel_spin.setValue(float(self._config.get("fusion", {}).get("voxel_length", 0.004)))
        self.sdf_spin = QDoubleSpinBox()
        self.sdf_spin.setDecimals(4)
        self.sdf_spin.setRange(0.005, 0.1)
        self.sdf_spin.setValue(float(self._config.get("fusion", {}).get("sdf_trunc", 0.02)))
        self.depth_trunc_spin = QDoubleSpinBox()
        self.depth_trunc_spin.setDecimals(2)
        self.depth_trunc_spin.setRange(0.3, 3.0)
        self.depth_trunc_spin.setValue(float(self._config.get("fusion", {}).get("depth_trunc", 1.0)))

        self.key_trans_spin = QDoubleSpinBox()
        self.key_trans_spin.setDecimals(3)
        self.key_trans_spin.setRange(0.005, 0.2)
        self.key_trans_spin.setValue(float(self._config.get("keyframe", {}).get("min_translation", 0.03)))
        self.key_rot_spin = QDoubleSpinBox()
        self.key_rot_spin.setDecimals(1)
        self.key_rot_spin.setRange(1.0, 30.0)
        self.key_rot_spin.setValue(float(self._config.get("keyframe", {}).get("min_rotation_deg", 5.0)))

        self.loop_gap_spin = QSpinBox()
        self.loop_gap_spin.setRange(10, 200)
        self.loop_gap_spin.setValue(int(self._config.get("loop_closure", {}).get("temporal_gap", 30)))
        self.loop_dist_spin = QDoubleSpinBox()
        self.loop_dist_spin.setDecimals(2)
        self.loop_dist_spin.setRange(0.1, 2.0)
        self.loop_dist_spin.setValue(float(self._config.get("loop_closure", {}).get("max_distance", 0.4)))
        self.loop_candidates_spin = QSpinBox()
        self.loop_candidates_spin.setRange(1, 20)
        self.loop_candidates_spin.setValue(int(self._config.get("loop_closure", {}).get("max_candidates", 5)))

        self.simplify_spin = QSpinBox()
        self.simplify_spin.setRange(1000, 2000000)
        self.simplify_spin.setValue(int(self._config.get("mesh", {}).get("simplify_target_triangles", 200000)))
        self.smooth_spin = QSpinBox()
        self.smooth_spin.setRange(0, 50)
        self.smooth_spin.setValue(int(self._config.get("mesh", {}).get("smooth_iterations", 5)))

        self.preview_mesh_checkbox = QCheckBox("预览 Mesh")
        self.preview_mesh_checkbox.setChecked(bool(self._config.get("fusion", {}).get("preview_mesh", False)))

        self.view_combo = QComboBox()
        self.view_combo.addItems(["pointcloud", "mesh"])
        self.reset_camera_btn = QPushButton("重置相机")
        self.toggle_axes_btn = QPushButton("切换坐标轴")

        params_layout.addRow("voxel_length", self.voxel_spin)
        params_layout.addRow("sdf_trunc", self.sdf_spin)
        params_layout.addRow("depth_trunc", self.depth_trunc_spin)
        params_layout.addRow("keyframe Δt", self.key_trans_spin)
        params_layout.addRow("keyframe ΔR", self.key_rot_spin)
        params_layout.addRow("loop gap", self.loop_gap_spin)
        params_layout.addRow("loop dist", self.loop_dist_spin)
        params_layout.addRow("loop cand", self.loop_candidates_spin)
        params_layout.addRow("mesh simplify", self.simplify_spin)
        params_layout.addRow("mesh smooth", self.smooth_spin)
        params_layout.addRow(self.preview_mesh_checkbox)
        params_layout.addRow("视图", self.view_combo)
        params_layout.addRow(self.reset_camera_btn, self.toggle_axes_btn)

        status_group = QGroupBox("状态")
        status_layout = QVBoxLayout(status_group)
        self.state_label = QLabel("State: IDLE")
        self.status_label = QLabel("Tracking: --")
        self.fps_label = QLabel("FPS: --")
        self.info_label = QLabel("")
        status_layout.addWidget(self.state_label)
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.fps_label)
        status_layout.addWidget(self.info_label)

        preview_group = QGroupBox("预览")
        preview_layout = QVBoxLayout(preview_group)
        self.color_label = QLabel("RGB")
        self.depth_label = QLabel("Depth")
        self.color_label.setFixedSize(320, 240)
        self.depth_label.setFixedSize(320, 240)
        self.color_label.setStyleSheet("background:#222;color:#ddd")
        self.depth_label.setStyleSheet("background:#222;color:#ddd")
        self.color_label.setAlignment(Qt.AlignCenter)
        self.depth_label.setAlignment(Qt.AlignCenter)
        preview_layout.addWidget(self.color_label)
        preview_layout.addWidget(self.depth_label)

        left_layout.addWidget(preview_group)
        left_layout.addWidget(status_group)
        left_layout.addStretch(1)

        right_panel = QWidget(self)
        right_layout = QVBoxLayout(right_panel)

        self.viewer = Qt3DViewer(self)
        right_layout.addWidget(self.viewer, 4)
        right_layout.addWidget(params_group, 0)
        right_layout.addWidget(control_group, 0)

        top_splitter = QSplitter(Qt.Horizontal)
        top_splitter.addWidget(left_panel)
        top_splitter.addWidget(right_panel)
        top_splitter.setStretchFactor(1, 1)

        self.log_panel = QTextEdit(self)
        self.log_panel.setReadOnly(True)
        self.log_panel.setFixedHeight(180)

        main_splitter = QSplitter(Qt.Vertical)
        main_splitter.addWidget(top_splitter)
        main_splitter.addWidget(self.log_panel)
        main_splitter.setStretchFactor(0, 1)

        layout.addWidget(main_splitter)
        self.setCentralWidget(central)

        self.start_btn.clicked.connect(self.start_scan)
        self.pause_btn.clicked.connect(self.pause_scan)
        self.resume_btn.clicked.connect(self.resume_scan)
        self.stop_btn.clicked.connect(self.stop_scan)
        self.optimize_btn.clicked.connect(self.optimize_scan)
        self.export_btn.clicked.connect(self.export_result)
        self.save_session_btn.clicked.connect(self.save_session)
        self.load_session_btn.clicked.connect(self.load_session)
        self.view_combo.currentTextChanged.connect(self._on_view_mode)
        self.reset_camera_btn.clicked.connect(self.viewer.reset_camera)
        self.toggle_axes_btn.clicked.connect(self.viewer.toggle_axes)

    @Slot()
    def start_scan(self) -> None:
        """
        启动采集与重建线程并创建会话。
        :return: 返回介绍
        """
        if self._capture_thread is not None:
            self.append_log("扫描已在运行")
            return

        self._apply_params()
        self._config["scan"]["mode"] = self.mode_combo.currentText()
        self._config["device"]["bag_path"] = self.bag_path_edit.text().strip()
        if self._config["scan"]["mode"] == "offline":
            if not self._config["device"]["bag_path"]:
                self.append_log("离线模式需要设置 bag 路径")
                return
            self._config["device"]["playback_bag"] = True
            self._config["device"]["record_bag"] = False
            self._config["device"]["playback_real_time"] = False
        else:
            self._config["device"]["playback_bag"] = False
            self._config["device"]["record_bag"] = self.record_checkbox.isChecked()
            if self._config["device"]["record_bag"] and not self._config["device"]["bag_path"]:
                self.append_log("录制模式需要设置 bag 路径")
                return

        self._queue = FrameQueue(maxsize=int(self._config.get("scan", {}).get("max_queue", 30)))
        device = RealSenseDevice(self._config.get("device", {}))
        self._capture_thread = CaptureThread(device, self._config)
        self._reconstruct_thread = ReconstructThread(self._queue, self._config)

        self._capture_thread.frame_signal.connect(self._on_frame)
        self._capture_thread.info_signal.connect(self._on_info)
        self._capture_thread.log_signal.connect(self.append_log)
        self._capture_thread.error_signal.connect(self.append_log)

        self._reconstruct_thread.preview_signal.connect(self._on_preview)
        self._reconstruct_thread.preview_mesh_signal.connect(self._on_mesh_preview)
        self._reconstruct_thread.status_signal.connect(self._on_status)
        self._reconstruct_thread.log_signal.connect(self.append_log)

        self._capture_thread.start()
        self._reconstruct_thread.start()

        self._start_session()
        self._set_state(ScannerState.PREVIEW)
        self.append_log("扫描启动")

    @Slot()
    def pause_scan(self) -> None:
        """
        暂停扫描与融合。
        :return: 返回介绍
        """
        if self._reconstruct_thread is not None:
            self._reconstruct_thread.pause()
            self._set_state(ScannerState.PAUSED)
            self.append_log("已暂停")

    @Slot()
    def resume_scan(self) -> None:
        """
        从暂停状态恢复扫描。
        :return: 返回介绍
        """
        if self._reconstruct_thread is not None:
            self._reconstruct_thread.resume()
            self._set_state(ScannerState.SCANNING)
            self.append_log("已继续")

    @Slot()
    def stop_scan(self) -> None:
        """
        停止扫描线程并保存会话元数据。
        :return: 返回介绍
        """
        reconstruct_thread = self._reconstruct_thread
        if self._capture_thread:
            self._capture_thread.stop()
            self._capture_thread.wait()
            self._capture_thread = None
        if reconstruct_thread:
            reconstruct_thread.stop()
            reconstruct_thread.wait()
            self._reconstruct_thread = None
        if self._queue:
            self._queue.clear()
        self.append_log("扫描已停止")
        self._finalize_session(reconstruct_thread)
        self._set_state(ScannerState.IDLE)

    @Slot()
    def optimize_scan(self) -> None:
        """
        触发回环检测与位姿图全局优化。
        :return: 返回介绍
        """
        if self._reconstruct_thread is None:
            self.append_log("请先完成扫描")
            return
        self._set_state(ScannerState.OPTIMIZING)
        keyframes = self._reconstruct_thread.get_keyframes()
        self._optimize_thread = OptimizeThread(keyframes, self._config)
        self._optimize_thread.mesh_signal.connect(self._on_mesh)
        self._optimize_thread.log_signal.connect(self.append_log)
        self._optimize_thread.start()

    @Slot()
    def export_result(self) -> None:
        """
        导出当前网格与点云结果。
        :return: 返回介绍
        """
        if self._session is None:
            self.append_log("没有会话数据")
            return

        self._set_state(ScannerState.EXPORTING)
        export_dir = ensure_dir(self._session.root / "exports")
        fmt = self.export_combo.currentText()
        mesh = self._current_mesh
        pcd = self._current_pcd
        if mesh is not None:
            mesh = postprocess_mesh(
                mesh,
                simplify_target=int(self._config.get("mesh", {}).get("simplify_target_triangles", 200000)),
                smooth_iterations=int(self._config.get("mesh", {}).get("smooth_iterations", 5)),
                remove_small=bool(self._config.get("mesh", {}).get("remove_small_components", True)),
                min_triangles=int(self._config.get("mesh", {}).get("min_triangles", 500)),
                keep_largest=bool(self._config.get("mesh", {}).get("keep_largest", True)),
            )
            path = export_mesh(mesh, export_dir / "mesh", fmt)
            self._session.exports.append(
                {
                    "type": "mesh",
                    "path": str(path),
                    "format": fmt,
                    "params": {
                        "simplify_target": self._config.get("mesh", {}).get("simplify_target_triangles"),
                        "smooth_iterations": self._config.get("mesh", {}).get("smooth_iterations"),
                        "remove_small": self._config.get("mesh", {}).get("remove_small_components"),
                        "min_triangles": self._config.get("mesh", {}).get("min_triangles"),
                        "keep_largest": self._config.get("mesh", {}).get("keep_largest"),
                    },
                }
            )
            self.append_log(f"网格已导出: {path}")

        if pcd is not None:
            cloud_fmt = self.cloud_export_combo.currentText()
            path = export_point_cloud(pcd, export_dir / "cloud", cloud_fmt)
            self._session.exports.append(
                {"type": "cloud", "path": str(path), "format": cloud_fmt, "params": {}}
            )
            self.append_log(f"点云已导出: {path}")

        if self._session is not None:
            self._session_manager.save_session(self._session)
        self._set_state(ScannerState.SCANNING if self._capture_thread else ScannerState.IDLE)

    @Slot(object)
    def _on_frame(self, frame: FrameBundle) -> None:
        """
        函数介绍。
        :param frame: 参数介绍
        :return: 返回介绍
        """
        if self._queue is not None:
            self._queue.put(frame)
        self._update_preview(frame)

    @Slot(dict)
    def _on_info(self, info: Dict[str, Any]) -> None:
        """
        函数介绍。
        :param info: 参数介绍
        :return: 返回介绍
        """
        fps = info.get("fps", 0.0)
        index = info.get("index", 0)
        self.fps_label.setText(f"FPS: {fps:.1f} | Frame: {index}")

    @Slot(object)
    def _on_preview(self, pcd: o3d.geometry.PointCloud) -> None:
        """
        函数介绍。
        :param pcd: 参数介绍
        :return: 返回介绍
        """
        self._current_pcd = pcd
        if self.view_combo.currentText() == "pointcloud":
            self.viewer.show_point_cloud(pcd)

    @Slot(str)
    def _on_status(self, status: str) -> None:
        """
        函数介绍。
        :param status: 参数介绍
        :return: 返回介绍
        """
        self.status_label.setText(f"Tracking: {status}")
        if status == "LOST":
            self._set_state(ScannerState.LOST)
        elif status in ("OK", "WARN") and self._state not in (ScannerState.OPTIMIZING, ScannerState.EXPORTING):
            self._set_state(ScannerState.SCANNING)

    @Slot(object)
    def _on_mesh(self, mesh: o3d.geometry.TriangleMesh) -> None:
        """
        函数介绍。
        :param mesh: 参数介绍
        :return: 返回介绍
        """
        self._current_mesh = mesh
        if self.view_combo.currentText() == "mesh":
            self.viewer.show_mesh(mesh)
        self._set_state(ScannerState.SCANNING)

    @Slot(object)
    def _on_mesh_preview(self, mesh: o3d.geometry.TriangleMesh) -> None:
        """
        函数介绍。
        :param mesh: 参数介绍
        :return: 返回介绍
        """
        self._current_mesh = mesh
        if self.view_combo.currentText() == "mesh":
            self.viewer.show_mesh(mesh)

    def _start_session(self) -> None:
        """
        创建会话目录并初始化日志。
        :return: 返回介绍
        """
        self._session = self._session_manager.create_session(self._config)
        log_path = self._session.root / "logs" / "session.log"
        self._logger = setup_logger(log_path)
        self._session.logs_path = str(log_path)
        if self._config.get("device", {}).get("record_bag") or self._config.get("device", {}).get("playback_bag"):
            self._session.bag_path = self._config.get("device", {}).get("bag_path")
        self.append_log(f"会话创建: {self._session.session_id}")

    def _finalize_session(self, reconstruct_thread: Optional[ReconstructThread]) -> None:
        """
        保存关键帧索引与轨迹到会话。
        :param reconstruct_thread: 参数介绍
        :return: 返回介绍
        """
        if self._session is None or reconstruct_thread is None:
            return
        keyframes = reconstruct_thread.get_keyframes()
        trajectory = reconstruct_thread.get_trajectory()
        self._session.keyframe_indices = [kf.index for kf in keyframes]
        self._session.trajectory = trajectory
        self._session_manager.save_session(self._session)

    def _update_preview(self, frame: FrameBundle) -> None:
        """
        函数介绍。
        :param frame: 参数介绍
        :return: 返回介绍
        """
        color_img = self._to_qimage(frame.color_rgb)
        depth_img = self._depth_to_colormap(frame.depth_mm)
        self.color_label.setPixmap(QPixmap.fromImage(color_img).scaled(320, 240, Qt.KeepAspectRatio))
        self.depth_label.setPixmap(QPixmap.fromImage(depth_img).scaled(320, 240, Qt.KeepAspectRatio))
        intr = frame.intrinsics
        self.info_label.setText(
            f"内参 fx={intr.fx:.1f} fy={intr.fy:.1f} cx={intr.cx:.1f} cy={intr.cy:.1f}"
        )

    def _to_qimage(self, rgb: np.ndarray) -> QImage:
        """
        函数介绍。
        :param rgb: 参数介绍
        :return: 返回介绍
        """
        rgb_img = rgb.copy()
        h, w, _ = rgb_img.shape
        return QImage(rgb_img.data, w, h, 3 * w, QImage.Format_RGB888).copy()

    def _depth_to_colormap(self, depth: np.ndarray) -> QImage:
        """
        函数介绍。
        :param depth: 参数介绍
        :return: 返回介绍
        """
        depth_m = depth.astype(np.float32) / 1000.0
        depth_trunc = float(self._config.get("fusion", {}).get("depth_trunc", 1.0))
        depth_m = np.clip(depth_m, 0.0, depth_trunc)
        if depth_m.max() > 0:
            depth_norm = depth_m / (depth_trunc + 1e-6)
        else:
            depth_norm = depth_m
        depth_uint8 = (depth_norm * 255).astype(np.uint8)
        if cv2 is not None:
            colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
        else:
            colored = np.stack([depth_uint8, depth_uint8, depth_uint8], axis=-1)
        h, w, _ = colored.shape
        return QImage(colored.data, w, h, 3 * w, QImage.Format_BGR888).copy()

    def append_log(self, message: str) -> None:
        """
        函数介绍。
        :param message: 参数介绍
        :return: 返回介绍
        """
        self.log_panel.append(message)
        max_lines = int(self._config.get("ui", {}).get("log_max_lines", 500))
        doc = self.log_panel.document()
        if doc.blockCount() > max_lines:
            cursor = self.log_panel.textCursor()
            cursor.movePosition(cursor.Start)
            cursor.select(cursor.LineUnderCursor)
            cursor.removeSelectedText()
            cursor.deleteChar()
        if self._logger is not None:
            self._logger.info(message)

    @Slot()
    def save_session(self) -> None:
        """
        函数介绍。
        :return: 返回介绍
        """
        if self._session is None:
            self.append_log("没有可保存的会话")
            return
        self._session_manager.save_session(self._session)
        self.append_log("会话已保存")

    @Slot()
    def load_session(self) -> None:
        """
        函数介绍。
        :return: 返回介绍
        """
        from PySide6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getOpenFileName(self, "选择会话文件", "", "Session (*.json)")
        if not path:
            return
        session = self._session_manager.load_session(Path(path))
        self._session = session
        self.append_log(f"会话已加载: {session.session_id}")
        self._set_state(ScannerState.IDLE)
        shown = False
        for item in session.exports:
            if item.get("type") == "mesh" and Path(item.get("path", "")).exists():
                mesh = o3d.io.read_triangle_mesh(item["path"])
                self._current_mesh = mesh
                self.viewer.show_mesh(mesh)
                shown = True
                break
        if not shown:
            for item in session.exports:
                if item.get("type") == "cloud" and Path(item.get("path", "")).exists():
                    pcd = o3d.io.read_point_cloud(item["path"])
                    self._current_pcd = pcd
                    self.viewer.show_point_cloud(pcd)
                    break

    def _apply_params(self) -> None:
        """
        函数介绍。
        :return: 返回介绍
        """
        self._config["fusion"]["voxel_length"] = float(self.voxel_spin.value())
        self._config["fusion"]["sdf_trunc"] = float(self.sdf_spin.value())
        self._config["fusion"]["depth_trunc"] = float(self.depth_trunc_spin.value())

        self._config["keyframe"]["min_translation"] = float(self.key_trans_spin.value())
        self._config["keyframe"]["min_rotation_deg"] = float(self.key_rot_spin.value())

        self._config["loop_closure"]["temporal_gap"] = int(self.loop_gap_spin.value())
        self._config["loop_closure"]["max_distance"] = float(self.loop_dist_spin.value())
        self._config["loop_closure"]["max_candidates"] = int(self.loop_candidates_spin.value())

        self._config["mesh"]["simplify_target_triangles"] = int(self.simplify_spin.value())
        self._config["mesh"]["smooth_iterations"] = int(self.smooth_spin.value())
        self._config["fusion"]["preview_mesh"] = bool(self.preview_mesh_checkbox.isChecked())

    def _set_state(self, state: ScannerState) -> None:
        """
        函数介绍。
        :param state: 参数介绍
        :return: 返回介绍
        """
        self._state = state
        self.state_label.setText(f"State: {state.value}")

    def _on_view_mode(self, mode: str) -> None:
        """
        函数介绍。
        :param mode: 参数介绍
        :return: 返回介绍
        """
        if mode == "mesh" and self._current_mesh is not None:
            self.viewer.show_mesh(self._current_mesh)
        elif mode == "pointcloud" and self._current_pcd is not None:
            self.viewer.show_point_cloud(self._current_pcd)
