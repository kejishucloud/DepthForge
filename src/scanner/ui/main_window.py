from __future__ import annotations

from pathlib import Path
import logging
import os
import threading
import traceback
from typing import Any, Dict, Optional, TYPE_CHECKING

import numpy as np
from PySide6.QtCore import Qt, Slot, QSignalBlocker, QTimer
from PySide6.QtGui import QImage, QPixmap, QOffscreenSurface, QOpenGLContext, QSurfaceFormat, QTextCursor
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QLabel,
    QApplication,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QCheckBox,
    QDoubleSpinBox,
    QSpinBox,
    QStackedWidget,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QToolBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QFrame,
)

from scanner.core.queues import FrameQueue
from scanner.core.state import ScannerState
from scanner.core.types import FrameBundle, SessionInfo
from scanner.io.session import SessionManager
from scanner.io.logger import setup_logger
from scanner.utils.path import ensure_dir

if TYPE_CHECKING:  # pragma: no cover
    import open3d as o3d

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


def _require_open3d():
    import open3d as o3d
    return o3d


class MainWindow(QMainWindow):
    def __init__(
        self,
        config: Dict[str, Any],
        config_path: Path,
        *,
        stop_event: Optional[threading.Event] = None,
    ) -> None:
        super().__init__()
        self._config = config
        self._config_path = config_path
        self._stop_event = stop_event
        self._lang = str(config.get("ui", {}).get("language", "zh-CN"))
        self._translations = self._build_translations()
        self._i18n_refs: Dict[str, list[QWidget]] = {}
        self._defer_ui = bool(config.get("ui", {}).get("defer_ui", True))
        self._enable_3d_viewer = bool(config.get("ui", {}).get("enable_3d_viewer", True))
        if os.environ.get("DEPTHFORGE_NO_3D", "0") == "1":
            self._enable_3d_viewer = False
        if os.environ.get("DEPTHFORGE_SAFE_MODE", "0") == "1":
            self._enable_3d_viewer = False
        self._session: Optional[SessionInfo] = None
        self._session_manager = SessionManager(Path(config.get("session_root", "sessions")))
        self._logger = None
        self._bootstrap_logger = logging.getLogger("scanner")

        self._capture_thread: Optional[CaptureThread] = None
        self._reconstruct_thread: Optional[ReconstructThread] = None
        self._optimize_thread: Optional[OptimizeThread] = None
        self._queue: Optional[FrameQueue[FrameBundle]] = None
        self._state = ScannerState.IDLE

        self._current_mesh: Optional[o3d.geometry.TriangleMesh] = None
        self._current_pcd: Optional[o3d.geometry.PointCloud] = None
        self._export_path_override: Optional[Path] = None
        self._log_entries: list[tuple[str, str]] = []
        self._last_fps = 0.0
        self._device_connected = False

        self.setWindowTitle("RealSense D435 手持 3D 扫描建模")
        self.resize(1400, 820)
        if self._defer_ui:
            placeholder = QWidget(self)
            placeholder_layout = QVBoxLayout(placeholder)
            placeholder_layout.setContentsMargins(16, 12, 16, 12)
            placeholder_layout.addWidget(QLabel(self._t("app.loading")))
            self.setCentralWidget(placeholder)
            QTimer.singleShot(0, self._finish_ui)
        else:
            self._build_ui()
            self._set_state(ScannerState.IDLE)

    def _build_ui(self) -> None:
        central = QWidget(self)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(16, 12, 16, 12)
        root_layout.setSpacing(12)

        header = QFrame(self)
        header.setObjectName("Header")
        header.setFixedHeight(48)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(12, 0, 12, 0)
        header_layout.setSpacing(12)

        header_left = QWidget(header)
        header_left_layout = QHBoxLayout(header_left)
        header_left_layout.setContentsMargins(0, 0, 0, 0)
        self.app_title_label = QLabel(self._t("app.title"))
        self.app_title_label.setObjectName("AppTitle")
        header_left_layout.addWidget(self.app_title_label)

        header_center = QWidget(header)
        header_center_layout = QHBoxLayout(header_center)
        header_center_layout.setContentsMargins(0, 0, 0, 0)
        header_center_layout.setSpacing(10)
        self.device_chip = QLabel(self._t("header.device.disconnected"))
        self.device_chip.setObjectName("DeviceChip")
        self.fps_label = QLabel(self._t("header.fps") + ": --")
        self.fps_label.setObjectName("HeaderFPS")
        header_center_layout.addWidget(self.device_chip)
        header_center_layout.addWidget(self.fps_label)

        header_right = QWidget(header)
        header_right_layout = QHBoxLayout(header_right)
        header_right_layout.setContentsMargins(0, 0, 0, 0)
        header_right_layout.setSpacing(8)
        self.lang_group = QButtonGroup(self)
        self.lang_group.setExclusive(True)
        self.lang_zh_btn = QToolButton(self)
        self.lang_zh_btn.setCheckable(True)
        self.lang_zh_btn.setText(self._t("header.lang.zh"))
        self.lang_en_btn = QToolButton(self)
        self.lang_en_btn.setCheckable(True)
        self.lang_en_btn.setText(self._t("header.lang.en"))
        self.lang_group.addButton(self.lang_zh_btn)
        self.lang_group.addButton(self.lang_en_btn)
        if self._lang.startswith("en"):
            self.lang_en_btn.setChecked(True)
        else:
            self.lang_zh_btn.setChecked(True)
        header_right_layout.addWidget(self.lang_zh_btn)
        header_right_layout.addWidget(self.lang_en_btn)
        self.opengl_diag_btn = QToolButton(self)
        self.opengl_diag_btn.setText(self._t("opengl.button"))
        header_right_layout.addWidget(self.opengl_diag_btn)
        self.settings_btn = QToolButton(self)
        self.settings_btn.setText(self._t("header.settings"))
        header_right_layout.addWidget(self.settings_btn)
        self.exit_btn = QToolButton(self)
        self.exit_btn.setText(self._t("header.exit"))
        header_right_layout.addWidget(self.exit_btn)

        header_layout.addWidget(header_left)
        header_layout.addStretch(1)
        header_layout.addWidget(header_center)
        header_layout.addStretch(1)
        header_layout.addWidget(header_right)

        self.tracking_banner = QFrame(self)
        self.tracking_banner.setObjectName("TrackingBanner")
        banner_layout = QHBoxLayout(self.tracking_banner)
        banner_layout.setContentsMargins(12, 6, 12, 6)
        self.tracking_banner_label = QLabel(self._t("banner.tracking_lost.body"))
        self.tracking_banner_action = QToolButton(self)
        self.tracking_banner_action.setText(self._t("banner.tracking_lost.action"))
        banner_layout.addWidget(self.tracking_banner_label)
        banner_layout.addStretch(1)
        banner_layout.addWidget(self.tracking_banner_action)
        self.tracking_banner.setVisible(False)

        body = QSplitter(Qt.Horizontal, self)
        body.setChildrenCollapsible(False)
        body.setHandleWidth(6)

        left_panel = QWidget(self)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)

        self.rgb_card = self._build_preview_card(self._t("preview.rgb.title"), kind="rgb")
        self.depth_card = self._build_preview_card(self._t("preview.depth.title"), kind="depth")
        left_layout.addWidget(self.rgb_card)
        left_layout.addWidget(self.depth_card)

        right_panel = QWidget(self)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)

        workflow_card = QFrame(self)
        workflow_card.setObjectName("WorkflowCard")
        workflow_layout = QVBoxLayout(workflow_card)
        workflow_layout.setContentsMargins(12, 12, 12, 12)
        workflow_layout.setSpacing(8)

        actions_row = QHBoxLayout()
        self.start_btn = QPushButton(self._t("workflow.primary.start"))
        self.start_btn.setObjectName("PrimaryButton")
        self.pause_btn = QPushButton(self._t("workflow.secondary.pause"))
        self.pause_btn.setObjectName("SecondaryButton")
        actions_row.addWidget(self.start_btn)
        actions_row.addWidget(self.pause_btn)
        actions_row.addStretch(1)

        mode_row = QHBoxLayout()
        self.mode_combo = QComboBox()
        self.mode_combo_io = QComboBox()
        self._populate_mode_combos(self._config.get("scan", {}).get("mode", "realtime"))
        mode_row.addWidget(QLabel(self._t("param.mode")))
        mode_row.addWidget(self.mode_combo, 1)

        status_row = QHBoxLayout()
        self.state_label = QLabel(self._t("workflow.state.idle"))
        self.state_label.setObjectName("StateLabel")
        self.tracking_pill = QLabel(self._t("workflow.tracking.ok"))
        self.tracking_pill.setObjectName("TrackingPill")
        self.tracking_pill.setProperty("state", "ok")
        status_row.addWidget(self.state_label)
        status_row.addStretch(1)
        status_row.addWidget(self.tracking_pill)

        workflow_layout.addLayout(actions_row)
        workflow_layout.addLayout(mode_row)
        workflow_layout.addLayout(status_row)
        self.info_label = QLabel("")
        self.info_label.setObjectName("InfoLabel")
        workflow_layout.addWidget(self.info_label)

        self.tabs = QTabWidget(self)

        params_tab = QWidget(self)
        params_layout = QVBoxLayout(params_tab)
        params_layout.setContentsMargins(8, 8, 8, 8)
        params_layout.setSpacing(8)

        basic_adv_row = QHBoxLayout()
        basic_adv_row.addWidget(QLabel(self._t("tabs.parameters")))
        basic_adv_row.addStretch(1)
        self.basic_adv_group = QButtonGroup(self)
        self.basic_adv_group.setExclusive(True)
        self.basic_btn = QToolButton(self)
        self.basic_btn.setCheckable(True)
        self.basic_btn.setText(self._t("toggle.basic"))
        self.advanced_btn = QToolButton(self)
        self.advanced_btn.setCheckable(True)
        self.advanced_btn.setText(self._t("toggle.advanced"))
        self.basic_adv_group.addButton(self.basic_btn)
        self.basic_adv_group.addButton(self.advanced_btn)
        self.basic_btn.setChecked(True)
        basic_adv_row.addWidget(self.basic_btn)
        basic_adv_row.addWidget(self.advanced_btn)
        params_layout.addLayout(basic_adv_row)

        self.params_toolbox = QToolBox(self)

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

        self.preview_mesh_checkbox = QCheckBox(self._t("param.preview_mesh"))
        self.preview_mesh_checkbox.setChecked(bool(self._config.get("fusion", {}).get("preview_mesh", False)))

        self.view_combo = QComboBox()
        default_view = self._resolve_default_view()
        self._populate_view_combo(default_view)
        self.reset_camera_btn = QPushButton(self._t("action.reset_camera"))
        self.toggle_axes_btn = QPushButton(self._t("action.toggle_axis"))

        self.bag_path_edit = QLineEdit()
        self.bag_path_edit.setPlaceholderText(self._t("param.bag_path"))
        self.bag_path_edit.setText(self._config.get("device", {}).get("bag_path", ""))
        self.browse_bag_btn = QPushButton(self._t("action.browse"))
        self.record_checkbox = QCheckBox(self._t("workflow.mode.record"))
        self.record_checkbox.setChecked(bool(self._config.get("device", {}).get("record_bag", False)))

        self.export_combo = QComboBox()
        self.export_combo.addItems(["ply", "obj", "stl"])
        self.export_combo.setCurrentText(self._config.get("export", {}).get("default_mesh_format", "ply"))
        self.cloud_export_combo = QComboBox()
        self.cloud_export_combo.addItems(["ply", "pcd"])
        self.cloud_export_combo.setCurrentText(self._config.get("export", {}).get("default_cloud_format", "ply"))

        def unit_widget(widget: QWidget, unit_text: str) -> QWidget:
            box = QWidget(self)
            box_layout = QHBoxLayout(box)
            box_layout.setContentsMargins(0, 0, 0, 0)
            box_layout.setSpacing(6)
            unit_label = QLabel(unit_text)
            unit_label.setObjectName("UnitLabel")
            box_layout.addWidget(widget)
            box_layout.addWidget(unit_label)
            box_layout.addStretch(1)
            return box

        # Reconstruction group
        recon_widget = QWidget(self)
        recon_layout = QFormLayout(recon_widget)
        self.label_voxel = QLabel(self._t("param.voxel"))
        self.label_sdf = QLabel(self._t("param.sdf"))
        self.label_depth_max = QLabel(self._t("param.depth_max"))
        self.voxel_widget = unit_widget(self.voxel_spin, "m")
        self.sdf_widget = unit_widget(self.sdf_spin, "m")
        self.depth_widget = unit_widget(self.depth_trunc_spin, "m")
        recon_layout.addRow(self.label_voxel, self.voxel_widget)
        recon_layout.addRow(self.label_sdf, self.sdf_widget)
        recon_layout.addRow(self.label_depth_max, self.depth_widget)
        self.params_toolbox.addItem(recon_widget, self._t("group.reconstruction"))

        # Tracking group
        tracking_widget = QWidget(self)
        tracking_layout = QFormLayout(tracking_widget)
        self.label_keyframe_dt = QLabel(self._t("param.keyframe_dt"))
        self.label_keyframe_dr = QLabel(self._t("param.keyframe_dr"))
        self.label_tracking_state = QLabel(self._t("param.tracking_state"))
        self.key_dt_widget = unit_widget(self.key_trans_spin, "m")
        self.key_dr_widget = unit_widget(self.key_rot_spin, "°")
        tracking_layout.addRow(self.label_keyframe_dt, self.key_dt_widget)
        tracking_layout.addRow(self.label_keyframe_dr, self.key_dr_widget)
        self.tracking_status_label = QLabel(self._t("workflow.tracking.ok"))
        tracking_layout.addRow(self.label_tracking_state, self.tracking_status_label)
        self.params_toolbox.addItem(tracking_widget, self._t("group.tracking"))

        # Loop Closure group
        loop_widget = QWidget(self)
        loop_layout = QFormLayout(loop_widget)
        self.label_loop_gap = QLabel(self._t("param.loop_gap"))
        self.label_loop_dist = QLabel(self._t("param.loop_dist"))
        self.label_loop_cand = QLabel(self._t("param.loop_cand"))
        self.loop_dist_widget = unit_widget(self.loop_dist_spin, "m")
        loop_layout.addRow(self.label_loop_gap, self.loop_gap_spin)
        loop_layout.addRow(self.label_loop_dist, self.loop_dist_widget)
        loop_layout.addRow(self.label_loop_cand, self.loop_candidates_spin)
        self.params_toolbox.addItem(loop_widget, self._t("group.loop"))

        # Meshing group
        mesh_widget = QWidget(self)
        mesh_layout = QFormLayout(mesh_widget)
        self.label_mesh_tri = QLabel(self._t("param.mesh_tri"))
        self.label_mesh_smooth = QLabel(self._t("param.mesh_smooth"))
        mesh_layout.addRow(self.label_mesh_tri, self.simplify_spin)
        mesh_layout.addRow(self.label_mesh_smooth, self.smooth_spin)
        mesh_layout.addRow(self.preview_mesh_checkbox)
        self.params_toolbox.addItem(mesh_widget, self._t("group.mesh"))

        # View group
        view_widget = QWidget(self)
        view_layout = QFormLayout(view_widget)
        self.label_view_mode = QLabel(self._t("param.view_mode"))
        view_layout.addRow(self.label_view_mode, self.view_combo)
        view_layout.addRow(self.reset_camera_btn, self.toggle_axes_btn)
        self.params_toolbox.addItem(view_widget, self._t("group.view"))

        # IO group
        io_widget = QWidget(self)
        io_layout = QFormLayout(io_widget)
        bag_row = QWidget(self)
        bag_layout = QHBoxLayout(bag_row)
        bag_layout.setContentsMargins(0, 0, 0, 0)
        bag_layout.setSpacing(6)
        bag_layout.addWidget(self.bag_path_edit)
        bag_layout.addWidget(self.browse_bag_btn)
        self.label_mode = QLabel(self._t("param.mode"))
        self.label_bag_path = QLabel(self._t("param.bag_path"))
        self.label_mesh_format = QLabel(self._t("param.mesh_format"))
        self.label_cloud_format = QLabel(self._t("param.cloud_format"))
        io_layout.addRow(self.label_mode, self.mode_combo_io)
        io_layout.addRow(self.label_bag_path, bag_row)
        io_layout.addRow(self.record_checkbox)
        io_layout.addRow(self.label_mesh_format, self.export_combo)
        io_layout.addRow(self.label_cloud_format, self.cloud_export_combo)
        self.params_toolbox.addItem(io_widget, self._t("group.io"))

        self._advanced_widgets = [
            self.label_sdf,
            self.sdf_widget,
            self.label_loop_dist,
            self.loop_dist_widget,
            self.label_loop_cand,
            self.loop_candidates_spin,
            self.label_mesh_smooth,
            self.smooth_spin,
            self.preview_mesh_checkbox,
            self.toggle_axes_btn,
            self.label_cloud_format,
            self.cloud_export_combo,
        ]

        params_layout.addWidget(self.params_toolbox, 1)

        export_tab = QWidget(self)
        export_layout = QVBoxLayout(export_tab)
        export_layout.setContentsMargins(8, 8, 8, 8)
        export_layout.setSpacing(8)
        self.export_title_label = QLabel(self._t("export.title"))
        export_layout.addWidget(self.export_title_label)

        self.export_stack = QStackedWidget(self)
        export_layout.addWidget(self.export_stack, 1)

        self.export_type_combo = QComboBox()
        self._populate_export_type_combo("mesh")
        type_page = QWidget(self)
        type_layout = QFormLayout(type_page)
        self.label_export_type = QLabel(self._t("export.step.type"))
        type_layout.addRow(self.label_export_type, self.export_type_combo)
        self.export_stack.addWidget(type_page)

        self.export_format_combo = QComboBox()
        format_page = QWidget(self)
        format_layout = QFormLayout(format_page)
        self.label_export_format = QLabel(self._t("export.step.format"))
        format_layout.addRow(self.label_export_format, self.export_format_combo)
        self.export_stack.addWidget(format_page)

        options_page = QWidget(self)
        options_layout = QFormLayout(options_page)
        self.export_simplify_check = QCheckBox(self._t("export.option.simplify"))
        self.export_smooth_check = QCheckBox(self._t("export.option.smooth"))
        self.export_simplify_check.setChecked(True)
        self.export_smooth_check.setChecked(True)
        options_layout.addRow(self.export_simplify_check)
        options_layout.addRow(self.export_smooth_check)
        self.export_stack.addWidget(options_page)

        path_page = QWidget(self)
        path_layout = QFormLayout(path_page)
        self.export_path_edit = QLineEdit()
        self.export_path_edit.setPlaceholderText(self._t("export.step.path"))
        self.export_path_btn = QPushButton(self._t("action.browse"))
        path_row = QWidget(self)
        path_row_layout = QHBoxLayout(path_row)
        path_row_layout.setContentsMargins(0, 0, 0, 0)
        path_row_layout.setSpacing(6)
        path_row_layout.addWidget(self.export_path_edit)
        path_row_layout.addWidget(self.export_path_btn)
        self.label_export_path = QLabel(self._t("export.step.path"))
        path_layout.addRow(self.label_export_path, path_row)
        self.export_stack.addWidget(path_page)

        export_page = QWidget(self)
        export_page_layout = QVBoxLayout(export_page)
        export_page_layout.addStretch(1)
        self.export_run_btn = QPushButton(self._t("export.action.export"))
        self.export_run_btn.setObjectName("PrimaryButton")
        export_page_layout.addWidget(self.export_run_btn)
        self.export_stack.addWidget(export_page)

        nav_row = QHBoxLayout()
        self.export_prev_btn = QPushButton("◀")
        self.export_next_btn = QPushButton("▶")
        nav_row.addWidget(self.export_prev_btn)
        nav_row.addWidget(self.export_next_btn)
        nav_row.addStretch(1)
        export_layout.addLayout(nav_row)

        session_tab = QWidget(self)
        session_layout = QVBoxLayout(session_tab)
        session_layout.setContentsMargins(8, 8, 8, 8)
        self.optimize_btn = QPushButton(self._t("workflow.optimize"))
        self.save_session_btn = QPushButton(self._t("session.save"))
        self.load_session_btn = QPushButton(self._t("session.load"))
        session_layout.addWidget(self.optimize_btn)
        session_layout.addWidget(self.save_session_btn)
        session_layout.addWidget(self.load_session_btn)
        session_layout.addStretch(1)

        self.tabs.addTab(params_tab, self._t("tabs.parameters"))
        self.tabs.addTab(export_tab, self._t("tabs.export"))
        self.tabs.addTab(session_tab, self._t("tabs.session"))

        right_layout.addWidget(workflow_card, 0)
        right_layout.addWidget(self.tabs, 1)

        body.addWidget(left_panel)
        body.addWidget(right_panel)
        body.setStretchFactor(0, 62)
        body.setStretchFactor(1, 38)

        self.log_panel = QFrame(self)
        self.log_panel.setObjectName("LogPanel")
        log_layout = QVBoxLayout(self.log_panel)
        log_layout.setContentsMargins(8, 8, 8, 8)
        log_layout.setSpacing(6)

        log_header = QHBoxLayout()
        self.label_log_title = QLabel(self._t("log.title"))
        log_header.addWidget(self.label_log_title)
        log_header.addStretch(1)
        self.log_info_chk = QCheckBox(self._t("log.filter.info"))
        self.log_warn_chk = QCheckBox(self._t("log.filter.warn"))
        self.log_error_chk = QCheckBox(self._t("log.filter.error"))
        self.log_info_chk.setChecked(True)
        self.log_warn_chk.setChecked(True)
        self.log_error_chk.setChecked(True)
        log_header.addWidget(self.log_info_chk)
        log_header.addWidget(self.log_warn_chk)
        log_header.addWidget(self.log_error_chk)
        self.log_copy_btn = QPushButton(self._t("log.copy"))
        self.log_clear_btn = QPushButton(self._t("log.clear"))
        self.log_autoscroll_chk = QCheckBox(self._t("log.autoscroll"))
        self.log_autoscroll_chk.setChecked(True)
        log_header.addWidget(self.log_copy_btn)
        log_header.addWidget(self.log_clear_btn)
        log_header.addWidget(self.log_autoscroll_chk)

        self.log_text = QTextEdit(self)
        self.log_text.setReadOnly(True)
        self.log_text.setFixedHeight(180)

        log_layout.addLayout(log_header)
        log_layout.addWidget(self.log_text)

        root_layout.addWidget(header)
        root_layout.addWidget(self.tracking_banner)
        root_layout.addWidget(body, 1)
        root_layout.addWidget(self.log_panel, 0)
        self.setCentralWidget(central)

        self.start_btn.clicked.connect(self._on_primary_clicked)
        self.pause_btn.clicked.connect(self._on_secondary_clicked)
        self.optimize_btn.clicked.connect(self.optimize_scan)
        self.export_run_btn.clicked.connect(self.export_result)
        self.export_prev_btn.clicked.connect(self._export_prev)
        self.export_next_btn.clicked.connect(self._export_next)
        self.export_type_combo.currentTextChanged.connect(self._update_export_formats)
        self.export_path_btn.clicked.connect(self._choose_export_path)
        self.save_session_btn.clicked.connect(self.save_session)
        self.load_session_btn.clicked.connect(self.load_session)
        self.view_combo.currentTextChanged.connect(self._on_view_mode)
        self.reset_camera_btn.clicked.connect(self._on_reset_camera)
        self.toggle_axes_btn.clicked.connect(self._on_toggle_axes)
        self.browse_bag_btn.clicked.connect(self._choose_bag_path)
        self.lang_zh_btn.clicked.connect(lambda: self._set_language("zh-CN"))
        self.lang_en_btn.clicked.connect(lambda: self._set_language("en-US"))
        self.opengl_diag_btn.clicked.connect(self._diagnose_opengl)
        self.exit_btn.clicked.connect(self._request_exit)
        self.basic_btn.clicked.connect(self._apply_basic_mode)
        self.advanced_btn.clicked.connect(self._apply_advanced_mode)
        self.log_clear_btn.clicked.connect(self._clear_log)
        self.log_copy_btn.clicked.connect(self._copy_log)
        self.log_info_chk.clicked.connect(self._refresh_log_view)
        self.log_warn_chk.clicked.connect(self._refresh_log_view)
        self.log_error_chk.clicked.connect(self._refresh_log_view)
        self.tracking_banner_action.clicked.connect(self._show_tracking_tips)
        self.mode_combo.currentTextChanged.connect(lambda text: self._sync_mode_combo(self.mode_combo, self.mode_combo_io, text))
        self.mode_combo_io.currentTextChanged.connect(lambda text: self._sync_mode_combo(self.mode_combo_io, self.mode_combo, text))

        self._update_export_formats()
        self._apply_basic_mode()
        self._apply_theme("dark")
        self._on_view_mode(self.view_combo.currentText())
        if not self._enable_3d_viewer:
            self.view_combo.setToolTip(self._t("view.disabled"))

    def _build_preview_card(self, title: str, kind: str) -> QFrame:
        card = QFrame(self)
        card.setObjectName("PreviewCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(6)

        header = QHBoxLayout()
        title_label = QLabel(title)
        title_label.setObjectName("PreviewTitle")
        meta_label = QLabel("--")
        meta_label.setObjectName("PreviewMeta")
        align_label = QLabel(self._t("preview.aligned") if self._config.get("device", {}).get("align_to_color", True) else self._t("preview.not_aligned"))
        align_label.setObjectName("PreviewAlign")
        header.addWidget(title_label)
        header.addStretch(1)
        header.addWidget(meta_label)
        header.addWidget(align_label)

        if kind == "rgb":
            self.rgb_title_label = title_label
            self.rgb_meta_label = meta_label
            self.rgb_align_label = align_label
            self.color_label = QLabel("RGB")
            self.color_label.setAlignment(Qt.AlignCenter)
            self.color_label.setObjectName("PreviewImage")
            self.color_label.setMinimumHeight(240)
            layout.addLayout(header)
            layout.addWidget(self.color_label, 1)
        else:
            self.depth_title_label = title_label
            self.depth_meta_label = meta_label
            self.depth_align_label = align_label
            self.depth_label = QLabel("Depth")
            self.depth_label.setAlignment(Qt.AlignCenter)
            self.depth_label.setObjectName("PreviewImage")
            self.depth_label.setMinimumHeight(240)
            self.viewer = None
            self.viewer_placeholder = QLabel(self._t("view.pointcloud"))
            self.viewer_placeholder.setAlignment(Qt.AlignCenter)
            self.viewer_placeholder.setObjectName("PreviewPlaceholder")
            self.depth_stack = QStackedWidget(self)
            self.depth_stack.addWidget(self.depth_label)
            self.depth_stack.addWidget(self.viewer_placeholder)
            layout.addLayout(header)
            layout.addWidget(self.depth_stack, 1)
        return card

    def _resolve_default_view(self) -> str:
        default_view = str(self._config.get("ui", {}).get("default_view", "rgbd")).lower()
        if default_view not in ("rgbd", "pointcloud", "mesh"):
            default_view = "rgbd"
        if not self._enable_3d_viewer and default_view in ("pointcloud", "mesh"):
            default_view = "rgbd"
        return default_view

    def _finish_ui(self) -> None:
        if getattr(self, "_ui_ready", False):
            return
        self._build_ui()
        self._set_state(ScannerState.IDLE)
        self._ui_ready = True

    def _build_translations(self) -> Dict[str, Dict[str, str]]:
        return {
            "zh-CN": {
                "app.title": "DepthForge 手持 3D 扫描",
                "app.loading": "正在初始化界面…",
                "header.device.connected": "已连接",
                "header.device.disconnected": "未检测到设备",
                "header.fps": "帧率",
                "header.lang.zh": "中文",
                "header.lang.en": "EN",
                "header.settings": "设置",
                "header.exit": "退出",
                "preview.rgb.title": "RGB",
                "preview.depth.title": "Depth",
                "preview.aligned": "已对齐",
                "preview.not_aligned": "未对齐",
                "workflow.primary.start": "开始",
                "workflow.primary.stop": "停止",
                "workflow.secondary.pause": "暂停",
                "workflow.secondary.resume": "继续",
                "workflow.optimize": "全局优化",
                "workflow.state.idle": "空闲",
                "workflow.state.running": "运行中",
                "workflow.state.paused": "已暂停",
                "workflow.state.playback": "回放中",
                "workflow.state.recording": "录制中",
                "workflow.state.exporting": "导出中",
                "workflow.tracking.ok": "跟踪正常",
                "workflow.tracking.warn": "跟踪警告",
                "workflow.tracking.lost": "跟踪丢失",
                "tabs.parameters": "参数",
                "tabs.export": "导出",
                "tabs.session": "会话",
                "toggle.basic": "基础",
                "toggle.advanced": "高级",
                "group.reconstruction": "重建",
                "group.tracking": "跟踪",
                "group.loop": "回环",
                "group.mesh": "网格",
                "group.view": "视图",
                "group.io": "输入输出",
                "param.voxel": "体素大小",
                "param.sdf": "TSDF 截断",
                "param.depth_max": "深度上限",
                "param.keyframe_dt": "关键帧间隔",
                "param.keyframe_dr": "关键帧旋转",
                "param.tracking_state": "跟踪状态",
                "param.loop_gap": "回环间隔",
                "param.loop_dist": "回环距离",
                "param.loop_cand": "候选数",
                "param.mesh_tri": "目标三角面数",
                "param.mesh_smooth": "平滑迭代",
                "param.preview_mesh": "预览网格",
                "param.view_mode": "视图模式",
                "param.mode": "模式",
                "mode.realtime": "实时",
                "mode.semi": "半实时",
                "mode.offline": "离线",
                "param.bag_path": "bag 路径",
                "param.mesh_format": "网格格式",
                "param.cloud_format": "点云格式",
                "action.reset_camera": "重置相机",
                "action.toggle_axis": "切换坐标轴",
                "action.browse": "浏览",
                "workflow.mode.record": "录制",
                "view.pointcloud": "点云",
                "view.mesh": "网格",
                "view.rgbd": "RGBD",
                "view.disabled": "当前系统不支持 OpenGL 3.2，已禁用 3D 视图",
                "export.title": "导出向导",
                "export.type.mesh": "网格",
                "export.type.pointcloud": "点云",
                "export.step.type": "类型",
                "export.step.format": "格式",
                "export.step.path": "保存路径",
                "export.option.simplify": "简化",
                "export.option.smooth": "平滑",
                "export.action.export": "导出",
                "session.save": "保存会话",
                "session.load": "加载会话",
                "log.title": "日志",
                "log.filter.info": "信息",
                "log.filter.warn": "警告",
                "log.filter.error": "错误",
                "log.copy": "复制",
                "log.clear": "清空",
                "log.autoscroll": "自动滚动",
                "banner.tracking_lost.body": "跟踪丢失：请减速、拉远、增加纹理或降低 depth_trunc",
                "banner.tracking_lost.action": "查看建议",
                "banner.tracking_lost.tips": "建议：\n- 减速并保持匀速\n- 增加相机与物体距离\n- 提升纹理（贴纸/报纸）\n- 适当降低 depth_trunc",
                "opengl.button": "OpenGL 诊断",
                "opengl.title": "OpenGL 诊断",
                "opengl.fail": "无法获取 OpenGL 信息",
            },
            "en-US": {
                "app.title": "DepthForge Handheld 3D Scanner",
                "app.loading": "Loading UI…",
                "header.device.connected": "Connected",
                "header.device.disconnected": "No device",
                "header.fps": "FPS",
                "header.lang.zh": "ZH",
                "header.lang.en": "EN",
                "header.settings": "Settings",
                "header.exit": "Exit",
                "preview.rgb.title": "RGB",
                "preview.depth.title": "Depth",
                "preview.aligned": "Aligned",
                "preview.not_aligned": "Not aligned",
                "workflow.primary.start": "Start",
                "workflow.primary.stop": "Stop",
                "workflow.secondary.pause": "Pause",
                "workflow.secondary.resume": "Resume",
                "workflow.optimize": "Global Optimize",
                "workflow.state.idle": "Idle",
                "workflow.state.running": "Running",
                "workflow.state.paused": "Paused",
                "workflow.state.playback": "Playback",
                "workflow.state.recording": "Recording",
                "workflow.state.exporting": "Exporting",
                "workflow.tracking.ok": "Tracking OK",
                "workflow.tracking.warn": "Tracking Warn",
                "workflow.tracking.lost": "Tracking Lost",
                "tabs.parameters": "Parameters",
                "tabs.export": "Export",
                "tabs.session": "Session",
                "toggle.basic": "Basic",
                "toggle.advanced": "Advanced",
                "group.reconstruction": "Reconstruction",
                "group.tracking": "Tracking",
                "group.loop": "Loop Closure",
                "group.mesh": "Meshing",
                "group.view": "View",
                "group.io": "IO",
                "param.voxel": "Voxel size",
                "param.sdf": "TSDF truncation",
                "param.depth_max": "Depth max",
                "param.keyframe_dt": "Keyframe interval",
                "param.keyframe_dr": "Keyframe rotation",
                "param.tracking_state": "Tracking status",
                "param.loop_gap": "Loop gap",
                "param.loop_dist": "Loop distance",
                "param.loop_cand": "Loop candidates",
                "param.mesh_tri": "Target triangles",
                "param.mesh_smooth": "Smooth iterations",
                "param.preview_mesh": "Preview mesh",
                "param.view_mode": "View mode",
                "param.mode": "Mode",
                "mode.realtime": "Realtime",
                "mode.semi": "Semi",
                "mode.offline": "Offline",
                "param.bag_path": "bag path",
                "param.mesh_format": "Mesh format",
                "param.cloud_format": "Cloud format",
                "action.reset_camera": "Reset camera",
                "action.toggle_axis": "Toggle axis",
                "action.browse": "Browse",
                "workflow.mode.record": "Record",
                "view.pointcloud": "Pointcloud",
                "view.mesh": "Mesh",
                "view.rgbd": "RGBD",
                "view.disabled": "OpenGL 3.2 is not available on this system. 3D view disabled.",
                "export.title": "Export Wizard",
                "export.type.mesh": "Mesh",
                "export.type.pointcloud": "Pointcloud",
                "export.step.type": "Type",
                "export.step.format": "Format",
                "export.step.path": "Save path",
                "export.option.simplify": "Simplify",
                "export.option.smooth": "Smooth",
                "export.action.export": "Export",
                "session.save": "Save session",
                "session.load": "Load session",
                "log.title": "Log",
                "log.filter.info": "Info",
                "log.filter.warn": "Warn",
                "log.filter.error": "Error",
                "log.copy": "Copy",
                "log.clear": "Clear",
                "log.autoscroll": "Auto-scroll",
                "banner.tracking_lost.body": "Tracking lost: slow down, increase distance, add texture, or reduce depth_trunc",
                "banner.tracking_lost.action": "View tips",
                "banner.tracking_lost.tips": "Tips:\n- Move slower and steadily\n- Increase distance to the object\n- Add texture (stickers/newspaper)\n- Reduce depth_trunc",
                "opengl.button": "OpenGL",
                "opengl.title": "OpenGL Diagnostics",
                "opengl.fail": "Failed to query OpenGL info",
            },
        }

    def _t(self, key: str) -> str:
        lang_pack = self._translations.get(self._lang, {})
        if key in lang_pack:
            return lang_pack[key]
        return self._translations.get("zh-CN", {}).get(key, key)

    def _set_language(self, lang: str) -> None:
        self._lang = lang
        self._apply_language()

    def _apply_language(self) -> None:
        self.app_title_label.setText(self._t("app.title"))
        self.device_chip.setText(
            self._t("header.device.connected") if self._device_connected else self._t("header.device.disconnected")
        )
        self.fps_label.setText(self._t("header.fps") + f": {self._last_fps:.1f}" if self._last_fps else self._t("header.fps") + ": --")
        self.lang_zh_btn.setText(self._t("header.lang.zh"))
        self.lang_en_btn.setText(self._t("header.lang.en"))
        self.opengl_diag_btn.setText(self._t("opengl.button"))
        self.settings_btn.setText(self._t("header.settings"))
        self.exit_btn.setText(self._t("header.exit"))
        self.tracking_banner_label.setText(self._t("banner.tracking_lost.body"))
        self.tracking_banner_action.setText(self._t("banner.tracking_lost.action"))
        self.start_btn.setText(self._t("workflow.primary.start") if self._state == ScannerState.IDLE else self._t("workflow.primary.stop"))
        self.pause_btn.setText(self._t("workflow.secondary.resume") if self._state == ScannerState.PAUSED else self._t("workflow.secondary.pause"))
        self.state_label.setText(self._state_text(self._state))
        status = getattr(self, "_last_tracking_status", "OK")
        self.tracking_pill.setText(self._tracking_text(status))
        self.tracking_status_label.setText(self.tracking_pill.text())
        self.basic_btn.setText(self._t("toggle.basic"))
        self.advanced_btn.setText(self._t("toggle.advanced"))
        self.label_voxel.setText(self._t("param.voxel"))
        self.label_sdf.setText(self._t("param.sdf"))
        self.label_depth_max.setText(self._t("param.depth_max"))
        self.label_keyframe_dt.setText(self._t("param.keyframe_dt"))
        self.label_keyframe_dr.setText(self._t("param.keyframe_dr"))
        self.label_tracking_state.setText(self._t("param.tracking_state"))
        self.label_loop_gap.setText(self._t("param.loop_gap"))
        self.label_loop_dist.setText(self._t("param.loop_dist"))
        self.label_loop_cand.setText(self._t("param.loop_cand"))
        self.label_mesh_tri.setText(self._t("param.mesh_tri"))
        self.label_mesh_smooth.setText(self._t("param.mesh_smooth"))
        self.preview_mesh_checkbox.setText(self._t("param.preview_mesh"))
        self.label_view_mode.setText(self._t("param.view_mode"))
        self.label_mode.setText(self._t("param.mode"))
        self.label_bag_path.setText(self._t("param.bag_path"))
        self.label_mesh_format.setText(self._t("param.mesh_format"))
        self.label_cloud_format.setText(self._t("param.cloud_format"))
        self.rgb_title_label.setText(self._t("preview.rgb.title"))
        self.depth_title_label.setText(self._t("preview.depth.title"))
        if hasattr(self, "viewer_placeholder"):
            self.viewer_placeholder.setText(self._t("view.pointcloud"))
        self.rgb_align_label.setText(self._t("preview.aligned") if self._config.get("device", {}).get("align_to_color", True) else self._t("preview.not_aligned"))
        self.depth_align_label.setText(self._t("preview.aligned") if self._config.get("device", {}).get("align_to_color", True) else self._t("preview.not_aligned"))
        self.reset_camera_btn.setText(self._t("action.reset_camera"))
        self.toggle_axes_btn.setText(self._t("action.toggle_axis"))
        self.browse_bag_btn.setText(self._t("action.browse"))
        self.record_checkbox.setText(self._t("workflow.mode.record"))
        self.export_title_label.setText(self._t("export.title"))
        self.label_export_type.setText(self._t("export.step.type"))
        self.label_export_format.setText(self._t("export.step.format"))
        self.label_export_path.setText(self._t("export.step.path"))
        self.export_simplify_check.setText(self._t("export.option.simplify"))
        self.export_smooth_check.setText(self._t("export.option.smooth"))
        self.export_run_btn.setText(self._t("export.action.export"))
        self.export_path_btn.setText(self._t("action.browse"))
        self.save_session_btn.setText(self._t("session.save"))
        self.load_session_btn.setText(self._t("session.load"))
        self.optimize_btn.setText(self._t("workflow.optimize"))
        self.label_log_title.setText(self._t("log.title"))
        self.log_info_chk.setText(self._t("log.filter.info"))
        self.log_warn_chk.setText(self._t("log.filter.warn"))
        self.log_error_chk.setText(self._t("log.filter.error"))
        self.log_copy_btn.setText(self._t("log.copy"))
        self.log_clear_btn.setText(self._t("log.clear"))
        self.log_autoscroll_chk.setText(self._t("log.autoscroll"))
        self.export_path_edit.setPlaceholderText(self._t("export.step.path"))
        self.bag_path_edit.setPlaceholderText(self._t("param.bag_path"))
        self._populate_mode_combos(self.mode_combo.currentData() or self.mode_combo.currentText())
        self._populate_view_combo(self.view_combo.currentData() or self.view_combo.currentText())
        self._populate_export_type_combo(self.export_type_combo.currentData() or self.export_type_combo.currentText())
        self._update_export_formats()
        if not self._enable_3d_viewer:
            self.view_combo.setToolTip(self._t("view.disabled"))
        self.params_toolbox.setItemText(0, self._t("group.reconstruction"))
        self.params_toolbox.setItemText(1, self._t("group.tracking"))
        self.params_toolbox.setItemText(2, self._t("group.loop"))
        self.params_toolbox.setItemText(3, self._t("group.mesh"))
        self.params_toolbox.setItemText(4, self._t("group.view"))
        self.params_toolbox.setItemText(5, self._t("group.io"))
        self.tabs.setTabText(0, self._t("tabs.parameters"))
        self.tabs.setTabText(1, self._t("tabs.export"))
        self.tabs.setTabText(2, self._t("tabs.session"))

    def _populate_mode_combos(self, current: str) -> None:
        for combo in (self.mode_combo, self.mode_combo_io):
            blocker = QSignalBlocker(combo)
            combo.clear()
            combo.addItem(self._t("mode.realtime"), "realtime")
            combo.addItem(self._t("mode.semi"), "semi")
            combo.addItem(self._t("mode.offline"), "offline")
            for index in range(combo.count()):
                if combo.itemData(index) == current:
                    combo.setCurrentIndex(index)
                    break
            del blocker

    def _populate_view_combo(self, current: str) -> None:
        blocker = QSignalBlocker(self.view_combo)
        self.view_combo.clear()
        self.view_combo.addItem(self._t("view.pointcloud"), "pointcloud")
        self.view_combo.addItem(self._t("view.mesh"), "mesh")
        self.view_combo.addItem(self._t("view.rgbd"), "rgbd")
        if not self._enable_3d_viewer:
            for index in range(self.view_combo.count()):
                if self.view_combo.itemData(index) in ("pointcloud", "mesh"):
                    self.view_combo.setItemData(index, 0, Qt.ItemDataRole.UserRole - 1)
        for index in range(self.view_combo.count()):
            if self.view_combo.itemData(index) == current:
                self.view_combo.setCurrentIndex(index)
                break
        if not self._enable_3d_viewer and (self.view_combo.currentData() in ("pointcloud", "mesh")):
            for index in range(self.view_combo.count()):
                if self.view_combo.itemData(index) == "rgbd":
                    self.view_combo.setCurrentIndex(index)
                    break
        del blocker

    def _populate_export_type_combo(self, current: str) -> None:
        blocker = QSignalBlocker(self.export_type_combo)
        self.export_type_combo.clear()
        self.export_type_combo.addItem(self._t("export.type.mesh"), "mesh")
        self.export_type_combo.addItem(self._t("export.type.pointcloud"), "pointcloud")
        for index in range(self.export_type_combo.count()):
            if self.export_type_combo.itemData(index) == current:
                self.export_type_combo.setCurrentIndex(index)
                break
        del blocker

    def _apply_theme(self, theme: str) -> None:
        if theme == "light":
            palette = {
                "bg": "#F6F7FB",
                "surface": "#FFFFFF",
                "surface_alt": "#F1F3F7",
                "card": "#FFFFFF",
                "border": "#DCE1E8",
                "text": "#1A1D24",
                "text_muted": "#5B6472",
                "accent": "#2B7CFF",
                "accent_text": "#FFFFFF",
                "warn": "#F6C453",
                "error": "#F26666",
            }
        else:
            palette = {
                "bg": "#0F1115",
                "surface": "#171A20",
                "surface_alt": "#1E232B",
                "card": "#1B1F26",
                "border": "#2A313B",
                "text": "#E9EDF2",
                "text_muted": "#A7B0BE",
                "accent": "#4AA3FF",
                "accent_text": "#0B0F14",
                "warn": "#F6C453",
                "error": "#F26666",
            }
        self.setStyleSheet(
            f"""
            QMainWindow {{
                background: {palette["bg"]};
                color: {palette["text"]};
            }}
            QLabel {{
                color: {palette["text"]};
            }}
            #Header, #WorkflowCard, #LogPanel {{
                background: {palette["surface"]};
                border: 1px solid {palette["border"]};
                border-radius: 12px;
            }}
            #PreviewCard {{
                background: {palette["card"]};
                border: 1px solid {palette["border"]};
                border-radius: 12px;
            }}
            #DeviceChip {{
                padding: 4px 10px;
                border-radius: 12px;
                border: 1px solid {palette["border"]};
                background: {palette["surface_alt"]};
                color: {palette["text"]};
            }}
            #TrackingPill {{
                padding: 4px 10px;
                border-radius: 12px;
                background: {palette["surface_alt"]};
            }}
            #TrackingPill[state="ok"] {{
                background: rgba(85, 211, 139, 0.25);
            }}
            #TrackingPill[state="warn"] {{
                background: rgba(246, 196, 83, 0.25);
            }}
            #TrackingPill[state="error"] {{
                background: rgba(242, 102, 102, 0.25);
            }}
            #PrimaryButton {{
                background: {palette["accent"]};
                color: {palette["accent_text"]};
                border-radius: 8px;
                padding: 6px 14px;
            }}
            #SecondaryButton {{
                background: {palette["surface_alt"]};
                color: {palette["text"]};
                border-radius: 8px;
                padding: 6px 14px;
            }}
            QTextEdit {{
                background: {palette["surface_alt"]};
                border: 1px solid {palette["border"]};
                border-radius: 8px;
                color: {palette["text"]};
            }}
            QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {{
                background: {palette["surface_alt"]};
                border: 1px solid {palette["border"]};
                border-radius: 6px;
                padding: 4px 6px;
                color: {palette["text"]};
            }}
            QToolButton {{
                background: {palette["surface_alt"]};
                border: 1px solid {palette["border"]};
                border-radius: 6px;
                padding: 4px 8px;
            }}
            QToolButton:checked {{
                background: {palette["accent"]};
                color: {palette["accent_text"]};
            }}
            #TrackingBanner {{
                background: {palette["surface_alt"]};
                border: 1px solid {palette["warn"]};
                border-radius: 10px;
            }}
            #PreviewMeta, #PreviewAlign, #UnitLabel, #InfoLabel {{
                color: {palette["text_muted"]};
            }}
            """
        )

    def _state_text(self, state: ScannerState) -> str:
        if state == ScannerState.IDLE:
            return self._t("workflow.state.idle")
        if state == ScannerState.PAUSED:
            return self._t("workflow.state.paused")
        if state == ScannerState.RECORDING:
            return self._t("workflow.state.recording")
        if state == ScannerState.PLAYBACK:
            return self._t("workflow.state.playback")
        if state == ScannerState.EXPORTING:
            return self._t("workflow.state.exporting")
        if state == ScannerState.LOST:
            return self._t("workflow.tracking.lost")
        if state == ScannerState.PREVIEW:
            return self._t("workflow.state.running")
        if state == ScannerState.SCANNING:
            return self._t("workflow.state.running")
        if state == ScannerState.STOPPING:
            return self._t("workflow.state.running")
        return self._t("workflow.state.running")

    def _tracking_text(self, status: str) -> str:
        if status == "LOST":
            return self._t("workflow.tracking.lost")
        if status == "WARN":
            return self._t("workflow.tracking.warn")
        return self._t("workflow.tracking.ok")

    def _on_primary_clicked(self) -> None:
        if self._state == ScannerState.IDLE:
            self.start_scan()
        else:
            self.stop_scan()

    def _on_secondary_clicked(self) -> None:
        if self._state == ScannerState.PAUSED:
            self.resume_scan()
        elif self._state not in (ScannerState.IDLE, ScannerState.STOPPING):
            self.pause_scan()

    def _apply_basic_mode(self) -> None:
        self.basic_btn.setChecked(True)
        for widget in self._advanced_widgets:
            widget.setVisible(False)

    def _apply_advanced_mode(self) -> None:
        self.advanced_btn.setChecked(True)
        for widget in self._advanced_widgets:
            widget.setVisible(True)

    def _update_export_formats(self) -> None:
        export_type = self.export_type_combo.currentData() or self.export_type_combo.currentText().lower()
        self.export_format_combo.clear()
        if export_type == "mesh":
            self.export_format_combo.addItems(["ply", "obj", "stl", "glb"])
        else:
            self.export_format_combo.addItems(["ply", "pcd"])

    def _export_next(self) -> None:
        index = self.export_stack.currentIndex()
        if index < self.export_stack.count() - 1:
            self.export_stack.setCurrentIndex(index + 1)

    def _export_prev(self) -> None:
        index = self.export_stack.currentIndex()
        if index > 0:
            self.export_stack.setCurrentIndex(index - 1)

    def _sync_mode_combo(self, source: QComboBox, target: QComboBox, text: str) -> None:
        if target.currentText() == text:
            return
        blocker = QSignalBlocker(target)
        try:
            target.setCurrentText(text)
        finally:
            del blocker

    def _choose_export_path(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        filename, _ = QFileDialog.getSaveFileName(self, self._t("export.step.path"), "", "All Files (*)")
        if not filename:
            return
        self.export_path_edit.setText(filename)
        self._export_path_override = Path(filename)

    def _choose_bag_path(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        filename, _ = QFileDialog.getSaveFileName(self, self._t("param.bag_path"), "", "RealSense bag (*.bag)")
        if not filename:
            return
        self.bag_path_edit.setText(filename)

    def _refresh_log_view(self) -> None:
        show_info = self.log_info_chk.isChecked()
        show_warn = self.log_warn_chk.isChecked()
        show_error = self.log_error_chk.isChecked()
        lines = []
        for level, message in self._log_entries:
            if level == "info" and not show_info:
                continue
            if level == "warn" and not show_warn:
                continue
            if level == "error" and not show_error:
                continue
            lines.append(message)
        self.log_text.setPlainText("\n".join(lines))
        if self.log_autoscroll_chk.isChecked():
            cursor = self.log_text.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self.log_text.setTextCursor(cursor)

    def _clear_log(self) -> None:
        self._log_entries.clear()
        self.log_text.clear()

    def _copy_log(self) -> None:
        clipboard = QApplication.clipboard()
        clipboard.setText(self.log_text.toPlainText())

    def _ensure_viewer(self) -> None:
        if getattr(self, "viewer", None) is not None:
            return
        if not self._enable_3d_viewer:
            self._show_error("3D 视图不可用", self._t("view.disabled"))
            return
        try:
            from scanner.ui.qt_viz import Qt3DViewer
        except Exception as exc:
            self._show_error("3D 视图不可用", f"Qt3DViewer 加载失败: {exc}")
            if hasattr(self, "depth_stack"):
                self.depth_stack.setCurrentWidget(self.depth_label)
            return
        try:
            self.viewer = Qt3DViewer(self)
            if hasattr(self, "depth_stack"):
                self.depth_stack.addWidget(self.viewer)
                self.depth_stack.setCurrentWidget(self.viewer)
        except Exception as exc:
            self._show_error("3D 视图不可用", f"3D 视图初始化失败: {exc}")
            self.viewer = None
            if hasattr(self, "depth_stack"):
                self.depth_stack.setCurrentWidget(self.depth_label)

    def _on_reset_camera(self) -> None:
        self._ensure_viewer()
        self.viewer.reset_camera()

    def _on_toggle_axes(self) -> None:
        self._ensure_viewer()
        self.viewer.toggle_axes()

    def _show_tracking_tips(self) -> None:
        tips = self._t("banner.tracking_lost.tips")
        QMessageBox.information(self, self._t("workflow.tracking.lost"), tips)

    def _diagnose_opengl(self) -> None:
        try:
            fmt = QSurfaceFormat.defaultFormat()
            surface = QOffscreenSurface()
            surface.setFormat(fmt)
            surface.create()

            ctx = QOpenGLContext()
            ctx.setFormat(fmt)
            if not ctx.create():
                raise RuntimeError("QOpenGLContext 创建失败")
            if not ctx.makeCurrent(surface):
                raise RuntimeError("无法创建当前 OpenGL 上下文")

            funcs = ctx.functions()

            def _get_string(name: int) -> str:
                value = funcs.glGetString(name)
                if value is None:
                    return "N/A"
                if isinstance(value, (bytes, bytearray)):
                    return value.decode(errors="ignore")
                try:
                    return bytes(value).decode(errors="ignore")
                except Exception:
                    return str(value)

            gl_version = _get_string(0x1F02)  # GL_VERSION
            glsl_version = _get_string(0x8B8C)  # GL_SHADING_LANGUAGE_VERSION
            gl_vendor = _get_string(0x1F00)  # GL_VENDOR
            gl_renderer = _get_string(0x1F01)  # GL_RENDERER
            ctx.doneCurrent()

            message = (
                f"OpenGL: {gl_version}\n"
                f"GLSL: {glsl_version}\n"
                f"Vendor: {gl_vendor}\n"
                f"Renderer: {gl_renderer}\n"
                f"Requested: {fmt.majorVersion()}.{fmt.minorVersion()} core"
            )
            QMessageBox.information(self, self._t("opengl.title"), message)
        except Exception as exc:
            self._show_error(self._t("opengl.title"), f"{self._t('opengl.fail')}: {exc}")

    @Slot()
    def start_scan(self) -> None:
        """
        启动采集与重建线程并创建会话。
        :return: 返回介绍
        """
        try:
            if self._capture_thread is not None:
                self.append_log("扫描已在运行")
                return

            self.append_log("启动扫描：准备参数与线程")
            self._apply_params()
            self._config["scan"]["mode"] = self.mode_combo.currentData() or self.mode_combo.currentText()
            self._config["device"]["bag_path"] = self.bag_path_edit.text().strip()
            if self._config["scan"]["mode"] == "offline":
                bag_path = self._validate_bag_path("playback")
                if bag_path is None:
                    return
                self._config["device"]["bag_path"] = str(bag_path)
                self._config["device"]["playback_bag"] = True
                self._config["device"]["record_bag"] = False
                self._config["device"]["playback_real_time"] = False
            else:
                self._config["device"]["playback_bag"] = False
                self._config["device"]["record_bag"] = self.record_checkbox.isChecked()
                if self._config["device"]["record_bag"]:
                    bag_path = self._validate_bag_path("record")
                    if bag_path is None:
                        return
                    self._config["device"]["bag_path"] = str(bag_path)

            self.append_log(
                f"扫描参数: mode={self._config['scan']['mode']} "
                f"record={self._config['device'].get('record_bag')} "
                f"bag={self._config['device'].get('bag_path', '')}"
            )

            self._queue = FrameQueue(maxsize=int(self._config.get("scan", {}).get("max_queue", 30)))
            from scanner.devices.realsense_device import RealSenseDevice
            from scanner.ui.threads import CaptureThread, ReconstructThread

            device = RealSenseDevice(self._config.get("device", {}))
            self._capture_thread = CaptureThread(device, self._config)
            self._reconstruct_thread = ReconstructThread(self._queue, self._config)

            self._capture_thread.frame_signal.connect(self._on_frame)
            self._capture_thread.info_signal.connect(self._on_info)
            self._capture_thread.log_signal.connect(self.append_log)
            self._capture_thread.error_signal.connect(self._on_error)
            self._capture_thread.device_signal.connect(self._on_device_info)

            self._reconstruct_thread.preview_signal.connect(self._on_preview)
            self._reconstruct_thread.preview_mesh_signal.connect(self._on_mesh_preview)
            self._reconstruct_thread.status_signal.connect(self._on_status)
            self._reconstruct_thread.log_signal.connect(self.append_log)
            self._reconstruct_thread.error_signal.connect(self._on_error)

            self._capture_thread.start()
            self._reconstruct_thread.start()

            self._start_session()
            if self._config.get("scan", {}).get("mode") == "offline":
                self._set_state(ScannerState.PLAYBACK if hasattr(ScannerState, "PLAYBACK") else ScannerState.PREVIEW)
            elif self._config.get("device", {}).get("record_bag"):
                self._set_state(ScannerState.RECORDING)
            else:
                self._set_state(ScannerState.PREVIEW)
            self.append_log("扫描启动")
        except Exception as exc:
            self._handle_exception("启动扫描", exc)
            self._shutdown_threads()
            self._set_state(ScannerState.IDLE)

    @Slot()
    def pause_scan(self) -> None:
        """
        暂停扫描与融合。
        :return: 返回介绍
        """
        try:
            if self._reconstruct_thread is not None:
                self._reconstruct_thread.pause()
                self._set_state(ScannerState.PAUSED)
                self.append_log("已暂停")
        except Exception as exc:
            self._handle_exception("暂停扫描", exc)

    @Slot()
    def resume_scan(self) -> None:
        """
        从暂停状态恢复扫描。
        :return: 返回介绍
        """
        try:
            if self._reconstruct_thread is not None:
                self._reconstruct_thread.resume()
                if self._config.get("scan", {}).get("mode") == "offline":
                    self._set_state(ScannerState.PLAYBACK)
                elif self._config.get("device", {}).get("record_bag"):
                    self._set_state(ScannerState.RECORDING)
                else:
                    self._set_state(ScannerState.SCANNING)
                self.append_log("已继续")
        except Exception as exc:
            self._handle_exception("继续扫描", exc)

    @Slot()
    def stop_scan(self) -> None:
        """
        停止扫描线程并保存会话元数据。
        :return: 返回介绍
        """
        try:
            if self._capture_thread is None and self._reconstruct_thread is None:
                self.append_log("扫描未启动")
                return
            self._set_state(ScannerState.STOPPING)
            reconstruct_thread = self._shutdown_threads()
            self.append_log("扫描已停止")
            self._finalize_session(reconstruct_thread)
            self._set_state(ScannerState.IDLE)
        except Exception as exc:
            self._handle_exception("停止扫描", exc)
            self._set_state(ScannerState.IDLE)

    @Slot()
    def _request_exit(self) -> None:
        if self._state == ScannerState.STOPPING:
            return
        self.append_log("退出中...")
        try:
            self._set_state(ScannerState.STOPPING)
            self._shutdown_threads()
        finally:
            if self._stop_event is not None:
                self._stop_event.set()
            app = QApplication.instance()
            if app is not None:
                app.quit()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._request_exit()
        event.accept()

    @Slot()
    def optimize_scan(self) -> None:
        """
        触发回环检测与位姿图全局优化。
        :return: 返回介绍
        """
        try:
            if self._reconstruct_thread is None:
                self.append_log("请先完成扫描")
                return
            self._set_state(ScannerState.OPTIMIZING)
            keyframes = self._reconstruct_thread.get_keyframes()
            from scanner.ui.threads import OptimizeThread

            self._optimize_thread = OptimizeThread(keyframes, self._config)
            self._optimize_thread.mesh_signal.connect(self._on_mesh)
            self._optimize_thread.log_signal.connect(self.append_log)
            self._optimize_thread.error_signal.connect(self._on_error)
            self._optimize_thread.start()
        except Exception as exc:
            self._handle_exception("全局优化", exc)
            if self._capture_thread:
                if self._config.get("scan", {}).get("mode") == "offline":
                    self._set_state(ScannerState.PLAYBACK)
                else:
                    self._set_state(ScannerState.SCANNING)
            else:
                self._set_state(ScannerState.IDLE)

    @Slot()
    def export_result(self) -> None:
        """
        导出当前网格与点云结果。
        :return: 返回介绍
        """
        try:
            if self._session is None:
                self.append_log("没有会话数据")
                return

            self._set_state(ScannerState.EXPORTING)
            export_dir = ensure_dir(self._session.root / "exports")
            export_type = self.export_type_combo.currentData() or self.export_type_combo.currentText().lower()
            fmt = self.export_format_combo.currentText()
            base_name = "mesh" if export_type == "mesh" else "cloud"
            if self._export_path_override is not None:
                export_dir = ensure_dir(self._export_path_override.parent)
                base_name = self._export_path_override.stem
            mesh = self._current_mesh
            pcd = self._current_pcd
            if export_type == "mesh" and mesh is not None:
                from scanner.geometry.exporter import export_mesh
                from scanner.geometry.postprocess import postprocess_mesh

                simplify_target = (
                    int(self._config.get("mesh", {}).get("simplify_target_triangles", 200000))
                    if self.export_simplify_check.isChecked()
                    else None
                )
                smooth_iterations = (
                    int(self._config.get("mesh", {}).get("smooth_iterations", 5))
                    if self.export_smooth_check.isChecked()
                    else 0
                )
                if simplify_target is not None or smooth_iterations > 0:
                    mesh = postprocess_mesh(
                        mesh,
                        simplify_target=simplify_target or int(len(mesh.triangles)),
                        smooth_iterations=smooth_iterations,
                        remove_small=bool(self._config.get("mesh", {}).get("remove_small_components", True)),
                        min_triangles=int(self._config.get("mesh", {}).get("min_triangles", 500)),
                        keep_largest=bool(self._config.get("mesh", {}).get("keep_largest", True)),
                    )
                path = export_mesh(mesh, export_dir / base_name, fmt)
                self._session.exports.append(
                    {
                        "type": "mesh",
                        "path": str(path),
                        "format": fmt,
                        "params": {
                            "simplify_target": simplify_target,
                            "smooth_iterations": smooth_iterations,
                            "remove_small": self._config.get("mesh", {}).get("remove_small_components"),
                            "min_triangles": self._config.get("mesh", {}).get("min_triangles"),
                            "keep_largest": self._config.get("mesh", {}).get("keep_largest"),
                        },
                    }
                )
                self.append_log(f"网格已导出: {path}")

            if export_type != "mesh" and pcd is not None:
                from scanner.geometry.exporter import export_point_cloud

                path = export_point_cloud(pcd, export_dir / base_name, fmt)
                self._session.exports.append(
                    {"type": "cloud", "path": str(path), "format": fmt, "params": {}}
                )
                self.append_log(f"点云已导出: {path}")

            if self._session is not None:
                self._session_manager.save_session(self._session)
            self._export_path_override = None
            if self._capture_thread:
                if self._config.get("scan", {}).get("mode") == "offline":
                    self._set_state(ScannerState.PLAYBACK)
                elif self._config.get("device", {}).get("record_bag"):
                    self._set_state(ScannerState.RECORDING)
                else:
                    self._set_state(ScannerState.SCANNING)
            else:
                self._set_state(ScannerState.IDLE)
        except Exception as exc:
            self._handle_exception("导出结果", exc)
            if self._capture_thread:
                if self._config.get("scan", {}).get("mode") == "offline":
                    self._set_state(ScannerState.PLAYBACK)
                else:
                    self._set_state(ScannerState.SCANNING)
            else:
                self._set_state(ScannerState.IDLE)

    @Slot(object)
    def _on_frame(self, frame: FrameBundle) -> None:
        """
        接收采集帧：写入队列并刷新预览。
        :param frame: RGB-D 帧数据
        :return: None
        """
        if self._queue is not None:
            self._queue.put(frame)
        self._update_preview(frame)

    @Slot(dict)
    def _on_info(self, info: Dict[str, Any]) -> None:
        """
        更新采集统计信息（FPS/帧号）。
        :param info: 信息字典（含 fps/index）
        :return: None
        """
        fps = info.get("fps", 0.0)
        index = info.get("index", 0)
        self._last_fps = float(fps)
        self.fps_label.setText(f"{self._t('header.fps')}: {fps:.1f}")

    @Slot(object)
    def _on_device_info(self, info: Any) -> None:
        self._device_connected = True
        self.device_chip.setText(self._t("header.device.connected"))
        try:
            self.device_chip.setToolTip(f"{info.name} {info.serial}")
        except Exception:
            pass

    @Slot(str)
    def _on_error(self, message: str) -> None:
        """
        统一处理采集线程错误，确保可见性与安全停机。
        :param message: 错误信息
        :return: None
        """
        self._device_connected = False
        self.device_chip.setText(self._t("header.device.disconnected"))
        self._show_error("设备错误", message)
        if self._capture_thread or self._reconstruct_thread:
            self._set_state(ScannerState.STOPPING)
            reconstruct_thread = self._shutdown_threads()
            self._finalize_session(reconstruct_thread)
            self._set_state(ScannerState.IDLE)

    @Slot(object)
    def _on_preview(self, pcd: o3d.geometry.PointCloud) -> None:
        """
        更新预览点云并按当前视图显示。
        :param pcd: 点云数据
        :return: None
        """
        self._current_pcd = pcd
        mode = self.view_combo.currentData() or self.view_combo.currentText()
        if mode == "pointcloud":
            self._ensure_viewer()
            self.viewer.show_point_cloud(pcd)

    @Slot(str)
    def _on_status(self, status: str) -> None:
        """
        更新跟踪状态标签与 UI 状态机。
        :param status: 跟踪状态（OK/WARN/LOST）
        :return: None
        """
        self._last_tracking_status = status
        self.tracking_pill.setText(self._tracking_text(status))
        self.tracking_status_label.setText(self._tracking_text(status))
        if status == "LOST":
            self.tracking_pill.setProperty("state", "error")
        elif status == "WARN":
            self.tracking_pill.setProperty("state", "warn")
        else:
            self.tracking_pill.setProperty("state", "ok")
        self.tracking_pill.style().unpolish(self.tracking_pill)
        self.tracking_pill.style().polish(self.tracking_pill)
        if status == "LOST":
            self._set_state(ScannerState.LOST)
            self.tracking_banner.setVisible(True)
        elif status in ("OK", "WARN") and self._state not in (
            ScannerState.OPTIMIZING,
            ScannerState.EXPORTING,
            ScannerState.STOPPING,
        ):
            if self._config.get("scan", {}).get("mode") == "offline":
                self._set_state(ScannerState.PLAYBACK)
            elif self._config.get("device", {}).get("record_bag"):
                self._set_state(ScannerState.RECORDING)
            else:
                self._set_state(ScannerState.SCANNING)
            self.tracking_banner.setVisible(False)

    @Slot(object)
    def _on_mesh(self, mesh: o3d.geometry.TriangleMesh) -> None:
        """
        更新重建结果网格并刷新显示。
        :param mesh: 网格数据
        :return: None
        """
        self._current_mesh = mesh
        mode = self.view_combo.currentData() or self.view_combo.currentText()
        if mode == "mesh":
            self._ensure_viewer()
            self.viewer.show_mesh(mesh)
        if self._capture_thread:
            if self._config.get("scan", {}).get("mode") == "offline":
                self._set_state(ScannerState.PLAYBACK)
            elif self._config.get("device", {}).get("record_bag"):
                self._set_state(ScannerState.RECORDING)
            else:
                self._set_state(ScannerState.SCANNING)
        else:
            self._set_state(ScannerState.IDLE)

    @Slot(object)
    def _on_mesh_preview(self, mesh: o3d.geometry.TriangleMesh) -> None:
        """
        更新实时预览网格并按需显示。
        :param mesh: 预览网格
        :return: None
        """
        self._current_mesh = mesh
        mode = self.view_combo.currentData() or self.view_combo.currentText()
        if mode == "mesh":
            self._ensure_viewer()
            self.viewer.show_mesh(mesh)

    def _write_bootstrap_log(self, message: str) -> None:
        path = Path(self._config.get("session_root", "sessions")) / "app.log"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(message.rstrip() + "\n")
        except Exception:
            pass

    def _show_error(self, title: str, message: str) -> None:
        self.append_log(message, level="error")
        logger = self._logger or self._bootstrap_logger
        if logger is not None:
            logger.error(message)
        else:
            self._write_bootstrap_log(message)
        QMessageBox.critical(self, title, message)

    def _handle_exception(self, context: str, exc: Exception) -> None:
        message = f"{context} 失败: {exc}"
        detail = traceback.format_exc()
        self.append_log(message, level="error")
        logger = self._logger or self._bootstrap_logger
        if logger is not None:
            logger.error("%s\n%s", message, detail)
        else:
            self._write_bootstrap_log(f"{message}\n{detail}")
        QMessageBox.critical(self, "错误", message)

    def _validate_bag_path(self, mode: str) -> Optional[Path]:
        text = self.bag_path_edit.text().strip()
        if not text:
            self._show_error("参数错误", "请填写 bag 路径")
            return None
        path = Path(text).expanduser()
        if path.suffix.lower() != ".bag":
            self._show_error("参数错误", "bag 文件必须以 .bag 结尾")
            return None
        path = path.resolve()
        if mode == "record":
            parent = path.parent
            parent.mkdir(parents=True, exist_ok=True)
            if not parent.is_dir():
                self._show_error("参数错误", f"bag 目录不存在: {parent}")
                return None
            if not os.access(parent, os.W_OK):
                self._show_error("参数错误", f"bag 目录不可写: {parent}")
                return None
        else:
            if not path.exists():
                self._show_error("参数错误", f"bag 文件不存在: {path}")
                return None
        self.bag_path_edit.setText(str(path))
        return path

    def _check_device_available(self) -> bool:
        try:
            from scanner.devices.realsense_device import RealSenseDevice

            return RealSenseDevice.has_device()
        except Exception as exc:
            self._show_error("设备错误", str(exc))
            return False

    def _shutdown_threads(self) -> Optional[ReconstructThread]:
        reconstruct_thread = self._reconstruct_thread
        if self._capture_thread:
            self._capture_thread.stop()
            if not self._capture_thread.wait(3000):
                self.append_log("采集线程停止超时", level="warn")
            self._capture_thread = None
        if self._optimize_thread:
            self._optimize_thread.stop()
            if not self._optimize_thread.wait(3000):
                self.append_log("优化线程停止超时", level="warn")
            self._optimize_thread = None
        if reconstruct_thread:
            reconstruct_thread.stop()
            if not reconstruct_thread.wait(3000):
                self.append_log("重建线程停止超时", level="warn")
            self._reconstruct_thread = None
        if self._queue:
            self._queue.clear()
        return reconstruct_thread

    def _start_session(self) -> None:
        """
        创建会话目录并初始化日志。
        :return: None
        """
        self._session = self._session_manager.create_session(self._config)
        log_path = self._session.root / "logs" / "session.log"
        log_level = self._config.get("logging", {}).get("level") or os.environ.get("DEPTHFORGE_LOG_LEVEL", "INFO")
        self._logger = setup_logger(log_path, level=log_level, name="scanner.session", clear_handlers=True)
        self._session.logs_path = str(log_path)
        if self._config.get("device", {}).get("record_bag") or self._config.get("device", {}).get("playback_bag"):
            self._session.bag_path = self._config.get("device", {}).get("bag_path")
        self.append_log(f"会话创建: {self._session.session_id}")

    def _finalize_session(self, reconstruct_thread: Optional[ReconstructThread]) -> None:
        """
        保存关键帧索引与轨迹到会话。
        :param reconstruct_thread: 重建线程（提供关键帧与轨迹）
        :return: None
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
        刷新 RGB/Depth 预览画面与相机信息。
        :param frame: RGB-D 帧
        :return: None
        """
        color_img = self._to_qimage(frame.color_rgb)
        depth_img = self._depth_to_colormap(frame.depth_mm)
        self.color_label.setPixmap(QPixmap.fromImage(color_img).scaled(640, 360, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.depth_label.setPixmap(QPixmap.fromImage(depth_img).scaled(640, 360, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        h, w, _ = frame.color_rgb.shape
        self.rgb_meta_label.setText(f"{w}x{h} · {self._last_fps:.1f} FPS")
        self.depth_meta_label.setText(f"{w}x{h} · {self._last_fps:.1f} FPS")
        intr = frame.intrinsics
        self.info_label.setText(
            f"内参 fx={intr.fx:.1f} fy={intr.fy:.1f} cx={intr.cx:.1f} cy={intr.cy:.1f}"
        )

    def _to_qimage(self, rgb: np.ndarray) -> QImage:
        """
        将 RGB 图像数组转换为 QImage。
        :param rgb: HxWx3 的 RGB 图像
        :return: QImage
        """
        rgb_img = rgb.copy()
        h, w, _ = rgb_img.shape
        return QImage(rgb_img.data, w, h, 3 * w, QImage.Format_RGB888).copy()

    def _depth_to_colormap(self, depth: np.ndarray) -> QImage:
        """
        将深度图转换为伪彩色 QImage。
        :param depth: 深度图（mm）
        :return: QImage
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

    def append_log(self, message: str, level: str = "info") -> None:
        """
        追加日志并同步到 UI 与文件。
        :param message: 日志内容
        :param level: info|warn|error
        :return: None
        """
        self._log_entries.append((level, message))
        max_lines = int(self._config.get("ui", {}).get("log_max_lines", 500))
        if len(self._log_entries) > max_lines:
            self._log_entries = self._log_entries[-max_lines:]
        self._refresh_log_view()
        if self._logger is not None:
            if level == "error":
                self._logger.error(message)
            elif level == "warn":
                self._logger.warning(message)
            else:
                self._logger.info(message)
        else:
            self._write_bootstrap_log(message)

    @Slot()
    def save_session(self) -> None:
        """
        将当前会话元数据写入 session.json。
        :return: None
        """
        try:
            if self._session is None:
                self.append_log("没有可保存的会话")
                return
            self._session_manager.save_session(self._session)
            self.append_log("会话已保存")
        except Exception as exc:
            self._handle_exception("保存会话", exc)

    @Slot()
    def load_session(self) -> None:
        """
        载入历史会话并尝试显示导出结果。
        :return: None
        """
        try:
            from PySide6.QtWidgets import QFileDialog

            path, _ = QFileDialog.getOpenFileName(self, "选择会话文件", "", "Session (*.json)")
            if not path:
                return
            session = self._session_manager.load_session(Path(path))
            self._session = session
            self.append_log(f"会话已加载: {session.session_id}")
            self._set_state(ScannerState.IDLE)
            shown = False
            o3d = _require_open3d()
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
        except Exception as exc:
            self._handle_exception("加载会话", exc)

    def _apply_params(self) -> None:
        """
        将 UI 参数写回到运行配置。
        :return: None
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
        更新状态机并刷新 UI 控件状态。
        :param state: 目标状态
        :return: None
        """
        self._state = state
        self.state_label.setText(self._state_text(state))
        is_idle = state == ScannerState.IDLE
        self.start_btn.setText(self._t("workflow.primary.start") if is_idle else self._t("workflow.primary.stop"))
        self.pause_btn.setText(self._t("workflow.secondary.resume") if state == ScannerState.PAUSED else self._t("workflow.secondary.pause"))
        self.start_btn.setEnabled(state != ScannerState.STOPPING)
        self.mode_combo.setEnabled(is_idle)
        self.mode_combo_io.setEnabled(is_idle)
        self.bag_path_edit.setEnabled(is_idle)
        self.record_checkbox.setEnabled(is_idle)
        self.pause_btn.setEnabled(state in (ScannerState.SCANNING, ScannerState.RECORDING, ScannerState.PLAYBACK, ScannerState.PREVIEW, ScannerState.PAUSED))

    def _on_view_mode(self, mode: str) -> None:
        """
        切换预览模式（RGBD/点云/网格）。
        :param mode: 视图模式
        :return: None
        """
        mode_value = self.view_combo.currentData() or mode
        if hasattr(self, "depth_stack"):
            if mode_value == "rgbd":
                self.depth_stack.setCurrentWidget(self.depth_label)
            else:
                self._ensure_viewer()
                if self.viewer is None:
                    if self.view_combo.currentData() != "rgbd":
                        for i in range(self.view_combo.count()):
                            if self.view_combo.itemData(i) == "rgbd":
                                self.view_combo.setCurrentIndex(i)
                                break
                    self.depth_stack.setCurrentWidget(self.depth_label)
                else:
                    self.depth_stack.setCurrentWidget(self.viewer)
        if mode_value == "mesh" and self._current_mesh is not None:
            self.viewer.show_mesh(self._current_mesh)
        elif mode_value == "pointcloud" and self._current_pcd is not None:
            self.viewer.show_point_cloud(self._current_pcd)
