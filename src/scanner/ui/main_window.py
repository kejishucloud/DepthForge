from __future__ import annotations

from pathlib import Path
import copy
import html
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
    QDialog,
    QDialogButtonBox,
    QFormLayout,
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
    QScrollArea,
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
from scanner.io.config import save_config
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
        self._defaults = copy.deepcopy(config)
        self._lang = str(config.get("ui", {}).get("language", "en-US"))
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
        self._last_tracking_status = "INIT"
        self._log_collapsed = False
        self._toast_timer: Optional[QTimer] = None

        self.setWindowTitle(self._t("app.title"))
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
        aligned = bool(self._config.get("device", {}).get("align_to_color", True))
        self.align_chip = QLabel(self._t("header.aligned") if aligned else self._t("header.not_aligned"))
        self.align_chip.setObjectName("AlignmentChip")
        header_center_layout.addWidget(self.device_chip)
        header_center_layout.addWidget(self.fps_label)
        header_center_layout.addWidget(self.align_chip)

        header_right = QWidget(header)
        header_right_layout = QHBoxLayout(header_right)
        header_right_layout.setContentsMargins(0, 0, 0, 0)
        header_right_layout.setSpacing(8)
        self.opengl_diag_btn = QToolButton(self)
        self.opengl_diag_btn.setText(self._t("header.diagnostics"))
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

        self.toast = QFrame(self)
        self.toast.setObjectName("Toast")
        toast_layout = QHBoxLayout(self.toast)
        toast_layout.setContentsMargins(12, 6, 12, 6)
        self.toast_label = QLabel("")
        self.toast_label.setObjectName("ToastLabel")
        toast_layout.addWidget(self.toast_label)
        self.toast.setVisible(False)

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

        self.mode_combo = QComboBox()
        self._populate_mode_combo(self._resolve_initial_mode())
        mode_form = QFormLayout()
        self.label_mode = QLabel(self._t("workflow.mode"))
        mode_form.addRow(self.label_mode, self.mode_combo)

        self.bag_path_edit = QLineEdit()
        self.bag_path_edit.setPlaceholderText(self._t("workflow.bag_path"))
        self.bag_path_edit.setText(self._config.get("device", {}).get("bag_path", ""))
        self.browse_bag_btn = QPushButton(self._t("action.browse"))
        self.bag_row = QWidget(self)
        bag_layout = QHBoxLayout(self.bag_row)
        bag_layout.setContentsMargins(0, 0, 0, 0)
        bag_layout.setSpacing(6)
        bag_layout.addWidget(self.bag_path_edit)
        bag_layout.addWidget(self.browse_bag_btn)
        self.bag_row_label = QLabel(self._t("workflow.bag_path"))
        mode_form.addRow(self.bag_row_label, self.bag_row)

        status_row = QHBoxLayout()
        self.state_label = QLabel(self._t("workflow.state.idle"))
        self.state_label.setObjectName("StateLabel")
        self.tracking_pill = QLabel(self._t("workflow.tracking.init"))
        self.tracking_pill.setObjectName("TrackingPill")
        self.tracking_pill.setProperty("state", "init")
        status_row.addWidget(self.state_label)
        status_row.addStretch(1)
        status_row.addWidget(self.tracking_pill)

        workflow_layout.addLayout(actions_row)
        workflow_layout.addLayout(mode_form)
        workflow_layout.addLayout(status_row)
        self.info_label = QLabel("")
        self.info_label.setObjectName("InfoLabel")
        workflow_layout.addWidget(self.info_label)

        params_tab = QWidget(self)
        params_layout = QVBoxLayout(params_tab)
        params_layout.setContentsMargins(8, 8, 8, 8)
        params_layout.setSpacing(8)

        basic_adv_row = QHBoxLayout()
        self.params_title_label = QLabel(self._t("section.parameters"))
        basic_adv_row.addWidget(self.params_title_label)
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
        self.restore_defaults_btn = QToolButton(self)
        self.restore_defaults_btn.setText(self._t("action.restore_defaults"))
        basic_adv_row.addWidget(self.restore_defaults_btn)
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

        def range_tip(min_v: float, max_v: float, unit: str) -> str:
            return self._t("tooltip.range").format(min=min_v, max=max_v, unit=unit)

        self.voxel_spin.setToolTip(range_tip(0.001, 0.02, " m"))
        self.sdf_spin.setToolTip(range_tip(0.005, 0.1, " m"))
        self.depth_trunc_spin.setToolTip(range_tip(0.3, 3.0, " m"))

        self.key_trans_spin = QDoubleSpinBox()
        self.key_trans_spin.setDecimals(3)
        self.key_trans_spin.setRange(0.005, 0.2)
        self.key_trans_spin.setValue(float(self._config.get("keyframe", {}).get("min_translation", 0.03)))
        self.key_rot_spin = QDoubleSpinBox()
        self.key_rot_spin.setDecimals(1)
        self.key_rot_spin.setRange(1.0, 30.0)
        self.key_rot_spin.setValue(float(self._config.get("keyframe", {}).get("min_rotation_deg", 5.0)))
        self.key_trans_spin.setToolTip(range_tip(0.005, 0.2, " m"))
        self.key_rot_spin.setToolTip(range_tip(1.0, 30.0, " °"))

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
        self.loop_gap_spin.setToolTip(range_tip(10, 200, " f"))
        self.loop_dist_spin.setToolTip(range_tip(0.1, 2.0, " m"))
        self.loop_candidates_spin.setToolTip(range_tip(1, 20, ""))

        self.simplify_spin = QSpinBox()
        self.simplify_spin.setRange(1000, 2000000)
        self.simplify_spin.setValue(int(self._config.get("mesh", {}).get("simplify_target_triangles", 200000)))
        self.smooth_spin = QSpinBox()
        self.smooth_spin.setRange(0, 50)
        self.smooth_spin.setValue(int(self._config.get("mesh", {}).get("smooth_iterations", 5)))
        self.simplify_spin.setToolTip(range_tip(1000, 2000000, " tris"))
        self.smooth_spin.setToolTip(range_tip(0, 50, ""))

        self.preview_mesh_checkbox = QCheckBox(self._t("param.preview_mesh"))
        self.preview_mesh_checkbox.setChecked(bool(self._config.get("fusion", {}).get("preview_mesh", False)))

        self.view_combo = QComboBox()
        default_view = self._resolve_default_view()
        self._populate_view_combo(default_view)
        self.reset_camera_btn = QPushButton(self._t("action.reset_camera"))
        self.toggle_axes_btn = QPushButton(self._t("action.toggle_axis"))

        # Bag path widgets are created in workflow section.

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
        self.tracking_status_label = QLabel(self._t("workflow.tracking.init"))
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
        ]

        params_layout.addWidget(self.params_toolbox, 1)

        export_tab = QWidget(self)
        export_layout = QVBoxLayout(export_tab)
        export_layout.setContentsMargins(8, 8, 8, 8)
        export_layout.setSpacing(8)

        export_form = QFormLayout()
        self.export_target_combo = QComboBox()
        self._populate_export_target_combo("mesh")
        self.label_export_target = QLabel(self._t("export.target"))
        export_form.addRow(self.label_export_target, self.export_target_combo)

        self.export_format_combo = QComboBox()
        self.label_export_format = QLabel(self._t("export.format"))
        export_form.addRow(self.label_export_format, self.export_format_combo)
        export_layout.addLayout(export_form)

        self.export_options_group = QGroupBox(self._t("export.options"))
        options_layout = QFormLayout(self.export_options_group)
        self.export_simplify_check = QCheckBox(self._t("export.option.simplify"))
        self.export_smooth_check = QCheckBox(self._t("export.option.smooth"))
        self.export_color_check = QCheckBox(self._t("export.option.color"))
        self.export_simplify_check.setChecked(True)
        self.export_smooth_check.setChecked(True)
        self.export_color_check.setChecked(True)
        self.label_export_coords = QLabel(self._t("export.option.coords"))
        self.export_coords_combo = QComboBox()
        self.export_coords_combo.addItem(self._t("export.coords.opengl"), "open3d")
        self.export_coords_combo.addItem(self._t("export.coords.unity"), "unity")
        options_layout.addRow(self.export_simplify_check)
        options_layout.addRow(self.export_smooth_check)
        options_layout.addRow(self.export_color_check)
        options_layout.addRow(self.label_export_coords, self.export_coords_combo)
        export_layout.addWidget(self.export_options_group)

        path_form = QFormLayout()
        self.export_path_edit = QLineEdit()
        self.export_path_edit.setPlaceholderText(self._t("export.output.placeholder"))
        self.export_path_btn = QPushButton(self._t("action.browse"))
        path_row = QWidget(self)
        path_row_layout = QHBoxLayout(path_row)
        path_row_layout.setContentsMargins(0, 0, 0, 0)
        path_row_layout.setSpacing(6)
        path_row_layout.addWidget(self.export_path_edit)
        path_row_layout.addWidget(self.export_path_btn)
        self.label_export_path = QLabel(self._t("export.output"))
        path_form.addRow(self.label_export_path, path_row)
        export_layout.addLayout(path_form)

        self.export_default_label = QLabel(self._default_export_hint())
        self.export_default_label.setObjectName("ExportHint")
        export_layout.addWidget(self.export_default_label)

        self.session_group = QGroupBox(self._t("section.session"))
        session_layout = QVBoxLayout(self.session_group)
        session_layout.setContentsMargins(8, 8, 8, 8)
        session_layout.setSpacing(8)
        self.optimize_btn = QPushButton(self._t("workflow.optimize"))
        self.save_session_btn = QPushButton(self._t("session.save"))
        self.load_session_btn = QPushButton(self._t("session.load"))
        session_layout.addWidget(self.optimize_btn)
        session_layout.addWidget(self.save_session_btn)
        session_layout.addWidget(self.load_session_btn)
        export_layout.addWidget(self.session_group)

        export_layout.addStretch(1)
        self.export_run_btn = QPushButton(self._t("export.action.export"))
        self.export_run_btn.setObjectName("SecondaryButton")
        export_layout.addWidget(self.export_run_btn)

        self.sidebar_stack = QStackedWidget(self)
        self.sidebar_stack.addWidget(params_tab)
        self.sidebar_stack.addWidget(export_tab)

        self.sidebar_params_btn = QToolButton(self)
        self.sidebar_params_btn.setCheckable(True)
        self.sidebar_params_btn.setObjectName("SidebarToggle")
        self.sidebar_params_btn.setText(self._t("section.parameters"))
        self.sidebar_export_btn = QToolButton(self)
        self.sidebar_export_btn.setCheckable(True)
        self.sidebar_export_btn.setObjectName("SidebarToggle")
        self.sidebar_export_btn.setText(self._t("section.export"))
        self.sidebar_btn_group = QButtonGroup(self)
        self.sidebar_btn_group.setExclusive(True)
        self.sidebar_btn_group.addButton(self.sidebar_params_btn, 0)
        self.sidebar_btn_group.addButton(self.sidebar_export_btn, 1)
        self.sidebar_params_btn.setChecked(True)

        switch_row = QHBoxLayout()
        switch_row.setSpacing(6)
        switch_row.addWidget(self.sidebar_params_btn)
        switch_row.addWidget(self.sidebar_export_btn)
        switch_row.addStretch(1)

        self.sidebar_scroll = QScrollArea(self)
        self.sidebar_scroll.setWidgetResizable(True)
        self.sidebar_scroll.setFrameShape(QFrame.NoFrame)
        self.sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.sidebar_scroll.setWidget(self.sidebar_stack)

        right_layout.addWidget(workflow_card, 0)
        right_layout.addLayout(switch_row)
        right_layout.addWidget(self.sidebar_scroll, 1)

        right_panel.setMinimumWidth(320)
        body.addWidget(left_panel)
        body.addWidget(right_panel)
        body.setStretchFactor(0, 4)
        body.setStretchFactor(1, 1)
        body.setSizes([int(self.width() * 0.8), max(320, int(self.width() * 0.2))])

        self.log_panel = QFrame(self)
        self.log_panel.setObjectName("LogPanel")
        log_layout = QVBoxLayout(self.log_panel)
        log_layout.setContentsMargins(8, 8, 8, 8)
        log_layout.setSpacing(6)

        log_header = QHBoxLayout()
        self.log_toggle_btn = QToolButton(self)
        self.log_toggle_btn.setText("▾")
        self.log_toggle_btn.setObjectName("LogToggle")
        log_header.addWidget(self.log_toggle_btn)
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
        self.log_text.setMinimumHeight(160)
        self.log_text.setMaximumHeight(220)

        log_layout.addLayout(log_header)
        log_layout.addWidget(self.log_text)

        root_layout.addWidget(header)
        root_layout.addWidget(self.tracking_banner)
        root_layout.addWidget(self.toast)
        root_layout.addWidget(body, 1)
        root_layout.addWidget(self.log_panel, 0)
        self.setCentralWidget(central)

        self.start_btn.clicked.connect(self._on_primary_clicked)
        self.pause_btn.clicked.connect(self._on_secondary_clicked)
        self.optimize_btn.clicked.connect(self.optimize_scan)
        self.export_run_btn.clicked.connect(self.export_result)
        self.export_target_combo.currentTextChanged.connect(self._update_export_formats)
        self.export_format_combo.currentTextChanged.connect(self._update_export_options)
        self.export_path_btn.clicked.connect(self._choose_export_path)
        self.save_session_btn.clicked.connect(self.save_session)
        self.load_session_btn.clicked.connect(self.load_session)
        self.view_combo.currentTextChanged.connect(self._on_view_mode)
        self.reset_camera_btn.clicked.connect(self._on_reset_camera)
        self.toggle_axes_btn.clicked.connect(self._on_toggle_axes)
        self.browse_bag_btn.clicked.connect(self._choose_bag_path)
        self.opengl_diag_btn.clicked.connect(self._diagnose_opengl)
        self.settings_btn.clicked.connect(self._open_settings)
        self.exit_btn.clicked.connect(self._request_exit)
        self.basic_btn.clicked.connect(self._apply_basic_mode)
        self.advanced_btn.clicked.connect(self._apply_advanced_mode)
        self.restore_defaults_btn.clicked.connect(self._restore_default_params)
        self.log_clear_btn.clicked.connect(self._clear_log)
        self.log_copy_btn.clicked.connect(self._copy_log)
        self.log_toggle_btn.clicked.connect(self._toggle_log_panel)
        self.log_info_chk.clicked.connect(self._refresh_log_view)
        self.log_warn_chk.clicked.connect(self._refresh_log_view)
        self.log_error_chk.clicked.connect(self._refresh_log_view)
        self.tracking_banner_action.clicked.connect(self._show_tracking_tips)
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        self.sidebar_params_btn.clicked.connect(lambda: self._set_sidebar_page(0))
        self.sidebar_export_btn.clicked.connect(lambda: self._set_sidebar_page(1))

        self._update_export_formats()
        self._update_export_options()
        self._on_mode_changed()
        self._apply_basic_mode()
        self._apply_theme(str(self._config.get("ui", {}).get("theme", "light")))
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
        status_dot = QFrame(self)
        status_dot.setObjectName("PreviewStatusDot")
        status_dot.setFixedSize(8, 8)
        status_dot.setProperty("state", "ok" if self._config.get("device", {}).get("align_to_color", True) else "warn")
        header.addWidget(title_label)
        header.addStretch(1)
        header.addWidget(meta_label)
        header.addWidget(status_dot)
        header.addWidget(align_label)

        if kind == "rgb":
            self.rgb_title_label = title_label
            self.rgb_meta_label = meta_label
            self.rgb_align_label = align_label
            self.rgb_status_dot = status_dot
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
            self.depth_status_dot = status_dot
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
                "header.device.disconnected": "未连接设备",
                "header.fps": "FPS",
                "header.aligned": "已对齐",
                "header.not_aligned": "未对齐",
                "header.diagnostics": "诊断",
                "header.settings": "设置 ⚙️",
                "header.exit": "退出",
                "preview.rgb.title": "RGB",
                "preview.depth.title": "Depth",
                "preview.aligned": "已对齐",
                "preview.not_aligned": "未对齐",
                "workflow.primary.start": "开始",
                "workflow.primary.stop": "停止",
                "workflow.secondary.pause": "暂停",
                "workflow.secondary.resume": "继续",
                "workflow.mode": "模式",
                "workflow.bag_path": "Bag 路径",
                "workflow.optimize": "全局优化",
                "workflow.state.idle": "空闲",
                "workflow.state.running": "运行中",
                "workflow.state.paused": "已暂停",
                "workflow.state.playback": "回放中",
                "workflow.state.recording": "录制中",
                "workflow.state.exporting": "导出中",
                "workflow.tracking.init": "初始化中",
                "workflow.tracking.ok": "正常",
                "workflow.tracking.lost": "丢失",
                "section.parameters": "参数",
                "section.export": "导出",
                "section.session": "会话",
                "toggle.basic": "基础",
                "toggle.advanced": "高级",
                "action.restore_defaults": "恢复默认值",
                "group.reconstruction": "重建",
                "group.tracking": "跟踪",
                "group.loop": "回环",
                "group.mesh": "网格",
                "group.view": "视图",
                "param.voxel": "体素大小",
                "param.sdf": "TSDF 截断",
                "param.depth_max": "深度截断",
                "param.keyframe_dt": "关键帧 Δt",
                "param.keyframe_dr": "关键帧 ΔR",
                "param.tracking_state": "跟踪状态",
                "param.loop_gap": "回环间隔",
                "param.loop_dist": "回环距离",
                "param.loop_cand": "回环候选数",
                "param.mesh_tri": "目标三角面数",
                "param.mesh_smooth": "平滑迭代",
                "param.preview_mesh": "预览网格",
                "param.view_mode": "视图模式",
                "action.reset_camera": "重置相机",
                "action.toggle_axis": "显示坐标轴",
                "action.browse": "浏览…",
                "mode.realtime": "实时",
                "mode.playback": "回放",
                "mode.record": "录制",
                "view.pointcloud": "点云",
                "view.mesh": "网格",
                "view.rgbd": "RGB-D",
                "view.disabled": "当前系统不支持 OpenGL 3.2，已禁用 3D 视图",
                "export.target": "导出目标",
                "export.target.mesh": "网格",
                "export.target.pointcloud": "点云",
                "export.format": "格式",
                "export.options": "选项",
                "export.option.simplify": "简化",
                "export.option.smooth": "平滑",
                "export.option.color": "带颜色",
                "export.option.coords": "坐标系",
                "export.coords.opengl": "Open3D（右手）",
                "export.coords.unity": "Unity（左手）",
                "export.output": "输出路径",
                "export.output.placeholder": "选择输出路径",
                "export.naming": "默认命名",
                "export.action.export": "导出",
                "export.action.exporting": "导出中…",
                "session.save": "保存会话",
                "session.load": "加载会话",
                "log.title": "日志",
                "log.filter.info": "信息",
                "log.filter.warn": "警告",
                "log.filter.error": "错误",
                "log.copy": "复制",
                "log.clear": "清空",
                "log.autoscroll": "自动滚动",
                "banner.tracking_lost.body": "跟踪丢失：请放慢移动、增加纹理或调整 depth_trunc",
                "banner.tracking_lost.action": "查看建议",
                "banner.tracking_lost.tips": "建议：\n- 缓慢稳定移动\n- 适当拉远相机\n- 增加纹理（贴纸/报纸）\n- 适当降低 depth_trunc",
                "settings.title": "设置",
                "settings.language": "语言",
                "settings.language.en": "English (US)",
                "settings.language.zh": "中文（简体）",
                "settings.theme": "主题",
                "settings.theme.system": "跟随系统",
                "settings.theme.light": "浅色",
                "settings.theme.dark": "深色",
                "settings.performance": "性能",
                "settings.performance.high": "高",
                "settings.performance.balanced": "平衡",
                "settings.performance.low": "低",
                "settings.shortcuts": "快捷键",
                "settings.shortcuts.hint": "常用快捷键将在此展示。",
                "tooltip.range": "范围：{min}–{max}{unit}",
                "info.intrinsics": "内参 fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}",
                "error.param.title": "参数错误",
                "error.device.title": "设备错误",
                "error.generic.title": "错误",
                "error.bag.required": "请填写 bag 路径",
                "error.bag.extension": "bag 文件必须以 .bag 结尾",
                "error.bag.dir_missing": "bag 目录不存在: {path}",
                "error.bag.dir_unwritable": "bag 目录不可写: {path}",
                "error.bag.missing": "bag 文件不存在: {path}",
                "opengl.title": "OpenGL 诊断",
                "opengl.fail": "无法获取 OpenGL 信息",
            },
            "en-US": {
                "app.title": "DepthForge Handheld 3D Scan",
                "app.loading": "Loading UI…",
                "header.device.connected": "Connected",
                "header.device.disconnected": "No device",
                "header.fps": "FPS",
                "header.aligned": "Aligned",
                "header.not_aligned": "Not aligned",
                "header.diagnostics": "Diagnostics",
                "header.settings": "Settings ⚙️",
                "header.exit": "Exit",
                "preview.rgb.title": "RGB",
                "preview.depth.title": "Depth",
                "preview.aligned": "Aligned",
                "preview.not_aligned": "Not aligned",
                "workflow.primary.start": "Start",
                "workflow.primary.stop": "Stop",
                "workflow.secondary.pause": "Pause",
                "workflow.secondary.resume": "Resume",
                "workflow.mode": "Mode",
                "workflow.bag_path": "Bag path",
                "workflow.optimize": "Global optimize",
                "workflow.state.idle": "Idle",
                "workflow.state.running": "Running",
                "workflow.state.paused": "Paused",
                "workflow.state.playback": "Playback",
                "workflow.state.recording": "Recording",
                "workflow.state.exporting": "Exporting",
                "workflow.tracking.init": "Initializing",
                "workflow.tracking.ok": "OK",
                "workflow.tracking.lost": "Lost",
                "section.parameters": "Parameters",
                "section.export": "Export",
                "section.session": "Session",
                "toggle.basic": "Basic",
                "toggle.advanced": "Advanced",
                "action.restore_defaults": "Restore defaults",
                "group.reconstruction": "Reconstruction",
                "group.tracking": "Tracking",
                "group.loop": "Loop Closure",
                "group.mesh": "Meshing",
                "group.view": "View",
                "param.voxel": "Voxel size",
                "param.sdf": "SDF truncation",
                "param.depth_max": "Depth trunc",
                "param.keyframe_dt": "Keyframe Δt",
                "param.keyframe_dr": "Keyframe ΔR",
                "param.tracking_state": "Tracking status",
                "param.loop_gap": "Loop gap",
                "param.loop_dist": "Loop distance",
                "param.loop_cand": "Loop candidates",
                "param.mesh_tri": "Target triangles",
                "param.mesh_smooth": "Smooth iterations",
                "param.preview_mesh": "Preview mesh",
                "param.view_mode": "View mode",
                "action.reset_camera": "Reset camera",
                "action.toggle_axis": "Show axis",
                "action.browse": "Browse…",
                "mode.realtime": "Realtime",
                "mode.playback": "Playback",
                "mode.record": "Record",
                "view.pointcloud": "Pointcloud",
                "view.mesh": "Mesh",
                "view.rgbd": "RGB-D",
                "view.disabled": "OpenGL 3.2 is not available on this system. 3D view disabled.",
                "export.target": "Export target",
                "export.target.mesh": "Mesh",
                "export.target.pointcloud": "Pointcloud",
                "export.format": "Format",
                "export.options": "Options",
                "export.option.simplify": "Simplify",
                "export.option.smooth": "Smooth",
                "export.option.color": "With color",
                "export.option.coords": "Coordinate system",
                "export.coords.opengl": "Open3D (right-handed)",
                "export.coords.unity": "Unity (left-handed)",
                "export.output": "Output path",
                "export.output.placeholder": "Choose output path",
                "export.naming": "Default name",
                "export.action.export": "Export",
                "export.action.exporting": "Exporting…",
                "session.save": "Save session",
                "session.load": "Load session",
                "log.title": "Log",
                "log.filter.info": "Info",
                "log.filter.warn": "Warn",
                "log.filter.error": "Error",
                "log.copy": "Copy",
                "log.clear": "Clear",
                "log.autoscroll": "Auto-scroll",
                "banner.tracking_lost.body": "Tracking lost: move slowly, add texture, or adjust depth truncation.",
                "banner.tracking_lost.action": "View tips",
                "banner.tracking_lost.tips": "Tips:\n- Move slower and steadily\n- Increase distance to the object\n- Add texture (stickers/newspaper)\n- Reduce depth_trunc",
                "settings.title": "Settings",
                "settings.language": "Language",
                "settings.language.en": "English (US)",
                "settings.language.zh": "中文（简体）",
                "settings.theme": "Theme",
                "settings.theme.system": "System",
                "settings.theme.light": "Light",
                "settings.theme.dark": "Dark",
                "settings.performance": "Performance",
                "settings.performance.high": "High",
                "settings.performance.balanced": "Balanced",
                "settings.performance.low": "Low",
                "settings.shortcuts": "Shortcuts",
                "settings.shortcuts.hint": "Common shortcuts will appear here.",
                "tooltip.range": "Range: {min}–{max} {unit}",
                "info.intrinsics": "Intrinsics fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}",
                "error.param.title": "Parameter error",
                "error.device.title": "Device error",
                "error.generic.title": "Error",
                "error.bag.required": "Please choose a bag path.",
                "error.bag.extension": "Bag file must end with .bag",
                "error.bag.dir_missing": "Bag directory does not exist: {path}",
                "error.bag.dir_unwritable": "Bag directory is not writable: {path}",
                "error.bag.missing": "Bag file not found: {path}",
                "opengl.title": "OpenGL Diagnostics",
                "opengl.fail": "Failed to query OpenGL info",
            },
        }

    def _t(self, key: str) -> str:
        lang_pack = self._translations.get(self._lang, {})
        if key in lang_pack:
            return lang_pack[key]
        return self._translations.get("en-US", {}).get(key, self._translations.get("zh-CN", {}).get(key, key))

    def _set_language(self, lang: str) -> None:
        self._lang = lang
        self._config.setdefault("ui", {})["language"] = lang
        self._save_config()
        self._apply_language()

    def _save_config(self) -> None:
        try:
            save_config(self._config_path, self._config)
        except Exception as exc:
            self.append_log(f"配置保存失败: {exc}", level="warn")

    def _apply_language(self) -> None:
        self.setWindowTitle(self._t("app.title"))
        self.app_title_label.setText(self._t("app.title"))
        self.device_chip.setText(
            self._t("header.device.connected") if self._device_connected else self._t("header.device.disconnected")
        )
        self.fps_label.setText(
            self._t("header.fps") + f": {self._last_fps:.1f}" if self._last_fps else self._t("header.fps") + ": --"
        )
        aligned = bool(self._config.get("device", {}).get("align_to_color", True))
        self.align_chip.setText(self._t("header.aligned") if aligned else self._t("header.not_aligned"))
        self.opengl_diag_btn.setText(self._t("header.diagnostics"))
        self.settings_btn.setText(self._t("header.settings"))
        self.exit_btn.setText(self._t("header.exit"))

        self.tracking_banner_label.setText(self._t("banner.tracking_lost.body"))
        self.tracking_banner_action.setText(self._t("banner.tracking_lost.action"))

        self.start_btn.setText(
            self._t("workflow.primary.start") if self._state == ScannerState.IDLE else self._t("workflow.primary.stop")
        )
        self.pause_btn.setText(
            self._t("workflow.secondary.resume") if self._state == ScannerState.PAUSED else self._t("workflow.secondary.pause")
        )
        self.state_label.setText(self._state_text(self._state))
        status = getattr(self, "_last_tracking_status", "INIT")
        self.tracking_pill.setText(self._tracking_text(status))
        self.tracking_status_label.setText(self._tracking_text(status))

        self.params_title_label.setText(self._t("section.parameters"))
        self.basic_btn.setText(self._t("toggle.basic"))
        self.advanced_btn.setText(self._t("toggle.advanced"))
        self.restore_defaults_btn.setText(self._t("action.restore_defaults"))

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

        def range_tip(min_v: float, max_v: float, unit: str) -> str:
            return self._t("tooltip.range").format(min=min_v, max=max_v, unit=unit)

        self.voxel_spin.setToolTip(range_tip(0.001, 0.02, " m"))
        self.sdf_spin.setToolTip(range_tip(0.005, 0.1, " m"))
        self.depth_trunc_spin.setToolTip(range_tip(0.3, 3.0, " m"))
        self.key_trans_spin.setToolTip(range_tip(0.005, 0.2, " m"))
        self.key_rot_spin.setToolTip(range_tip(1.0, 30.0, " °"))
        self.loop_gap_spin.setToolTip(range_tip(10, 200, " f"))
        self.loop_dist_spin.setToolTip(range_tip(0.1, 2.0, " m"))
        self.loop_candidates_spin.setToolTip(range_tip(1, 20, ""))
        self.simplify_spin.setToolTip(range_tip(1000, 2000000, " tris"))
        self.smooth_spin.setToolTip(range_tip(0, 50, ""))

        self.rgb_title_label.setText(self._t("preview.rgb.title"))
        self.depth_title_label.setText(self._t("preview.depth.title"))
        if hasattr(self, "viewer_placeholder"):
            self.viewer_placeholder.setText(self._t("view.pointcloud"))
        self.rgb_align_label.setText(
            self._t("preview.aligned")
            if self._config.get("device", {}).get("align_to_color", True)
            else self._t("preview.not_aligned")
        )
        self.depth_align_label.setText(
            self._t("preview.aligned")
            if self._config.get("device", {}).get("align_to_color", True)
            else self._t("preview.not_aligned")
        )
        align_state = "ok" if self._config.get("device", {}).get("align_to_color", True) else "warn"
        self.rgb_status_dot.setProperty("state", align_state)
        self.depth_status_dot.setProperty("state", align_state)
        self.rgb_status_dot.style().unpolish(self.rgb_status_dot)
        self.rgb_status_dot.style().polish(self.rgb_status_dot)
        self.depth_status_dot.style().unpolish(self.depth_status_dot)
        self.depth_status_dot.style().polish(self.depth_status_dot)
        self.reset_camera_btn.setText(self._t("action.reset_camera"))
        self.toggle_axes_btn.setText(self._t("action.toggle_axis"))

        self.label_mode.setText(self._t("workflow.mode"))
        self.bag_row_label.setText(self._t("workflow.bag_path"))
        self.bag_path_edit.setPlaceholderText(self._t("workflow.bag_path"))
        self.browse_bag_btn.setText(self._t("action.browse"))

        self.label_export_target.setText(self._t("export.target"))
        self.label_export_format.setText(self._t("export.format"))
        self.export_options_group.setTitle(self._t("export.options"))
        self.export_simplify_check.setText(self._t("export.option.simplify"))
        self.export_smooth_check.setText(self._t("export.option.smooth"))
        self.export_color_check.setText(self._t("export.option.color"))
        self.label_export_coords.setText(self._t("export.option.coords"))
        self.export_coords_combo.setItemText(0, self._t("export.coords.opengl"))
        self.export_coords_combo.setItemText(1, self._t("export.coords.unity"))
        self.label_export_path.setText(self._t("export.output"))
        self.export_path_btn.setText(self._t("action.browse"))
        self.export_path_edit.setPlaceholderText(self._t("export.output.placeholder"))
        self.export_run_btn.setText(self._t("export.action.export"))
        self.export_default_label.setText(self._default_export_hint())

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

        self._populate_mode_combo(self.mode_combo.currentData() or self.mode_combo.currentText())
        self._populate_view_combo(self.view_combo.currentData() or self.view_combo.currentText())
        self._populate_export_target_combo(self.export_target_combo.currentData() or self.export_target_combo.currentText())
        self._update_export_formats()
        self._update_export_options()
        if not self._enable_3d_viewer:
            self.view_combo.setToolTip(self._t("view.disabled"))

        self.params_toolbox.setItemText(0, self._t("group.reconstruction"))
        self.params_toolbox.setItemText(1, self._t("group.tracking"))
        self.params_toolbox.setItemText(2, self._t("group.loop"))
        self.params_toolbox.setItemText(3, self._t("group.mesh"))
        self.params_toolbox.setItemText(4, self._t("group.view"))

        self.sidebar_params_btn.setText(self._t("section.parameters"))
        self.sidebar_export_btn.setText(self._t("section.export"))
        self.session_group.setTitle(self._t("section.session"))
        self._on_mode_changed()

    def _resolve_initial_mode(self) -> str:
        scan_mode = str(self._config.get("scan", {}).get("mode", "realtime"))
        if scan_mode == "offline" or bool(self._config.get("device", {}).get("playback_bag", False)):
            return "playback"
        if bool(self._config.get("device", {}).get("record_bag", False)):
            return "record"
        return "realtime"

    def _populate_mode_combo(self, current: str) -> None:
        if current == "offline":
            current = "playback"
        blocker = QSignalBlocker(self.mode_combo)
        self.mode_combo.clear()
        self.mode_combo.addItem(self._t("mode.realtime"), "realtime")
        self.mode_combo.addItem(self._t("mode.playback"), "playback")
        self.mode_combo.addItem(self._t("mode.record"), "record")
        for index in range(self.mode_combo.count()):
            if self.mode_combo.itemData(index) == current:
                self.mode_combo.setCurrentIndex(index)
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

    def _populate_export_target_combo(self, current: str) -> None:
        blocker = QSignalBlocker(self.export_target_combo)
        self.export_target_combo.clear()
        self.export_target_combo.addItem(self._t("export.target.mesh"), "mesh")
        self.export_target_combo.addItem(self._t("export.target.pointcloud"), "pointcloud")
        for index in range(self.export_target_combo.count()):
            if self.export_target_combo.itemData(index) == current:
                self.export_target_combo.setCurrentIndex(index)
                break
        del blocker

    def _apply_theme(self, theme: str) -> None:
        if theme not in ("light", "dark"):
            theme = "light"
        if theme == "light":
            palette = {
                "bg": "#F5F6FA",
                "surface": "#FFFFFF",
                "surface_alt": "#F1F3F7",
                "card": "#FFFFFF",
                "border": "#D9DEE7",
                "text": "#0B0F14",
                "text_muted": "#2E3440",
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
            QPushButton, QCheckBox, QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox, QToolButton {{
                color: {palette["text"]};
            }}
            #Header, #WorkflowCard, #LogPanel {{
                background: {palette["surface"]};
                border: 1px solid {palette["border"]};
                border-radius: 12px;
            }}
            #Toast {{
                background: {palette["surface_alt"]};
                border: 1px solid {palette["border"]};
                border-radius: 10px;
            }}
            #Toast[level="error"] {{
                border-color: {palette["error"]};
            }}
            #PreviewCard {{
                background: {palette["card"]};
                border: 1px solid {palette["border"]};
                border-radius: 12px;
            }}
            #PreviewStatusDot {{
                border-radius: 4px;
                background: rgba(85, 211, 139, 0.9);
            }}
            #PreviewStatusDot[state="warn"] {{
                background: rgba(246, 196, 83, 0.9);
            }}
            #DeviceChip {{
                padding: 4px 10px;
                border-radius: 12px;
                border: 1px solid {palette["border"]};
                background: {palette["surface_alt"]};
                color: {palette["text"]};
            }}
            #AlignmentChip {{
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
            #TrackingPill[state="init"] {{
                background: rgba(43, 124, 255, 0.18);
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
            #SidebarToggle {{
                padding: 6px 10px;
                border-radius: 8px;
            }}
            QToolBox::tab {{
                background: {palette["surface_alt"]};
                border: 1px solid {palette["border"]};
                border-radius: 8px;
                padding: 6px 10px;
                margin-top: 6px;
            }}
            QToolBox::tab:selected {{
                background: {palette["surface"]};
                border-color: {palette["accent"]};
            }}
            QGroupBox {{
                border: 1px solid {palette["border"]};
                border-radius: 8px;
                margin-top: 6px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 6px;
                color: {palette["text_muted"]};
            }}
            #TrackingBanner {{
                background: {palette["surface_alt"]};
                border: 1px solid {palette["warn"]};
                border-radius: 10px;
            }}
            #PreviewMeta, #PreviewAlign, #UnitLabel, #InfoLabel, #ExportHint {{
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
        if status == "INIT":
            return self._t("workflow.tracking.init")
        if status == "LOST":
            return self._t("workflow.tracking.lost")
        if status == "WARN":
            return self._t("workflow.tracking.ok")
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
        export_type = self.export_target_combo.currentData() or self.export_target_combo.currentText().lower()
        self.export_format_combo.clear()
        if export_type == "mesh":
            self.export_format_combo.addItems(["ply", "obj", "glb"])
            default_fmt = self._config.get("export", {}).get("default_mesh_format", "ply")
        else:
            self.export_format_combo.addItems(["ply", "pcd"])
            default_fmt = self._config.get("export", {}).get("default_cloud_format", "ply")
        if default_fmt in [self.export_format_combo.itemText(i) for i in range(self.export_format_combo.count())]:
            self.export_format_combo.setCurrentText(default_fmt)
        self._update_export_options()

    def _update_export_options(self) -> None:
        export_type = self.export_target_combo.currentData() or self.export_target_combo.currentText().lower()
        is_mesh = export_type == "mesh"
        self.export_simplify_check.setVisible(is_mesh)
        self.export_smooth_check.setVisible(is_mesh)
        self.export_color_check.setVisible(True)
        self.label_export_coords.setVisible(True)
        self.export_coords_combo.setVisible(True)
        self.export_default_label.setText(self._default_export_hint())

    def _on_mode_changed(self) -> None:
        mode = self.mode_combo.currentData() or self.mode_combo.currentText()
        show_bag = mode in ("playback", "record")
        self.bag_row.setVisible(show_bag)
        self.bag_row_label.setVisible(show_bag)

    def _set_sidebar_page(self, index: int) -> None:
        self.sidebar_stack.setCurrentIndex(index)

    def _default_export_hint(self) -> str:
        base_name = "mesh" if (self.export_target_combo.currentData() or "mesh") == "mesh" else "cloud"
        if self._session is not None:
            base_name = f"{self._session.session_id}_{base_name}"
        return f"{self._t('export.naming')}: {base_name}"

    def _default_export_basename(self, export_type: str) -> str:
        base_name = "mesh" if export_type == "mesh" else "cloud"
        if self._session is not None:
            base_name = f"{self._session.session_id}_{base_name}"
        return base_name

    def _choose_export_path(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        filename, _ = QFileDialog.getSaveFileName(self, self._t("export.output"), "", "All Files (*)")
        if not filename:
            return
        self.export_path_edit.setText(filename)
        self._export_path_override = Path(filename)

    def _choose_bag_path(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        filename, _ = QFileDialog.getSaveFileName(self, self._t("workflow.bag_path"), "", "RealSense bag (*.bag)")
        if not filename:
            return
        self.bag_path_edit.setText(filename)

    def _refresh_log_view(self) -> None:
        show_info = self.log_info_chk.isChecked()
        show_warn = self.log_warn_chk.isChecked()
        show_error = self.log_error_chk.isChecked()
        lines: list[str] = []
        for level, message in self._log_entries:
            if level == "info" and not show_info:
                continue
            if level == "warn" and not show_warn:
                continue
            if level == "error" and not show_error:
                continue
            safe = html.escape(message)
            if level == "error":
                lines.append(f"<span style='color:#F26666;'>[ERROR] {safe}</span>")
            elif level == "warn":
                lines.append(f"<span style='color:#F6C453;'>[WARN] {safe}</span>")
            else:
                lines.append(f"[INFO] {safe}")
        self.log_text.setHtml("<br/>".join(lines))
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

    def _toggle_log_panel(self) -> None:
        self._log_collapsed = not self._log_collapsed
        self.log_text.setVisible(not self._log_collapsed)
        self.log_toggle_btn.setText("▸" if self._log_collapsed else "▾")

    def _show_toast(self, message: str, level: str = "info") -> None:
        self.toast_label.setText(message)
        self.toast.setProperty("level", level)
        self.toast.setVisible(True)
        self.toast.style().unpolish(self.toast)
        self.toast.style().polish(self.toast)
        if self._toast_timer is None:
            self._toast_timer = QTimer(self)
            self._toast_timer.setSingleShot(True)
            self._toast_timer.timeout.connect(lambda: self.toast.setVisible(False))
        self._toast_timer.start(3500)

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

    def _open_settings(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle(self._t("settings.title"))
        dialog.setModal(True)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(8)

        lang_combo = QComboBox(dialog)
        lang_combo.addItem(self._t("settings.language.en"), "en-US")
        lang_combo.addItem(self._t("settings.language.zh"), "zh-CN")
        for idx in range(lang_combo.count()):
            if lang_combo.itemData(idx) == self._lang:
                lang_combo.setCurrentIndex(idx)
                break
        form.addRow(self._t("settings.language"), lang_combo)

        theme_combo = QComboBox(dialog)
        theme_combo.addItem(self._t("settings.theme.system"), "system")
        theme_combo.addItem(self._t("settings.theme.light"), "light")
        theme_combo.addItem(self._t("settings.theme.dark"), "dark")
        current_theme = str(self._config.get("ui", {}).get("theme", "light"))
        for idx in range(theme_combo.count()):
            if theme_combo.itemData(idx) == current_theme:
                theme_combo.setCurrentIndex(idx)
                break
        form.addRow(self._t("settings.theme"), theme_combo)

        perf_combo = QComboBox(dialog)
        perf_combo.addItem(self._t("settings.performance.high"), "high")
        perf_combo.addItem(self._t("settings.performance.balanced"), "balanced")
        perf_combo.addItem(self._t("settings.performance.low"), "low")
        current_perf = str(self._config.get("ui", {}).get("performance", "balanced"))
        for idx in range(perf_combo.count()):
            if perf_combo.itemData(idx) == current_perf:
                perf_combo.setCurrentIndex(idx)
                break
        form.addRow(self._t("settings.performance"), perf_combo)

        layout.addLayout(form)

        shortcuts_group = QGroupBox(self._t("settings.shortcuts"))
        shortcuts_layout = QVBoxLayout(shortcuts_group)
        shortcuts_hint = QLabel(self._t("settings.shortcuts.hint"))
        shortcuts_hint.setWordWrap(True)
        shortcuts_layout.addWidget(shortcuts_hint)
        layout.addWidget(shortcuts_group)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        def _apply_language_setting() -> None:
            lang = lang_combo.currentData()
            if lang and lang != self._lang:
                self._set_language(str(lang))

        def _apply_theme_setting() -> None:
            theme = theme_combo.currentData() or "light"
            if theme == "system":
                theme = "light"
            self._config.setdefault("ui", {})["theme"] = theme
            self._apply_theme(str(theme))
            self._save_config()

        def _apply_perf_setting() -> None:
            perf = perf_combo.currentData() or "balanced"
            self._config.setdefault("ui", {})["performance"] = perf
            self._save_config()

        lang_combo.currentIndexChanged.connect(lambda _: _apply_language_setting())
        theme_combo.currentIndexChanged.connect(lambda _: _apply_theme_setting())
        perf_combo.currentIndexChanged.connect(lambda _: _apply_perf_setting())

        dialog.exec()

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
            ui_mode = self.mode_combo.currentData() or self.mode_combo.currentText()
            self._config["device"]["bag_path"] = self.bag_path_edit.text().strip()
            if ui_mode == "playback":
                self._config["scan"]["mode"] = "offline"
                bag_path = self._validate_bag_path("playback")
                if bag_path is None:
                    return
                self._config["device"]["bag_path"] = str(bag_path)
                self._config["device"]["playback_bag"] = True
                self._config["device"]["record_bag"] = False
                self._config["device"]["playback_real_time"] = False
            elif ui_mode == "record":
                self._config["scan"]["mode"] = "realtime"
                self._config["device"]["playback_bag"] = False
                self._config["device"]["record_bag"] = True
                bag_path = self._validate_bag_path("record")
                if bag_path is None:
                    return
                self._config["device"]["bag_path"] = str(bag_path)
            else:
                self._config["scan"]["mode"] = "realtime"
                self._config["device"]["playback_bag"] = False
                self._config["device"]["record_bag"] = False

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
            export_type = self.export_target_combo.currentData() or self.export_target_combo.currentText().lower()
            fmt = self.export_format_combo.currentText()
            base_name = self._default_export_basename(export_type)
            if self._export_path_override is not None:
                export_dir = ensure_dir(self._export_path_override.parent)
                base_name = self._export_path_override.stem
            mesh = self._current_mesh
            pcd = self._current_pcd
            o3d = _require_open3d()
            include_color = self.export_color_check.isChecked()
            coords = self.export_coords_combo.currentData() or "open3d"
            if export_type == "mesh" and mesh is not None:
                from scanner.geometry.exporter import export_mesh
                from scanner.geometry.postprocess import postprocess_mesh

                if not include_color:
                    mesh = mesh.clone()
                    mesh.vertex_colors = o3d.utility.Vector3dVector()
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
                            "include_color": include_color,
                            "coords": coords,
                        },
                    }
                )
                self.append_log(f"网格已导出: {path}")

            if export_type != "mesh" and pcd is not None:
                from scanner.geometry.exporter import export_point_cloud

                if not include_color:
                    pcd = pcd.clone()
                    pcd.colors = o3d.utility.Vector3dVector()
                path = export_point_cloud(pcd, export_dir / base_name, fmt)
                self._session.exports.append(
                    {
                        "type": "cloud",
                        "path": str(path),
                        "format": fmt,
                        "params": {"include_color": include_color, "coords": coords},
                    }
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
        self._show_error(self._t("error.device.title"), message)
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
        QMessageBox.critical(self, self._t("error.generic.title"), message)

    def _validate_bag_path(self, mode: str) -> Optional[Path]:
        text = self.bag_path_edit.text().strip()
        if not text:
            self._show_error(self._t("error.param.title"), self._t("error.bag.required"))
            return None
        path = Path(text).expanduser()
        if path.suffix.lower() != ".bag":
            self._show_error(self._t("error.param.title"), self._t("error.bag.extension"))
            return None
        path = path.resolve()
        if mode == "record":
            parent = path.parent
            parent.mkdir(parents=True, exist_ok=True)
            if not parent.is_dir():
                self._show_error(self._t("error.param.title"), self._t("error.bag.dir_missing").format(path=parent))
                return None
            if not os.access(parent, os.W_OK):
                self._show_error(self._t("error.param.title"), self._t("error.bag.dir_unwritable").format(path=parent))
                return None
        else:
            if not path.exists():
                self._show_error(self._t("error.param.title"), self._t("error.bag.missing").format(path=path))
                return None
        self.bag_path_edit.setText(str(path))
        return path

    def _check_device_available(self) -> bool:
        try:
            from scanner.devices.realsense_device import RealSenseDevice

            return RealSenseDevice.has_device()
        except Exception as exc:
            self._show_error(self._t("error.device.title"), str(exc))
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
        self.export_default_label.setText(self._default_export_hint())
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
            self._t("info.intrinsics").format(fx=intr.fx, fy=intr.fy, cx=intr.cx, cy=intr.cy)
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
        if level == "error":
            self._show_toast(message, level="error")
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
            self.export_default_label.setText(self._default_export_hint())
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

    def _restore_default_params(self) -> None:
        defaults = self._defaults
        fusion = defaults.get("fusion", {})
        keyframe = defaults.get("keyframe", {})
        loop_cfg = defaults.get("loop_closure", {})
        mesh_cfg = defaults.get("mesh", {})
        self.voxel_spin.setValue(float(fusion.get("voxel_length", self.voxel_spin.value())))
        self.sdf_spin.setValue(float(fusion.get("sdf_trunc", self.sdf_spin.value())))
        self.depth_trunc_spin.setValue(float(fusion.get("depth_trunc", self.depth_trunc_spin.value())))

        self.key_trans_spin.setValue(float(keyframe.get("min_translation", self.key_trans_spin.value())))
        self.key_rot_spin.setValue(float(keyframe.get("min_rotation_deg", self.key_rot_spin.value())))

        self.loop_gap_spin.setValue(int(loop_cfg.get("temporal_gap", self.loop_gap_spin.value())))
        self.loop_dist_spin.setValue(float(loop_cfg.get("max_distance", self.loop_dist_spin.value())))
        self.loop_candidates_spin.setValue(int(loop_cfg.get("max_candidates", self.loop_candidates_spin.value())))

        self.simplify_spin.setValue(int(mesh_cfg.get("simplify_target_triangles", self.simplify_spin.value())))
        self.smooth_spin.setValue(int(mesh_cfg.get("smooth_iterations", self.smooth_spin.value())))
        self.preview_mesh_checkbox.setChecked(bool(fusion.get("preview_mesh", self.preview_mesh_checkbox.isChecked())))

        default_view = str(defaults.get("ui", {}).get("default_view", "rgbd"))
        self._populate_view_combo(default_view)

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
        self.start_btn.setEnabled(state not in (ScannerState.STOPPING, ScannerState.EXPORTING))
        self.mode_combo.setEnabled(is_idle)
        self.bag_path_edit.setEnabled(is_idle)
        self.browse_bag_btn.setEnabled(is_idle)
        self.pause_btn.setEnabled(state in (ScannerState.SCANNING, ScannerState.RECORDING, ScannerState.PLAYBACK, ScannerState.PREVIEW, ScannerState.PAUSED))
        if state == ScannerState.IDLE:
            self._last_tracking_status = "INIT"
            self.tracking_pill.setText(self._tracking_text("INIT"))
            self.tracking_pill.setProperty("state", "init")
            self.tracking_status_label.setText(self._tracking_text("INIT"))
            self.tracking_pill.style().unpolish(self.tracking_pill)
            self.tracking_pill.style().polish(self.tracking_pill)
        if state == ScannerState.EXPORTING:
            self.export_run_btn.setEnabled(False)
            self.export_run_btn.setText(self._t("export.action.exporting"))
        else:
            self.export_run_btn.setEnabled(True)
            self.export_run_btn.setText(self._t("export.action.export"))

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
