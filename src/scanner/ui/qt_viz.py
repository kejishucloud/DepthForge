from __future__ import annotations

from typing import Optional

import numpy as np
import open3d as o3d
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6.QtWidgets import QVBoxLayout, QWidget


class Qt3DViewer(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._plotter = QtInteractor(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._plotter)
        self._axes = self._plotter.add_axes()
        self._plotter.set_background("#1b1e23")
        self._show_axes = True

    def show_point_cloud(self, pcd: o3d.geometry.PointCloud) -> None:
        """
        在 Qt 视图中显示点云。
        :param pcd: Open3D 点云
        :return: None
        """
        poly = self._pcd_to_polydata(pcd)
        self._plotter.clear()
        if poly is not None:
            self._plotter.add_mesh(poly, scalars="rgb" if "rgb" in poly.point_data else None, rgb=True, point_size=2.0)
        self._plotter.reset_camera()

    def show_mesh(self, mesh: o3d.geometry.TriangleMesh) -> None:
        """
        在 Qt 视图中显示网格。
        :param mesh: Open3D 网格
        :return: None
        """
        poly = self._mesh_to_polydata(mesh)
        self._plotter.clear()
        if poly is not None:
            self._plotter.add_mesh(poly, color="white", smooth_shading=True)
        self._plotter.reset_camera()

    def reset_camera(self) -> None:
        """
        重置视角到默认相机位置。
        :return: None
        """
        self._plotter.reset_camera()

    def toggle_axes(self) -> None:
        """
        切换坐标轴显示状态。
        :return: None
        """
        self._show_axes = not self._show_axes
        if self._show_axes:
            self._plotter.add_axes()
        else:
            self._plotter.hide_axes()

    def _pcd_to_polydata(self, pcd: o3d.geometry.PointCloud) -> Optional[pv.PolyData]:
        """
        将 Open3D 点云转换为 PyVista 数据。
        :param pcd: Open3D 点云
        :return: PyVista PolyData（若为空则返回 None）
        """
        points = np.asarray(pcd.points)
        if points.size == 0:
            return None
        poly = pv.PolyData(points)
        if pcd.has_colors():
            colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)
            poly.point_data["rgb"] = colors
        return poly

    def _mesh_to_polydata(self, mesh: o3d.geometry.TriangleMesh) -> Optional[pv.PolyData]:
        """
        将 Open3D 网格转换为 PyVista 数据。
        :param mesh: Open3D 网格
        :return: PyVista PolyData（若为空则返回 None）
        """
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        if vertices.size == 0 or triangles.size == 0:
            return None
        faces = np.hstack([np.full((triangles.shape[0], 1), 3), triangles]).astype(np.int64)
        faces = faces.reshape(-1)
        poly = pv.PolyData(vertices, faces)
        return poly
