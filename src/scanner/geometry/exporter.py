from __future__ import annotations

from pathlib import Path
from typing import Literal

import open3d as o3d


MeshFormat = Literal["ply", "obj", "stl"]
CloudFormat = Literal["ply", "pcd"]


def export_mesh(mesh: o3d.geometry.TriangleMesh, path: Path, fmt: MeshFormat) -> Path:
    """
    导出三角网格到指定格式。
    :param mesh: 需要导出的网格
    :param path: 不含后缀的目标路径
    :param fmt: 导出格式（ply/obj/stl）
    :return: 实际导出文件路径
    """
    file_path = path.with_suffix(f".{fmt}")
    o3d.io.write_triangle_mesh(str(file_path), mesh, write_triangle_uvs=False)
    return file_path


def export_point_cloud(pcd: o3d.geometry.PointCloud, path: Path, fmt: CloudFormat) -> Path:
    """
    导出点云到指定格式。
    :param pcd: 需要导出的点云
    :param path: 不含后缀的目标路径
    :param fmt: 导出格式（ply/pcd）
    :return: 实际导出文件路径
    """
    file_path = path.with_suffix(f".{fmt}")
    o3d.io.write_point_cloud(str(file_path), pcd)
    return file_path
