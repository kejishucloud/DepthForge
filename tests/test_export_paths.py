from pathlib import Path

import numpy as np
import open3d as o3d

from scanner.geometry.exporter import export_mesh, export_point_cloud


def test_export_paths(tmp_path: Path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array([[0.0, 0.0, 0.0]]))
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)

    mesh_path = export_mesh(mesh, tmp_path / "mesh", "ply")
    cloud_path = export_point_cloud(pcd, tmp_path / "cloud", "ply")

    assert mesh_path.exists()
    assert cloud_path.exists()
