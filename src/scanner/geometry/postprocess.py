from __future__ import annotations

from typing import Optional

import open3d as o3d
import numpy as np


def postprocess_mesh(
    mesh: o3d.geometry.TriangleMesh,
    simplify_target: Optional[int] = None,
    smooth_iterations: int = 0,
    remove_small: bool = True,
    min_triangles: int = 500,
    keep_largest: bool = True,
) -> o3d.geometry.TriangleMesh:
    """
    对网格执行清理、去小碎片与可选平滑/简化。
    :param mesh: 待处理网格
    :param simplify_target: 目标三角面数（None 表示不简化）
    :param smooth_iterations: 平滑迭代次数
    :param remove_small: 是否移除小连通片
    :param min_triangles: 小片段最小三角面数阈值
    :param keep_largest: 移除小片段时是否保留最大连通片
    :return: 处理后的网格
    """
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.compute_vertex_normals()

    if remove_small:
        clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
        cluster_n_triangles = list(cluster_n_triangles)
        triangles_to_remove = []
        if keep_largest and cluster_n_triangles:
            keep_cluster = int(np.argmax(cluster_n_triangles))
            for tri_idx, cluster_idx in enumerate(clusters):
                if int(cluster_idx) != keep_cluster:
                    triangles_to_remove.append(tri_idx)
        else:
            for tri_idx, cluster_idx in enumerate(clusters):
                if cluster_n_triangles[int(cluster_idx)] < min_triangles:
                    triangles_to_remove.append(tri_idx)
        mesh.remove_triangles_by_index(triangles_to_remove)
        mesh.remove_unreferenced_vertices()

    if simplify_target is not None and simplify_target > 0:
        mesh = mesh.simplify_quadric_decimation(simplify_target)
        mesh.remove_unreferenced_vertices()

    if smooth_iterations > 0:
        mesh = mesh.filter_smooth_simple(number_of_iterations=int(smooth_iterations))
        mesh.compute_vertex_normals()

    return mesh
