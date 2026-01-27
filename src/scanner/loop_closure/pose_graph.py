from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import open3d as o3d

from scanner.core.types import Keyframe
from scanner.loop_closure.candidates import CandidateConfig, generate_candidates
from scanner.loop_closure.features import build_point_cloud, compute_fpfh
from scanner.utils.matrix import invert


@dataclass
class PoseGraphResult:
    pose_graph: o3d.pipelines.registration.PoseGraph
    loop_edges: int
    odom_edges: int


class PoseGraphOptimizer:
    def __init__(self, config: Dict[str, Any]) -> None:
        self._config = config

    def build(self, keyframes: List[Keyframe]) -> PoseGraphResult:
        """
        构建包含里程计边与回环边的位姿图。
        :param keyframes: 关键帧列表
        :return: 位姿图与边统计结果
        """
        pose_graph = o3d.pipelines.registration.PoseGraph()
        if not keyframes:
            return PoseGraphResult(pose_graph=pose_graph, loop_edges=0, odom_edges=0)

        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(keyframes[0].pose))

        odom_edges = 0
        loop_edges = 0

        for i in range(1, len(keyframes)):
            # 相邻关键帧的相对位姿作为里程计边
            rel = invert(keyframes[i - 1].pose) @ keyframes[i].pose
            depth_trunc = float(self._config.get("depth_trunc", 1.0))
            source = build_point_cloud(keyframes[i - 1], depth_trunc)
            target = build_point_cloud(keyframes[i], depth_trunc)
            info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                source, target, float(self._config.get("icp", {}).get("max_correspondence", 0.04)), rel
            )
            pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(keyframes[i].pose))
            pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(i - 1, i, rel, info, uncertain=False)
            )
            odom_edges += 1

        if self._config.get("enable", True):
            loop_edges += self._add_loop_edges(keyframes, pose_graph)

        return PoseGraphResult(pose_graph=pose_graph, loop_edges=loop_edges, odom_edges=odom_edges)

    def optimize(self, pose_graph: o3d.pipelines.registration.PoseGraph) -> None:
        """
        执行位姿图全局优化。
        :param pose_graph: 待优化的位姿图
        :return: None
        """
        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=float(self._config.get("icp", {}).get("max_correspondence", 0.04)),
            edge_prune_threshold=0.25,
            reference_node=0,
        )
        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option,
        )

    def _add_loop_edges(self, keyframes: List[Keyframe], pose_graph: o3d.pipelines.registration.PoseGraph) -> int:
        """
        为候选关键帧对生成回环约束。
        :param keyframes: 关键帧列表
        :param pose_graph: 位姿图
        :return: 新增回环边数量
        """
        candidate_cfg = CandidateConfig(
            temporal_gap=int(self._config.get("temporal_gap", 30)),
            max_candidates=int(self._config.get("max_candidates", 5)),
            max_distance=float(self._config.get("max_distance", 0.4)),
            use_all_frames=bool(self._config.get("use_all_frames", False)),
        )
        pairs = generate_candidates(keyframes, candidate_cfg)
        if not pairs:
            return 0

        depth_trunc = float(self._config.get("depth_trunc", 1.0))
        voxel = float(self._config.get("ransac", {}).get("distance_threshold", 0.03))
        loop_edges = 0

        for i, j in pairs:
            # 生成候选对的点云并进行特征匹配
            source = build_point_cloud(keyframes[i], depth_trunc)
            target = build_point_cloud(keyframes[j], depth_trunc)
            source_down, source_fpfh = compute_fpfh(source, voxel)
            target_down, target_fpfh = compute_fpfh(target, voxel)
            source_down.estimate_normals()
            target_down.estimate_normals()

            result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                source_down,
                target_down,
                source_fpfh,
                target_fpfh,
                True,
                voxel,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                4,
                [
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel * 1.5),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                ],
                o3d.pipelines.registration.RANSACConvergenceCriteria(
                    int(self._config.get("ransac", {}).get("max_iterations", 4000)),
                    float(self._config.get("ransac", {}).get("confidence", 0.999)),
                ),
            )
            if result_ransac.fitness < 0.15:
                continue

            # RANSAC 通过后使用 ICP 精配
            icp_thresh = float(self._config.get("icp", {}).get("max_correspondence", 0.04))
            if self._config.get("use_colored_icp", True):
                result_icp = o3d.pipelines.registration.registration_colored_icp(
                    source_down,
                    target_down,
                    icp_thresh,
                    result_ransac.transformation,
                    o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                )
            else:
                result_icp = o3d.pipelines.registration.registration_icp(
                    source_down,
                    target_down,
                    icp_thresh,
                    result_ransac.transformation,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                )
            if result_icp.fitness < 0.2:
                continue

            info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                source_down,
                target_down,
                icp_thresh,
                result_icp.transformation,
            )
            pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(i, j, result_icp.transformation, info, uncertain=True)
            )
            loop_edges += 1

        return loop_edges
