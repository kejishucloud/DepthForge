from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d

from scanner.core.types import FrameBundle, Intrinsics


@dataclass
class OdomResult:
    success: bool
    pose: np.ndarray
    fitness: float
    rmse: float
    info: np.ndarray


class RgbdOdometryTracker:
    def __init__(self, config: Dict[str, Any]) -> None:
        self._config = config

    def estimate(self, prev: FrameBundle, curr: FrameBundle, init_pose: np.ndarray) -> OdomResult:
        """
        通过 RGB-D 里程计估计相邻帧相对位姿。
        :param prev: 参数介绍
        :param curr: 参数介绍
        :param init_pose: 参数介绍
        :return: 返回介绍
        """
        prev_rgbd, curr_rgbd, intrinsic = self._build_rgbd(prev, curr)
        option = o3d.pipelines.odometry.OdometryOption()
        option.max_depth_diff = float(self._config.get("max_depth_diff", 0.07))
        option.min_depth = float(self._config.get("min_depth", 0.2))
        option.max_depth = float(self._config.get("max_depth", 1.0))
        levels = int(self._config.get("pyramid_levels", 3))
        option.iteration_number_per_pyramid_level = [20, 10, 5][:levels]

        jacobian = o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()
        success, trans, _ = o3d.pipelines.odometry.compute_rgbd_odometry(
            prev_rgbd,
            curr_rgbd,
            intrinsic,
            init_pose,
            jacobian,
            option,
        )
        # 使用评估函数输出稳定的 fitness / rmse
        pcd_prev = o3d.geometry.PointCloud.create_from_rgbd_image(prev_rgbd, intrinsic)
        pcd_curr = o3d.geometry.PointCloud.create_from_rgbd_image(curr_rgbd, intrinsic)
        voxel = float(self._config.get("eval_voxel", 0.02))
        pcd_prev = pcd_prev.voxel_down_sample(voxel)
        pcd_curr = pcd_curr.voxel_down_sample(voxel)
        eval_thresh = float(self._config.get("eval_max_correspondence", 0.03))
        evaluation = o3d.pipelines.registration.evaluate_registration(pcd_prev, pcd_curr, eval_thresh, trans)
        info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            pcd_prev, pcd_curr, eval_thresh, trans
        )
        return OdomResult(
            success=bool(success),
            pose=trans,
            fitness=float(evaluation.fitness),
            rmse=float(evaluation.inlier_rmse),
            info=info,
        )

    def refine_icp(
        self,
        prev: FrameBundle,
        curr: FrameBundle,
        init_pose: np.ndarray,
        max_correspondence: float,
    ) -> OdomResult:
        """
        使用 ICP 对里程计结果进行精配。
        :param prev: 参数介绍
        :param curr: 参数介绍
        :param init_pose: 参数介绍
        :param max_correspondence: 参数介绍
        :return: 返回介绍
        """
        prev_rgbd, curr_rgbd, intrinsic = self._build_rgbd(prev, curr)
        prev_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(prev_rgbd, intrinsic)
        curr_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(curr_rgbd, intrinsic)
        voxel = float(self._config.get("icp_voxel", 0.02))
        prev_pcd = prev_pcd.voxel_down_sample(voxel)
        curr_pcd = curr_pcd.voxel_down_sample(voxel)
        prev_pcd.estimate_normals()
        curr_pcd.estimate_normals()
        result = o3d.pipelines.registration.registration_icp(
            prev_pcd,
            curr_pcd,
            max_correspondence,
            init_pose,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        )
        info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            prev_pcd, curr_pcd, max_correspondence, result.transformation
        )
        return OdomResult(
            success=bool(result.fitness > 0.0),
            pose=result.transformation,
            fitness=float(result.fitness),
            rmse=float(result.inlier_rmse),
            info=info,
        )

    def _build_rgbd(
        self, prev: FrameBundle, curr: FrameBundle
    ) -> tuple[o3d.geometry.RGBDImage, o3d.geometry.RGBDImage, o3d.camera.PinholeCameraIntrinsic]:
        """
        函数介绍。
        :param prev: 参数介绍
        :param curr: 参数介绍
        :return: 返回介绍
        """
        intrinsic = self._intrinsic_from_intrinsics(curr.intrinsics)
        prev_rgbd = self._rgbd_from_frame(prev)
        curr_rgbd = self._rgbd_from_frame(curr)
        return prev_rgbd, curr_rgbd, intrinsic

    def _intrinsic_from_intrinsics(self, intr: Intrinsics) -> o3d.camera.PinholeCameraIntrinsic:
        """
        函数介绍。
        :param intr: 参数介绍
        :return: 返回介绍
        """
        return o3d.camera.PinholeCameraIntrinsic(
            int(intr.width),
            int(intr.height),
            float(intr.fx),
            float(intr.fy),
            float(intr.cx),
            float(intr.cy),
        )

    def _rgbd_from_frame(self, frame: FrameBundle) -> o3d.geometry.RGBDImage:
        """
        函数介绍。
        :param frame: 参数介绍
        :return: 返回介绍
        """
        color = frame.color_rgb.copy()
        depth = frame.depth_mm.astype(np.uint16)
        color_o3d = o3d.geometry.Image(color)
        depth_o3d = o3d.geometry.Image(depth)
        depth_scale = 1000.0
        depth_trunc = float(self._config.get("max_depth", 1.0))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d,
            depth_o3d,
            depth_scale=depth_scale,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
        )
        return rgbd
