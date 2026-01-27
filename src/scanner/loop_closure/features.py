from __future__ import annotations

from typing import Any, Dict, Tuple

import open3d as o3d

from scanner.core.types import Keyframe


def rgbd_from_keyframe(keyframe: Keyframe, depth_trunc: float) -> o3d.geometry.RGBDImage:
    """
    将关键帧转换为 Open3D RGBDImage。
    :param keyframe: 参数介绍
    :param depth_trunc: 参数介绍
    :return: 返回介绍
    """
    color = keyframe.color.copy()
    depth = keyframe.depth.astype(np.uint16)
    color_o3d = o3d.geometry.Image(color)
    depth_o3d = o3d.geometry.Image(depth)
    depth_scale = 1000.0
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d,
        depth_o3d,
        depth_scale=depth_scale,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False,
    )


def intrinsic_from_keyframe(intr: Dict[str, Any]) -> o3d.camera.PinholeCameraIntrinsic:
    """
    从内参字典构建 Open3D 相机内参。
    :param intr: 参数介绍
    :return: 返回介绍
    """
    return o3d.camera.PinholeCameraIntrinsic(
        int(intr["width"]),
        int(intr["height"]),
        float(intr["fx"]),
        float(intr["fy"]),
        float(intr["cx"]),
        float(intr["cy"]),
    )


def build_point_cloud(keyframe: Keyframe, depth_trunc: float) -> o3d.geometry.PointCloud:
    """
    由关键帧生成点云用于配准。
    :param keyframe: 参数介绍
    :param depth_trunc: 参数介绍
    :return: 返回介绍
    """
    rgbd = rgbd_from_keyframe(keyframe, depth_trunc)
    intrinsic = intrinsic_from_keyframe(keyframe.intrinsics)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    return pcd


def compute_fpfh(pcd: o3d.geometry.PointCloud, voxel_size: float) -> Tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
    """
    计算下采样点云及其 FPFH 特征。
    :param pcd: 参数介绍
    :param voxel_size: 参数介绍
    :return: 返回介绍
    """
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30)
    )
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0, max_nn=100),
    )
    return pcd_down, fpfh
