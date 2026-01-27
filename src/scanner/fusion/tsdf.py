from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import open3d as o3d

from scanner.core.types import FrameBundle, Keyframe


class TSDFVolumeManager:
    def __init__(self, config: Dict[str, Any]) -> None:
        self._config = config
        self._volume = self._create_volume()

    def reset(self) -> None:
        """
        重置 TSDF 融合体。
        :return: 返回介绍
        """
        self._volume = self._create_volume()

    def integrate_frame(self, frame: FrameBundle, pose: np.ndarray) -> None:
        """
        融合单帧 RGB-D 到 TSDF 体。
        :param frame: 参数介绍
        :param pose: 参数介绍
        :return: 返回介绍
        """
        rgbd, intrinsic = self._rgbd_from_frame(frame)
        # 约定 pose 为 world_T_cam（相机到世界）
        self._volume.integrate(rgbd, intrinsic, pose)

    def integrate_keyframes(self, keyframes: Iterable[Keyframe], poses: Iterable[np.ndarray]) -> None:
        """
        使用优化后的位姿批量重融合关键帧。
        :param keyframes: 参数介绍
        :param poses: 参数介绍
        :return: 返回介绍
        """
        self.reset()
        for keyframe, pose in zip(keyframes, poses):
            rgbd, intrinsic = self._rgbd_from_keyframe(keyframe)
            self._volume.integrate(rgbd, intrinsic, pose)

    def extract_mesh(self) -> o3d.geometry.TriangleMesh:
        """
        从 TSDF 体提取三角网格。
        :return: 返回介绍
        """
        return self._volume.extract_triangle_mesh()

    def extract_point_cloud(self) -> o3d.geometry.PointCloud:
        """
        从 TSDF 体提取点云。
        :return: 返回介绍
        """
        return self._volume.extract_point_cloud()

    def extract_preview_point_cloud(self, voxel: Optional[float] = None) -> o3d.geometry.PointCloud:
        """
        函数介绍。
        :param voxel: 参数介绍
        :return: 返回介绍
        """
        pcd = self._volume.extract_point_cloud()
        if voxel is not None and voxel > 0:
            pcd = pcd.voxel_down_sample(voxel)
        return pcd

    def extract_preview_mesh(self) -> o3d.geometry.TriangleMesh:
        """
        函数介绍。
        :return: 返回介绍
        """
        mesh = self._volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        return mesh

    def _create_volume(self) -> o3d.pipelines.integration.ScalableTSDFVolume:
        voxel_length = float(self._config.get("voxel_length", 0.004))
        sdf_trunc = float(self._config.get("sdf_trunc", 0.02))
        color_type = self._config.get("color_type", "rgb8")
        return o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_length,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
            if color_type == "rgb8"
            else o3d.pipelines.integration.TSDFVolumeColorType.None,
        )

    def _rgbd_from_frame(self, frame: FrameBundle) -> Tuple[o3d.geometry.RGBDImage, o3d.camera.PinholeCameraIntrinsic]:
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
        depth_trunc = float(self._config.get("depth_trunc", 1.0))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d,
            depth_o3d,
            depth_scale=depth_scale,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
        )
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            int(frame.intrinsics.width),
            int(frame.intrinsics.height),
            float(frame.intrinsics.fx),
            float(frame.intrinsics.fy),
            float(frame.intrinsics.cx),
            float(frame.intrinsics.cy),
        )
        return rgbd, intrinsic

    def _rgbd_from_keyframe(self, keyframe: Keyframe) -> Tuple[o3d.geometry.RGBDImage, o3d.camera.PinholeCameraIntrinsic]:
        """
        函数介绍。
        :param keyframe: 参数介绍
        :return: 返回介绍
        """
        color = keyframe.color.copy()
        depth = keyframe.depth.astype(np.uint16)
        color_o3d = o3d.geometry.Image(color)
        depth_o3d = o3d.geometry.Image(depth)
        depth_scale = 1000.0
        depth_trunc = float(self._config.get("depth_trunc", 1.0))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d,
            depth_o3d,
            depth_scale=depth_scale,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
        )
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            int(keyframe.intrinsics["width"]),
            int(keyframe.intrinsics["height"]),
            float(keyframe.intrinsics["fx"]),
            float(keyframe.intrinsics["fy"]),
            float(keyframe.intrinsics["cx"]),
            float(keyframe.intrinsics["cy"]),
        )
        return rgbd, intrinsic
