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
        :return: None
        """
        self._volume = self._create_volume()

    def integrate_frame(self, frame: FrameBundle, pose: np.ndarray) -> None:
        """
        融合单帧 RGB-D 到 TSDF 体。
        :param frame: 当前帧 RGB-D 数据
        :param pose: 相机位姿（world_T_cam）
        :return: None
        """
        rgbd, intrinsic = self._rgbd_from_frame(frame)
        # 约定 pose 为 world_T_cam（相机到世界）
        self._volume.integrate(rgbd, intrinsic, pose)

    def integrate_keyframes(self, keyframes: Iterable[Keyframe], poses: Iterable[np.ndarray]) -> None:
        """
        使用优化后的位姿批量重融合关键帧。
        :param keyframes: 关键帧序列
        :param poses: 对应的优化位姿序列
        :return: None
        """
        self.reset()
        for keyframe, pose in zip(keyframes, poses):
            rgbd, intrinsic = self._rgbd_from_keyframe(keyframe)
            self._volume.integrate(rgbd, intrinsic, pose)

    def extract_mesh(self) -> o3d.geometry.TriangleMesh:
        """
        从 TSDF 体提取三角网格。
        :return: 三角网格
        """
        return self._volume.extract_triangle_mesh()

    def extract_point_cloud(self) -> o3d.geometry.PointCloud:
        """
        从 TSDF 体提取点云。
        :return: 点云
        """
        return self._volume.extract_point_cloud()

    def extract_preview_point_cloud(self, voxel: Optional[float] = None) -> o3d.geometry.PointCloud:
        """
        提取用于预览的点云，并可选进行体素下采样。
        :param voxel: 下采样体素大小（None 或 <=0 表示不下采样）
        :return: 预览点云
        """
        pcd = self._volume.extract_point_cloud()
        if voxel is not None and voxel > 0:
            pcd = pcd.voxel_down_sample(voxel)
        return pcd

    def extract_preview_mesh(self) -> o3d.geometry.TriangleMesh:
        """
        提取用于预览的网格并计算法线。
        :return: 预览网格
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
            color_type=self._resolve_color_type(color_type),
        )

    @staticmethod
    def _resolve_color_type(color_type: Optional[str]) -> o3d.pipelines.integration.TSDFVolumeColorType:
        requested = (color_type or "").strip().lower()
        tsdf_enum = o3d.pipelines.integration.TSDFVolumeColorType
        if requested in {"rgb", "rgb8", "color"}:
            return tsdf_enum.RGB8
        if requested in {"gray", "gray32"} and hasattr(tsdf_enum, "Gray32"):
            return tsdf_enum.Gray32
        # "None" is a reserved keyword in Python; use getattr for compatibility.
        for name in ("None", "NoColor", "NONE"):
            if hasattr(tsdf_enum, name):
                return getattr(tsdf_enum, name)
        return tsdf_enum.RGB8

    def _rgbd_from_frame(self, frame: FrameBundle) -> Tuple[o3d.geometry.RGBDImage, o3d.camera.PinholeCameraIntrinsic]:
        """
        将帧数据转换为 Open3D RGBDImage 与内参。
        :param frame: RGB-D 帧
        :return: (RGBDImage, 相机内参)
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
        将关键帧转换为 Open3D RGBDImage 与内参。
        :param keyframe: 关键帧
        :return: (RGBDImage, 相机内参)
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
