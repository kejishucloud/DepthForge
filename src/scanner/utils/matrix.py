from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class PoseDelta:
    translation: float
    rotation_deg: float


def identity() -> np.ndarray:
    """
    生成 4x4 单位位姿矩阵。
    :return: 返回介绍
    """
    return np.eye(4, dtype=np.float64)


def pose_from_rt(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """
    由旋转和平移构造 4x4 位姿矩阵。
    :param rotation: 参数介绍
    :param translation: 参数介绍
    :return: 返回介绍
    """
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    return pose


def invert(pose: np.ndarray) -> np.ndarray:
    """
    计算位姿矩阵逆。
    :param pose: 参数介绍
    :return: 返回介绍
    """
    r = pose[:3, :3]
    t = pose[:3, 3]
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = r.T
    inv[:3, 3] = -r.T @ t
    return inv


def transform_points(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """
    对点云执行位姿变换。
    :param points: 参数介绍
    :param pose: 参数介绍
    :return: 返回介绍
    """
    if points.shape[1] != 3:
        raise ValueError("points must be Nx3")
    ones = np.ones((points.shape[0], 1), dtype=points.dtype)
    homo = np.hstack([points, ones])
    transformed = (pose @ homo.T).T
    return transformed[:, :3]


def rotation_angle_deg(pose_a: np.ndarray, pose_b: np.ndarray) -> float:
    """
    计算两个位姿的相对旋转角度（度）。
    :param pose_a: 参数介绍
    :param pose_b: 参数介绍
    :return: 返回介绍
    """
    r = pose_a[:3, :3].T @ pose_b[:3, :3]
    trace_val = np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0)
    angle = math.degrees(math.acos(trace_val))
    return float(angle)


def pose_delta(pose_a: np.ndarray, pose_b: np.ndarray) -> PoseDelta:
    """
    计算位姿间的平移与旋转差。
    :param pose_a: 参数介绍
    :param pose_b: 参数介绍
    :return: 返回介绍
    """
    translation = float(np.linalg.norm(pose_a[:3, 3] - pose_b[:3, 3]))
    rotation = rotation_angle_deg(pose_a, pose_b)
    return PoseDelta(translation=translation, rotation_deg=rotation)


def is_valid_pose(pose: np.ndarray) -> bool:
    """
    校验位姿矩阵形状与数值有效性。
    :param pose: 参数介绍
    :return: 返回介绍
    """
    if pose.shape != (4, 4):
        return False
    if not np.isfinite(pose).all():
        return False
    return True


def pose_distance(pose_a: np.ndarray, pose_b: np.ndarray) -> Tuple[float, float]:
    """
    返回平移距离与旋转角度。
    :param pose_a: 参数介绍
    :param pose_b: 参数介绍
    :return: 返回介绍
    """
    delta = pose_delta(pose_a, pose_b)
    return delta.translation, delta.rotation_deg
