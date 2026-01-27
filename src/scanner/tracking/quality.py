from __future__ import annotations

from dataclasses import dataclass

from scanner.core.types import TrackingStatus


@dataclass
class QualityThresholds:
    fitness_ok: float
    rmse_ok: float
    fitness_warn: float
    rmse_warn: float
    motion_translation_warn: float
    motion_translation_lost: float
    motion_rotation_warn: float
    motion_rotation_lost: float


def evaluate_quality(
    inlier_ratio: float,
    rmse: float,
    delta_translation: float,
    delta_rotation: float,
    thresholds: QualityThresholds,
) -> TrackingStatus:
    """
    根据 fitness/rmse 阈值评估跟踪质量。
    :param inlier_ratio: 参数介绍
    :param rmse: 参数介绍
    :param delta_translation: 参数介绍
    :param delta_rotation: 参数介绍
    :param thresholds: 参数介绍
    :return: 返回介绍
    """
    motion_lost = (
        delta_translation >= thresholds.motion_translation_lost
        or delta_rotation >= thresholds.motion_rotation_lost
    )
    motion_warn = (
        delta_translation >= thresholds.motion_translation_warn
        or delta_rotation >= thresholds.motion_rotation_warn
    )

    if inlier_ratio >= thresholds.fitness_ok and rmse <= thresholds.rmse_ok and not motion_warn:
        return TrackingStatus.OK
    if not motion_lost and inlier_ratio >= thresholds.fitness_warn and rmse <= thresholds.rmse_warn:
        return TrackingStatus.WARN
    return TrackingStatus.LOST
