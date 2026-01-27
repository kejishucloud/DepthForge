from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from scanner.utils.matrix import pose_delta
from scanner.core.types import Intrinsics, Keyframe


@dataclass
class KeyframePolicy:
    min_translation: float
    min_rotation_deg: float
    min_quality: float
    min_interval: int
    max_keyframes: int = 0
    mode_overrides: Dict[str, Dict[str, float]] = field(default_factory=dict)


class KeyframeSelector:
    def __init__(self, policy: KeyframePolicy, mode: str = "realtime") -> None:
        self._policy = policy
        self._mode = mode
        self._last_key_index: Optional[int] = None
        self._last_key_pose: Optional[np.ndarray] = None

    def should_add(self, index: int, pose: np.ndarray, quality: float) -> bool:
        """
        判断当前帧是否满足关键帧采样策略。
        :param index: 当前帧序号
        :param pose: 当前帧的全局位姿（4x4）
        :param quality: 当前帧质量评分（如里程计 fitness）
        :return: 是否应采样为关键帧
        """
        policy = self._effective_policy()
        if quality < policy.min_quality:
            return False
        if self._last_key_index is None or self._last_key_pose is None:
            return True
        if index - self._last_key_index < policy.min_interval:
            return False
        delta = pose_delta(self._last_key_pose, pose)
        if delta.translation >= policy.min_translation:
            return True
        if delta.rotation_deg >= policy.min_rotation_deg:
            return True
        return False

    def update(self, index: int, pose: np.ndarray, keyframes: Optional[list[Keyframe]] = None) -> None:
        """
        更新关键帧缓存。
        :param index: 最新关键帧序号
        :param pose: 最新关键帧位姿
        :param keyframes: 可选关键帧列表，用于执行窗口裁剪
        :return: None
        """
        self._last_key_index = index
        self._last_key_pose = pose
        self._enforce_window(keyframes)

    def set_mode(self, mode: str) -> None:
        """
        设置扫描模式，用于应用模式覆盖的关键帧策略。
        :param mode: 模式名称（realtime/semi/offline）
        :return: None
        """
        self._mode = mode

    def intrinsics_to_dict(self, intr: Intrinsics) -> Dict[str, Any]:
        """
        将相机内参对象转换为可序列化的字典。
        :param intr: 相机内参
        :return: 内参字典
        """
        return {
            "width": intr.width,
            "height": intr.height,
            "fx": intr.fx,
            "fy": intr.fy,
            "cx": intr.cx,
            "cy": intr.cy,
            "depth_scale": intr.depth_scale,
        }

    def _effective_policy(self) -> KeyframePolicy:
        """
        根据当前模式生成实际生效的关键帧策略。
        :return: 合并模式覆盖后的策略
        """
        overrides = self._policy.mode_overrides.get(self._mode, {})
        return KeyframePolicy(
            min_translation=float(overrides.get("min_translation", self._policy.min_translation)),
            min_rotation_deg=float(overrides.get("min_rotation_deg", self._policy.min_rotation_deg)),
            min_quality=float(overrides.get("min_quality", self._policy.min_quality)),
            min_interval=int(overrides.get("min_interval", self._policy.min_interval)),
            max_keyframes=self._policy.max_keyframes,
            mode_overrides=self._policy.mode_overrides,
        )

    def _enforce_window(self, keyframes: Optional[list[Keyframe]]) -> None:
        """
        限制关键帧数量，超出时丢弃最早的关键帧。
        :param keyframes: 关键帧列表
        :return: None
        """
        if keyframes is None:
            return
        max_keyframes = int(self._policy.max_keyframes or 0)
        if max_keyframes <= 0:
            return
        while len(keyframes) > max_keyframes:
            keyframes.pop(0)
