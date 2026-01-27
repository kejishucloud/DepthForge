from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from scanner.core.types import Keyframe


@dataclass
class CandidateConfig:
    temporal_gap: int
    max_candidates: int
    max_distance: float
    use_all_frames: bool = False


def generate_candidates(keyframes: List[Keyframe], config: CandidateConfig) -> List[Tuple[int, int]]:
    """
    根据时间间隔与空间距离生成回环候选对。
    :param keyframes: 关键帧列表（包含位姿）
    :param config: 候选生成配置
    :return: 候选对索引列表 (source_index, target_index)
    """
    pairs: List[Tuple[int, int]] = []
    if len(keyframes) < 2:
        return pairs

    if config.use_all_frames:
        for j in range(1, len(keyframes)):
            target = keyframes[j]
            candidates: List[Tuple[int, float]] = []
            for i in range(0, j):
                source = keyframes[i]
                if target.index - source.index < config.temporal_gap:
                    continue
                dist = float(np.linalg.norm(target.pose[:3, 3] - source.pose[:3, 3]))
                if dist <= config.max_distance:
                    candidates.append((i, dist))
            candidates.sort(key=lambda x: x[1])
            for i, _ in candidates[: config.max_candidates]:
                pairs.append((i, j))
    else:
        last_index = len(keyframes) - 1
        last_kf = keyframes[last_index]
        candidates = []
        for i, kf in enumerate(keyframes[:-1]):
            if last_kf.index - kf.index < config.temporal_gap:
                continue
            dist = float(np.linalg.norm(last_kf.pose[:3, 3] - kf.pose[:3, 3]))
            if dist <= config.max_distance:
                candidates.append((i, dist))
        candidates.sort(key=lambda x: x[1])
        for i, _ in candidates[: config.max_candidates]:
            pairs.append((i, last_index))

    return pairs
