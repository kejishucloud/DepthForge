from __future__ import annotations

import time


class FPSCounter:
    def __init__(self, window: int = 30) -> None:
        self._window = window
        self._times: list[float] = []

    def tick(self) -> float:
        """
        更新窗口并计算平均 FPS。
        :return: 返回介绍
        """
        now = time.time()
        self._times.append(now)
        if len(self._times) > self._window:
            self._times.pop(0)
        if len(self._times) < 2:
            return 0.0
        return (len(self._times) - 1) / (self._times[-1] - self._times[0] + 1e-6)
