from __future__ import annotations

import queue
from typing import Generic, Optional, TypeVar

T = TypeVar("T")


class FrameQueue(Generic[T]):
    def __init__(self, maxsize: int) -> None:
        self._queue: queue.Queue[T] = queue.Queue(maxsize=maxsize)

    def put(self, item: T) -> None:
        """
        写入队列，满时丢弃最旧元素。
        :param item: 待写入的元素
        :return: None
        """
        if self._queue.full():
            try:
                _ = self._queue.get_nowait()
            except queue.Empty:
                pass
        self._queue.put(item)

    def get(self, timeout: Optional[float] = None) -> Optional[T]:
        """
        读取队列元素，超时返回 None。
        :param timeout: 超时秒数，None 表示一直等待
        :return: 读取到的元素或 None
        """
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def clear(self) -> None:
        """
        清空队列。
        :return: None
        """
        while not self._queue.empty():
            try:
                _ = self._queue.get_nowait()
            except queue.Empty:
                break
