"""Includes common filters."""


from __future__ import annotations

from abc import ABC, abstractmethod
import statistics


__all__ = ['WindowFilter', 'MovingAverage', 'MedianFilter']


class WindowFilter(ABC):
    """An abstract base class for MovingAverage and Median filters.

    If you want to make your own WindowFilter, then it's pretty simple because
    you only need to override _calculate_from_window. Also, I've used type hints
    here because the editor I use (PyCharm) makes my life easier if I include
    them but you can just ignore them if you want, they're completely optional.
    """

    def __init__(self, window_size: int = None) -> None:
        self.window_size = window_size
        self.window = []

    @abstractmethod
    def _calculate_from_window(self) -> float:
        pass

    def calculate(self, value: float = None) -> float:
        if value is not None:
            self.window.append(value)
            if self.window_size is not None and len(self.window) > self.window_size:
                self.window = self.window[1:]
        return self._calculate_from_window()


class MedianFilter(WindowFilter):
    def _calculate_from_window(self) -> float:
        return statistics.median(self.window)


class MovingAverage(WindowFilter):
    def _calculate_from_window(self) -> float:
        return statistics.mean(self.window)
