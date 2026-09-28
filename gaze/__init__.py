"""Mouse-backed gaze sampling and coordinate helpers."""

from __future__ import annotations

import ctypes
import threading
from typing import Callable, Literal, Sequence


Region = Literal["left", "right"]


def gaze_region(position: tuple[int, int], window_width: int) -> Region:
    if window_width <= 0:
        raise ValueError("window_width must be positive")
    return "left" if position[0] < window_width / 2.0 else "right"


def psychopy_to_top_left(
    position: Sequence[float], window_size: Sequence[int]
) -> tuple[int, int]:
    """Convert PsychoPy center-origin pixels to top-left window coordinates."""
    x, y = position
    width, height = window_size
    return round(float(x) + width / 2.0), round(height / 2.0 - float(y))


class _Point(ctypes.Structure):
    _fields_ = (("x", ctypes.c_long), ("y", ctypes.c_long))


def windows_cursor_position() -> tuple[int, int]:
    point = _Point()
    if not ctypes.windll.user32.GetCursorPos(ctypes.byref(point)):
        raise OSError("GetCursorPos failed")
    return int(point.x), int(point.y)


class GazeTracker(threading.Thread):
    """Store the latest mouse position as the simulated gaze point."""

    def __init__(
        self,
        position_provider: Callable[[], tuple[int, int]] | None = None,
        *,
        poll_interval: float = 0.01,
        window_size: Sequence[int] = (1920, 1080),
    ) -> None:
        super().__init__(daemon=True)
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        self.position_provider = position_provider
        self.poll_interval = float(poll_interval)
        self.window_size = int(window_size[0]), int(window_size[1])
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._position: tuple[int, int] | None = None
        self._last_error: str | None = None
        self.ready_event = threading.Event()

    def run(self) -> None:
        while not self._stop_event.is_set():
            if self.position_provider is not None:
                try:
                    self.update_position(self.position_provider())
                except Exception as error:
                    with self._lock:
                        self._last_error = f"{type(error).__name__}: {error}"
            self._stop_event.wait(self.poll_interval)

    def update_position(self, position: Sequence[float]) -> None:
        normalized = int(position[0]), int(position[1])
        with self._lock:
            self._position = normalized
            self._last_error = None
        self.ready_event.set()

    def stop(self) -> None:
        self._stop_event.set()

    def get_current_position(self) -> tuple[int, int] | None:
        with self._lock:
            return self._position

    def get_current_region(self) -> Region | None:
        position = self.get_current_position()
        return None if position is None else self.region_for(position)

    def region_for(self, position: tuple[int, int]) -> Region:
        return gaze_region(position, self.window_size[0])

    @property
    def last_error(self) -> str | None:
        with self._lock:
            return self._last_error
