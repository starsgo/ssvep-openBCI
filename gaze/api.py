"""Read measured eye coordinates from the Slippage HTTP API."""
from __future__ import annotations

import http.client
import json
import math
import time
from urllib.parse import urlsplit

from gaze import GazeTracker


class ApiGazeTracker(GazeTracker):
    """Poll the latest sample; never substitute mouse or expired coordinates."""

    def __init__(self, url="http://127.0.0.1:8765/gaze", *, window_size=(1920, 1080),
                 poll_interval=.01, max_age_sec=.2, request_timeout=.1):
        super().__init__(window_size=window_size, poll_interval=poll_interval)
        parsed = urlsplit(url)
        if parsed.scheme != "http" or not parsed.hostname:
            raise ValueError("gaze API URL must be an http URL")
        if max_age_sec <= 0 or request_timeout <= 0 or min(self.window_size) <= 0:
            raise ValueError("gaze API timing and window size must be positive")
        self.host, self.port = parsed.hostname, parsed.port or 80
        self.path = parsed.path or "/gaze"
        if parsed.query:
            self.path += "?" + parsed.query
        self.max_age_sec = float(max_age_sec)
        self.request_timeout = float(request_timeout)
        self._expires_at = 0.
        self._sample_received_at = None
        self._sample_age_at_receipt = None
        self.frame_id = None
        self.api_connected = False

    def _invalidate(self, reason):
        with self._lock:
            self._position = None
            self._last_error = reason
            self._expires_at = 0.

    def update_from_api(self, payload, *, received_at=None, request_elapsed=0.):
        received_at = time.monotonic() if received_at is None else float(received_at)
        try:
            if payload.get("schema") != "slippage.gaze.v1":
                raise ValueError("unsupported gaze API schema")
            if payload.get("coordinate_space") != "normalized_top_left":
                raise ValueError("unsupported gaze coordinate space")
            self.api_connected = True
            if payload.get("valid") is not True:
                raise ValueError(str(payload.get("reason", "invalid gaze sample")))
            u, v, age_ms = (float(payload[key]) for key in ("u", "v", "age_ms"))
            age = age_ms / 1000. + max(0., float(request_elapsed))
            if not all(math.isfinite(value) for value in (u, v, age)):
                raise ValueError("nonfinite gaze sample")
            if not 0. <= age < self.max_age_sec:
                raise ValueError("stale gaze sample")
            width, height = self.window_size
            # Keep finite off-screen estimates usable for gaze display and flicker selection.
            position = (min(width - 1, round(min(1., max(0., u)) * width)),
                        min(height - 1, round(min(1., max(0., v)) * height)))
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            self._invalidate(str(error))
            return False
        with self._lock:
            self._position = position
            self._last_error = None
            self._expires_at = received_at + self.max_age_sec - age
            self._sample_received_at = received_at
            self._sample_age_at_receipt = age
            self.frame_id = payload.get("frame_id")
        self.ready_event.set()
        return True

    def get_current_position(self):
        with self._lock:
            if time.monotonic() >= self._expires_at:
                return None
            return self._position

    @property
    def sample_age_ms(self):
        with self._lock:
            if self._sample_received_at is None:
                return None
            return (self._sample_age_at_receipt + time.monotonic() - self._sample_received_at) * 1000.

    def run(self):
        connection = None
        try:
            while not self._stop_event.is_set():
                try:
                    if connection is None:
                        connection = http.client.HTTPConnection(self.host, self.port,
                                                                timeout=self.request_timeout)
                    started_at = time.monotonic()
                    connection.request("GET", self.path)
                    response = connection.getresponse()
                    body = response.read(16385)
                    if response.status != 200 or len(body) > 16384:
                        raise ValueError(f"gaze API HTTP {response.status} or oversized response")
                    now = time.monotonic()
                    self.update_from_api(json.loads(body), received_at=now,
                                         request_elapsed=now - started_at)
                except (OSError, ValueError, http.client.HTTPException) as error:
                    self.api_connected = False
                    self._invalidate(f"{type(error).__name__}: {error}")
                    if connection is not None:
                        connection.close()
                    connection = None
                self._stop_event.wait(self.poll_interval)
        finally:
            if connection is not None:
                connection.close()
