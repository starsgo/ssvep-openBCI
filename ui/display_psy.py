"""Frame-locked PsychoPy display backend for SSVEP stimulation.

The public surface mirrors :mod:`ui.display`, while rendering is driven by
PsychoPy ``Window.flip()`` calls on the main thread.  Worker threads communicate
with the renderer through :class:`ObjAdder`'s queued signals.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import queue
import re
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from gaze import psychopy_to_top_left


TAU = 2.0 * math.pi
CSV_FIELDS = (
    "trial_id",
    "frame_index",
    "target_hz",
    "flip_time_s",
    "elapsed_s",
    "frame_interval_s",
    "predicted_phase_rad",
    "actual_phase_rad",
    "phase_error_rad",
    "applied_brightness",
    "dropped_frame",
    "trial_valid",
)


def phase_at(elapsed_s: float, frequency_hz: float) -> float:
    """Return the continuous cosine phase in the range ``[0, 2*pi)``."""
    if frequency_hz <= 0:
        return 0.0
    return (TAU * frequency_hz * max(0.0, elapsed_s)) % TAU


def brightness_at_phase(phase_rad: float) -> float:
    """Map cosine phase to a black-to-white brightness value."""
    return 0.5 * (1.0 + math.cos(phase_rad))


def format_prediction_text(frequency_hz: float, side: str | None = None) -> str:
    side_text = {"left": "左边", "right": "右边"}.get(side, "")
    suffix = f" {side_text}" if side_text else ""
    return f"predit:{frequency_hz:g}Hz{suffix}"


def wrapped_phase_error(predicted: float, actual: float) -> float:
    """Return ``predicted - actual`` wrapped to ``[-pi, pi)``."""
    return (predicted - actual + math.pi) % TAU - math.pi


def trial_should_end(
    predicted_flip_time: float, onset_time: float, duration_s: float
) -> bool:
    """Return whether the next refresh is the first one at/after trial end."""
    return predicted_flip_time - onset_time >= duration_s


def top_left_to_center(
    position: Sequence[float], size: Sequence[float], window_size: Sequence[float]
) -> tuple[float, float]:
    """Convert legacy top-left pixel coordinates to PsychoPy center pixels."""
    x, y = position
    width, height = size
    window_width, window_height = window_size
    return (
        float(x) + float(width) / 2.0 - float(window_width) / 2.0,
        float(window_height) / 2.0 - (float(y) + float(height) / 2.0),
    )


class _Dispatcher:
    def __init__(self) -> None:
        self._queue: queue.Queue[tuple[tuple[Callable[..., Any], ...], tuple, dict]] = (
            queue.Queue()
        )

    def enqueue(
        self,
        callbacks: tuple[Callable[..., Any], ...],
        args: tuple,
        kwargs: dict,
    ) -> None:
        self._queue.put((callbacks, args, kwargs))

    def drain(self) -> None:
        while True:
            try:
                callbacks, args, kwargs = self._queue.get_nowait()
            except queue.Empty:
                return
            for callback in callbacks:
                callback(*args, **kwargs)


class QueuedSignal:
    """Small thread-safe replacement for the ``connect``/``emit`` Qt surface."""

    def __init__(self, dispatcher: _Dispatcher) -> None:
        self._dispatcher = dispatcher
        self._callbacks: list[Callable[..., Any]] = []
        self._lock = threading.Lock()

    def connect(self, callback: Callable[..., Any]) -> None:
        with self._lock:
            self._callbacks.append(callback)

    def emit(self, *args: Any, **kwargs: Any) -> None:
        with self._lock:
            callbacks = tuple(self._callbacks)
        self._dispatcher.enqueue(callbacks, args, kwargs)


class ObjAdder:
    """Queued signals used by experiment worker threads.

    The first four signals retain the names and behavior used by the Qt
    implementation. ``start_trial`` atomically submits all targets for a
    frame-synchronised trial.
    """

    def __init__(self) -> None:
        self.dispatcher = _Dispatcher()
        self.add_obj = QueuedSignal(self.dispatcher)
        self.remove_all_obj = QueuedSignal(self.dispatcher)
        self.add_visual_point = QueuedSignal(self.dispatcher)
        self.delete_visual_point = QueuedSignal(self.dispatcher)
        self.set_static_targets = QueuedSignal(self.dispatcher)
        self.set_status_text = QueuedSignal(self.dispatcher)
        self.set_prompt_text = QueuedSignal(self.dispatcher)
        self.start_trial = QueuedSignal(self.dispatcher)
        self.close_requested = QueuedSignal(self.dispatcher)


@dataclass
class VisualObj_psy:
    position: tuple[float, float]
    size: tuple[float, float]
    blink_frequency: float = 0.0
    image: Any = None
    name: str = "visualObj"
    log_blink: bool = False
    _stim: Any = field(default=None, init=False, repr=False)

    def phase(self, elapsed_s: float) -> float:
        return phase_at(elapsed_s, self.blink_frequency)

    def brightness(self, elapsed_s: float) -> float:
        return brightness_at_phase(self.phase(elapsed_s))

    def attach(self, win: Any) -> None:
        """Create the PsychoPy stimulus lazily in the owning render thread."""
        if self._stim is not None:
            return
        from psychopy import visual

        pos = top_left_to_center(self.position, self.size, win.size)
        if self.image is None:
            self._stim = visual.Rect(
                win,
                width=self.size[0],
                height=self.size[1],
                pos=pos,
                units="pix",
                lineWidth=0,
                fillColor=1.0,
                colorSpace="rgb",
                autoLog=False,
            )
        else:
            self._stim = visual.ImageStim(
                win,
                image=self.image,
                pos=pos,
                size=self.size,
                units="pix",
                opacity=1.0,
                autoLog=False,
            )

    def draw(self, brightness: float) -> None:
        if self._stim is None:
            raise RuntimeError("VisualObj_psy must be attached before drawing")
        brightness = min(1.0, max(0.0, float(brightness)))
        if self.image is None:
            self._stim.fillColor = 2.0 * brightness - 1.0
        else:
            self._stim.opacity = brightness
        self._stim.draw()


@dataclass
class FrameRecord:
    trial_id: int
    frame_index: int
    target_hz: float
    flip_time_s: float
    elapsed_s: float
    frame_interval_s: float | None
    predicted_phase_rad: float
    actual_phase_rad: float
    phase_error_rad: float
    applied_brightness: float
    dropped_frame: bool
    trial_valid: bool = True


@dataclass
class _Trial:
    trial_id: int
    objects: list[VisualObj_psy]
    duration_s: float
    on_start: Callable[[], Any] | None
    on_stop: Callable[[], Any] | None
    done_event: threading.Event | None
    static_objects: list[VisualObj_psy] = field(default_factory=list)
    onset_time: float | None = None
    offset_time: float | None = None
    frame_index: int = 0
    dropped_frames: int = 0
    records: list[tuple[str, FrameRecord]] = field(default_factory=list)
    aborted: bool = False
    callback_errors_at_start: int = 0
    primed: bool = False


class display_psy:
    """PsychoPy SSVEP display with flip-time phase diagnostics."""

    def __init__(
        self,
        visual_objs: Iterable[VisualObj_psy] | None = None,
        parent: Any = None,
        background: Any = 0.0,
        *,
        fullscr: bool = False,
        size: tuple[int, int] = (1920, 1080),
        screen: int = 0,
        cue_diameter: float = 50.0,
        warmup_frames: int = 120,
        record_root: str | Path = "record",
        run_id: str | None = None,
    ) -> None:
        del parent
        from psychopy import event, visual

        self._visual = visual
        self._background = background
        self._fullscr = fullscr
        self._requested_size = tuple(size)
        self._title = "PsychoPy SSVEP display"
        self._cue_diameter = float(cue_diameter)
        self._warmup_frames = int(warmup_frames)
        self._close_requested = False
        self._closed = False
        self._saved = False
        self._cue_position: tuple[float, float] | None = None
        self._cue_stim: Any = None
        self._status_text: str | None = None
        self._status_stim: Any = None
        self._prompt_text: str | None = None
        self._prompt_stim: Any = None
        self._static_brightness = 1.0
        self.visual_objs: list[VisualObj_psy] = list(visual_objs or [])
        self._pending_trials: deque[_Trial] = deque()
        self._active_trial: _Trial | None = None
        self._next_trial_id = 1
        self._last_flip_time: float | None = None
        self._frame_period: float | None = None
        self._refresh_hz: float | None = None
        self._drop_threshold: float | None = None
        self._records: dict[str, list[FrameRecord]] = defaultdict(list)
        self._trial_summaries: list[dict[str, Any]] = []
        self._callback_errors: list[str] = []
        self.ready_event = threading.Event()
        self._gaze_callback: Callable[[tuple[int, int]], Any] | None = None

        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.record_dir = Path(record_root) / self.run_id
        self.win = visual.Window(
            size=self._requested_size,
            fullscr=fullscr,
            screen=screen,
            pos=(0, 0) if not fullscr else None,
            units="pix",
            color=background,
            colorSpace="rgb",
            # A borderless screen-sized window can still be switched with Alt+Tab.
            allowGUI=False,
            waitBlanking=True,
            checkTiming=True,
            title=self._title,
            autoLog=False,
        )
        # The mouse is the gaze-point substitute in this experiment. PsychoPy
        # hides it by default for stimulus windows, so explicitly show it.
        try:
            self.win.mouseVisible = True
        except AttributeError:
            pass
        self._mouse = event.Mouse(win=self.win, visible=True)

    def setWindowTitle(self, title: str) -> None:  # Qt-compatible spelling
        self._title = str(title)

    def resize(self, width: int, height: int) -> None:
        """Retain the Qt-compatible method for the requested window size."""
        self._requested_size = (int(width), int(height))

    def show(self) -> None:
        """Compatibility no-op: PsychoPy realizes the window during construction."""

    def update(self) -> bool:
        """Compatibility no-op; the next render-loop flip applies current state."""
        return False

    def add_visual_obj(self, obj: VisualObj_psy) -> None:
        if not isinstance(obj, VisualObj_psy):
            raise TypeError("display_psy accepts VisualObj_psy instances")
        obj.attach(self.win)
        self.visual_objs.append(obj)

    def add_visual_objs(self, objects: Iterable[VisualObj_psy]) -> None:
        for obj in objects:
            self.add_visual_obj(obj)

    def clear_visual_objs(self) -> None:
        self.visual_objs.clear()

    def set_static_targets(self, objects: Iterable[VisualObj_psy]) -> None:
        """Show all configured targets as black rectangles."""
        targets = list(objects)
        for obj in targets:
            if not isinstance(obj, VisualObj_psy):
                raise TypeError("all static targets must be VisualObj_psy instances")
            obj.attach(self.win)
        self.visual_objs[:] = targets
        self._static_brightness = 0.0

    def set_status_text(self, text: str | None) -> None:
        """Set the small status message shown at the bottom of the screen."""
        normalized = None if text is None else str(text).strip()
        self._status_text = normalized or None
        if self._status_text is None:
            self._status_stim = None
            return
        _, window_height = self.win.size
        self._status_stim = self._visual.TextStim(
            self.win,
            text=self._status_text,
            pos=(0, -float(window_height) / 2.0 + 28.0),
            height=18.0,
            units="pix",
            color="white",
            autoLog=False,
        )

    def set_prompt_text(self, text: str | None) -> None:
        """Set the large instruction shown at the centre of the screen."""
        normalized = None if text is None else str(text).strip()
        self._prompt_text = normalized or None
        self._prompt_stim = (
            None
            if self._prompt_text is None
            else self._visual.TextStim(
                self.win,
                text=self._prompt_text,
                pos=(0, 0),
                height=36.0,
                units="pix",
                color="white",
                autoLog=False,
            )
        )

    def add_visual_point(self, x: float, y: float) -> None:
        window_width, window_height = self.win.size
        self._cue_position = (
            float(x) - float(window_width) / 2.0,
            float(window_height) / 2.0 - float(y),
        )
        self._cue_stim = self._visual.Circle(
            self.win,
            radius=self._cue_diameter / 2.0,
            pos=self._cue_position,
            units="pix",
            # PsychoPy 2026.2.3 Circle divides by line width while choosing
            # polygon edges, so zero raises ZeroDivisionError.
            lineWidth=1,
            lineColor="red",
            fillColor="red",
            autoLog=False,
        )

    def delete_visual_point(self) -> None:
        self._cue_position = None
        self._cue_stim = None

    def start_visual_trial(
        self,
        objects: Iterable[VisualObj_psy],
        duration_s: float = 1.5,
        on_start: Callable[[], Any] | None = None,
        on_stop: Callable[[], Any] | None = None,
        done_event: threading.Event | None = None,
        static_objects: Iterable[VisualObj_psy] | None = None,
    ) -> None:
        objects = list(objects)
        static_objects = list(static_objects or [])
        if not objects:
            raise ValueError("a visual trial requires at least one object")
        if duration_s <= 0:
            raise ValueError("duration_s must be positive")
        for obj in objects:
            if not isinstance(obj, VisualObj_psy):
                raise TypeError("all trial objects must be VisualObj_psy instances")
            obj.attach(self.win)
        for obj in static_objects:
            if not isinstance(obj, VisualObj_psy):
                raise TypeError("all static trial objects must be VisualObj_psy instances")
            obj.attach(self.win)
        trial = _Trial(
            trial_id=self._next_trial_id,
            objects=objects,
            static_objects=static_objects,
            duration_s=float(duration_s),
            on_start=on_start,
            on_stop=on_stop,
            done_event=done_event,
            callback_errors_at_start=len(self._callback_errors),
        )
        self._next_trial_id += 1
        self._pending_trials.append(trial)

    def request_close(self) -> None:
        self._close_requested = True

    def set_gaze_callback(
        self, callback: Callable[[tuple[int, int]], Any] | None
    ) -> None:
        self._gaze_callback = callback

    def _publish_mouse_position(self) -> None:
        if self._gaze_callback is None:
            return
        self._gaze_callback(psychopy_to_top_left(self._mouse.getPos(), self.win.size))

    def _draw_cue(self) -> None:
        if self._cue_stim is not None:
            self._cue_stim.draw()

    def _draw_status(self) -> None:
        status_stim = getattr(self, "_status_stim", None)
        if status_stim is not None:
            status_stim.draw()

    def _draw_prompt(self) -> None:
        prompt_stim = getattr(self, "_prompt_stim", None)
        if prompt_stim is not None:
            prompt_stim.draw()

    @staticmethod
    def _draw_black(objects: Iterable[VisualObj_psy]) -> None:
        for obj in objects:
            obj.draw(0.0)

    def _safe_callback(self, callback: Callable[[], Any] | None) -> None:
        if callback is None:
            return
        try:
            callback()
        except Exception as exc:  # keep the render loop alive and invalidate the run
            self._callback_errors.append(f"{type(exc).__name__}: {exc}")

    def _warm_up(self) -> None:
        flip_times: list[float] = []
        self.win.recordFrameIntervals = False
        for _ in range(max(2, self._warmup_frames)):
            self._draw_legacy_objects(0.0)
            flip_time = self.win.flip()
            if flip_time is not None:
                flip_times.append(float(flip_time))
        intervals = [b - a for a, b in zip(flip_times, flip_times[1:]) if b > a]
        if not intervals:
            raise RuntimeError("PsychoPy did not return VSync flip timestamps")
        ordered = sorted(intervals)
        self._frame_period = ordered[len(ordered) // 2]
        if self._frame_period <= 0:
            raise RuntimeError("invalid monitor frame period")
        self._refresh_hz = 1.0 / self._frame_period
        self._drop_threshold = self._frame_period * 1.2
        self._last_flip_time = flip_times[-1]
        self.win.refreshThreshold = self._drop_threshold
        self.win.frameIntervals = []

    def _window_wants_close(self) -> bool:
        from psychopy import event

        if event.getKeys(keyList=["escape"]):
            # Escape only leaves fullscreen. Acquisition and the command input
            # remain alive until the operator explicitly sends `s`/`exit`.
            if bool(getattr(self.win, "fullscr", self._fullscr)):
                self.win.fullscr = False
            self._fullscr = False
        handle = getattr(self.win, "winHandle", None)
        return bool(getattr(handle, "has_exit", False))

    def _activate_next_trial(self) -> None:
        if self._active_trial is None and self._pending_trials:
            self._active_trial = self._pending_trials.popleft()
            self.visual_objs[:] = (
                self._active_trial.static_objects + self._active_trial.objects
            )
            self._static_brightness = 0.0

    def _prime_trial(self, trial: _Trial) -> float:
        """Upload/draw resources invisibly before the time-critical onset flip."""
        self._draw_black(trial.static_objects)
        self._draw_black(trial.objects)
        self._draw_cue()
        self._draw_status()
        self._draw_prompt()
        flip_time = float(self.win.flip())
        trial.primed = True
        self.win.frameIntervals = []
        self.win.recordFrameIntervals = True
        return flip_time

    def _append_frame_records(
        self,
        trial: _Trial,
        flip_time: float,
        predicted_elapsed: float,
        predicted_phases: list[float],
        brightnesses: list[float],
        *,
        first_frame: bool,
    ) -> None:
        if first_frame:
            trial.onset_time = flip_time
            elapsed = 0.0
        else:
            if trial.onset_time is None:
                raise RuntimeError("trial onset was not captured")
            elapsed = max(0.0, flip_time - trial.onset_time)
        frame_interval = (
            None if self._last_flip_time is None else flip_time - self._last_flip_time
        )
        dropped = bool(
            frame_interval is not None
            and self._drop_threshold is not None
            and frame_interval > self._drop_threshold
        )
        if dropped:
            trial.dropped_frames += 1

        for obj, predicted_phase, brightness in zip(
            trial.objects, predicted_phases, brightnesses
        ):
            actual_phase = 0.0 if first_frame else obj.phase(elapsed)
            record = FrameRecord(
                trial_id=trial.trial_id,
                frame_index=trial.frame_index,
                target_hz=float(obj.blink_frequency),
                flip_time_s=flip_time,
                elapsed_s=elapsed,
                frame_interval_s=frame_interval,
                predicted_phase_rad=predicted_phase,
                actual_phase_rad=actual_phase,
                phase_error_rad=wrapped_phase_error(predicted_phase, actual_phase),
                applied_brightness=brightness,
                dropped_frame=dropped,
            )
            trial.records.append((obj.name, record))
        trial.frame_index += 1

    def _draw_first_trial_frame(self, trial: _Trial) -> float:
        phases = [0.0 for _ in trial.objects]
        brightnesses = [1.0 for _ in trial.objects]
        self._draw_black(trial.static_objects)
        for obj in trial.objects:
            obj.draw(1.0)
        self._draw_cue()
        self._draw_status()
        self._draw_prompt()
        self.win.callOnFlip(self._safe_callback, trial.on_start)
        flip_time = float(self.win.flip())
        self._append_frame_records(
            trial,
            flip_time,
            predicted_elapsed=0.0,
            predicted_phases=phases,
            brightnesses=brightnesses,
            first_frame=True,
        )
        return flip_time

    def _draw_trial_frame(self, trial: _Trial, predicted_flip: float) -> float:
        if trial.onset_time is None:
            return self._draw_first_trial_frame(trial)
        predicted_elapsed = max(0.0, predicted_flip - trial.onset_time)
        phases = [obj.phase(predicted_elapsed) for obj in trial.objects]
        brightnesses = [brightness_at_phase(value) for value in phases]
        self._draw_black(trial.static_objects)
        for obj, brightness in zip(trial.objects, brightnesses):
            obj.draw(brightness)
        self._draw_cue()
        self._draw_status()
        self._draw_prompt()
        flip_time = float(self.win.flip())
        self._append_frame_records(
            trial,
            flip_time,
            predicted_elapsed,
            phases,
            brightnesses,
            first_frame=False,
        )
        return flip_time

    def _finish_trial(self, trial: _Trial, *, aborted: bool = False) -> float:
        trial.aborted = aborted
        self._draw_black(trial.static_objects)
        self._draw_black(trial.objects)
        self._draw_cue()
        self._draw_status()
        self._draw_prompt()
        self.win.callOnFlip(self._safe_callback, trial.on_stop)
        flip_time = float(self.win.flip())
        trial.offset_time = flip_time
        offset_interval = (
            None if self._last_flip_time is None else flip_time - self._last_flip_time
        )
        if (
            offset_interval is not None
            and self._drop_threshold is not None
            and offset_interval > self._drop_threshold
        ):
            trial.dropped_frames += 1
        new_callback_errors = (
            len(self._callback_errors) > trial.callback_errors_at_start
        )
        valid = not aborted and trial.dropped_frames == 0 and not new_callback_errors
        for name, record in trial.records:
            record.trial_valid = valid
            self._records[name].append(record)
        actual_duration = (
            None
            if trial.onset_time is None
            else max(0.0, trial.offset_time - trial.onset_time)
        )
        self._trial_summaries.append(
            {
                "trial_id": trial.trial_id,
                "target_hz": [float(obj.blink_frequency) for obj in trial.objects],
                "static_target_hz": [
                    float(obj.blink_frequency) for obj in trial.static_objects
                ],
                "onset_time_s": trial.onset_time,
                "offset_time_s": trial.offset_time,
                "requested_duration_s": trial.duration_s,
                "actual_duration_s": actual_duration,
                "frames_presented": trial.frame_index,
                "dropped_frames": trial.dropped_frames,
                "valid": valid,
                "aborted": aborted,
            }
        )
        self.visual_objs[:] = trial.static_objects + trial.objects
        self._static_brightness = 0.0
        if trial.done_event is not None:
            trial.done_event.set()
        self._active_trial = None
        self.win.recordFrameIntervals = False
        return flip_time

    def _draw_legacy_objects(self, predicted_flip: float) -> None:
        del predicted_flip
        for obj in self.visual_objs:
            obj.draw(self._static_brightness)
        self._draw_cue()
        self._draw_status()
        self._draw_prompt()

    def run(self, adder: ObjAdder, stop_event: threading.Event | None = None) -> int:
        """Run the frame loop on the main thread until close is requested."""
        self._warm_up()
        self._publish_mouse_position()
        self.ready_event.set()
        try:
            while not self._close_requested:
                if stop_event is not None and stop_event.is_set():
                    break
                adder.dispatcher.drain()
                self._publish_mouse_position()
                if self._window_wants_close():
                    break
                self._activate_next_trial()
                predicted_flip = float(self.win.getFutureFlipTime(clock=None))
                trial = self._active_trial
                if trial is None:
                    self._draw_legacy_objects(predicted_flip)
                    flip_time = float(self.win.flip())
                elif not trial.primed:
                    flip_time = self._prime_trial(trial)
                elif trial.onset_time is None:
                    flip_time = self._draw_first_trial_frame(trial)
                elif trial_should_end(
                    predicted_flip, trial.onset_time, trial.duration_s
                ):
                    flip_time = self._finish_trial(trial)
                else:
                    flip_time = self._draw_trial_frame(trial, predicted_flip)
                self._last_flip_time = flip_time
        finally:
            if self._active_trial is not None:
                self._finish_trial(self._active_trial, aborted=True)
            while self._pending_trials:
                pending = self._pending_trials.popleft()
                pending.aborted = True
                if pending.done_event is not None:
                    pending.done_event.set()
            self._close_requested = True
        return 0

    @staticmethod
    def _safe_filename(name: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
        return cleaned or "visual_object"

    def _save_csv_files(self) -> None:
        for name, records in self._records.items():
            path = self.record_dir / f"{self._safe_filename(name)}_phase.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
                writer.writeheader()
                for record in records:
                    writer.writerow(asdict(record))

    def _save_summary(self) -> None:
        all_records = [record for records in self._records.values() for record in records]
        phase_errors = [abs(record.phase_error_rad) for record in all_records]
        summary = {
            "run_id": self.run_id,
            "phase_definition": "actual phase is derived from PsychoPy flip timestamps",
            "physical_timing_note": "use a photodiode to validate display light output",
            "measured_refresh_hz": self._refresh_hz,
            "median_frame_period_s": self._frame_period,
            "drop_threshold_s": self._drop_threshold,
            "total_dropped_frames": sum(
                item["dropped_frames"] for item in self._trial_summaries
            ),
            "mean_abs_phase_error_rad": (
                sum(phase_errors) / len(phase_errors) if phase_errors else None
            ),
            "max_abs_phase_error_rad": max(phase_errors, default=None),
            "callback_errors": self._callback_errors,
            "trials": self._trial_summaries,
        }
        path = self.record_dir / "timing_summary.json"
        path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _save_plot(self) -> None:
        if not self._records:
            return
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = sorted(self._records)
        fig, axes = plt.subplots(
            len(names) + 1,
            2,
            figsize=(14, 3.2 * (len(names) + 1)),
            squeeze=False,
        )
        for row, name in enumerate(names):
            records = self._records[name]
            trials: dict[int, list[FrameRecord]] = defaultdict(list)
            for record in records:
                trials[record.trial_id].append(record)

            valid_label_added = False
            invalid_label_added = False
            for trial_records in trials.values():
                elapsed = [record.elapsed_s for record in trial_records]
                brightnesses = [record.applied_brightness for record in trial_records]
                brightness_errors = [
                    record.applied_brightness
                    - brightness_at_phase(record.actual_phase_rad)
                    for record in trial_records
                ]
                valid = all(record.trial_valid for record in trial_records)
                if valid:
                    color = "tab:blue"
                    alpha = 0.18
                    label = "valid trials" if not valid_label_added else None
                    valid_label_added = True
                else:
                    color = "tab:red"
                    alpha = 0.65
                    label = "dropped/aborted" if not invalid_label_added else None
                    invalid_label_added = True
                axes[row][0].plot(
                    elapsed,
                    brightnesses,
                    color=color,
                    alpha=alpha,
                    linewidth=0.7,
                    marker=".",
                    markersize=1.5,
                    label=label,
                )
                axes[row][1].plot(
                    elapsed,
                    brightness_errors,
                    color=color,
                    alpha=alpha,
                    linewidth=0.7,
                    label=label,
                )

            axes[row][0].set_title(f"{name}: frame-sampled brightness")
            axes[row][0].set_ylabel("brightness (0-1)")
            axes[row][0].set_xlim(left=0.0)
            axes[row][0].set_ylim(-0.05, 1.05)
            axes[row][0].legend(loc="upper right")
            axes[row][1].axhline(0.0, color="black", linewidth=0.5)
            axes[row][1].set_title(f"{name}: commanded - ideal brightness")
            axes[row][1].set_ylabel("brightness error")
            axes[row][1].set_xlim(left=0.0)
            axes[row][1].set_ylim(-1.05, 1.05)
            axes[row][1].legend(loc="upper right")

        representative = next(iter(self._records.values()))
        intervals_ms = [
            record.frame_interval_s * 1000.0
            for record in representative
            if record.frame_interval_s is not None
        ]
        axes[-1][0].plot(intervals_ms, linewidth=0.8)
        if self._drop_threshold is not None:
            axes[-1][0].axhline(
                self._drop_threshold * 1000.0,
                color="red",
                linestyle="--",
                label="drop threshold",
            )
            axes[-1][0].legend()
        axes[-1][0].set_title("Frame intervals")
        axes[-1][0].set_ylabel("milliseconds")
        axes[-1][0].set_xlabel("recorded frame")
        axes[-1][1].axis("off")
        fig.tight_layout()
        fig.savefig(self.record_dir / "waveform_diagnostics.png", dpi=160)
        plt.close(fig)

    def _save_3d_waveform_plot(self) -> None:
        """Save terrain with trial, elapsed time, and brightness axes."""
        if not self._records:
            return
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        names = sorted(self._records)
        fig = plt.figure(figsize=(6.4 * len(names), 6.2), constrained_layout=True)
        axes = []
        surface = None
        for column, name in enumerate(names, start=1):
            axis = fig.add_subplot(1, len(names), column, projection="3d")
            axes.append(axis)
            trials: dict[int, list[FrameRecord]] = defaultdict(list)
            for record in self._records[name]:
                trials[record.trial_id].append(record)

            trial_ids = np.asarray(sorted(trials), dtype=float)
            common_end = min(
                max(record.elapsed_s for record in trial_records)
                for trial_records in trials.values()
            )
            time_axis = np.linspace(0.0, common_end, 181)
            brightness_grid = np.asarray(
                [
                    np.interp(
                        time_axis,
                        [record.elapsed_s for record in trials[int(trial_id)]],
                        [
                            record.applied_brightness
                            for record in trials[int(trial_id)]
                        ],
                    )
                    for trial_id in trial_ids
                ]
            )
            trial_grid, time_grid = np.meshgrid(
                trial_ids, time_axis, indexing="ij"
            )
            surface = axis.plot_surface(
                trial_grid,
                time_grid,
                brightness_grid,
                cmap="turbo",
                vmin=0.0,
                vmax=1.0,
                rstride=1,
                cstride=3,
                linewidth=0.12,
                antialiased=True,
                alpha=0.95,
            )

            axis.set_title(f"{name}: waveform terrain")
            axis.set_xlabel("trial")
            axis.set_ylabel("elapsed (s)")
            axis.set_zlabel("brightness (0-1)")
            axis.set_xlim(float(trial_ids.min()), float(trial_ids.max()))
            axis.set_ylim(0.0, common_end)
            axis.set_zlim(-0.05, 1.05)
            axis.view_init(elev=28, azim=-58)
            axis.set_box_aspect((2.8, 1.5, 1.0))

        if surface is not None:
            colorbar = fig.colorbar(surface, ax=axes, shrink=0.62, pad=0.03)
            colorbar.set_label("brightness (0-1)")
        fig.savefig(self.record_dir / "waveform_3d_trials.png", dpi=180)
        plt.close(fig)

    def save_diagnostics(self) -> Path:
        if self._saved:
            return self.record_dir
        self.record_dir.mkdir(parents=True, exist_ok=True)
        self._save_csv_files()
        self._save_summary()
        self._save_plot()
        self._save_3d_waveform_plot()
        self._saved = True
        return self.record_dir

    def close(self) -> Path:
        if not self._closed:
            self.win.recordFrameIntervals = False
            self.win.close()
            self._closed = True
        return self.save_diagnostics()


display = display_psy
VisualObj = VisualObj_psy
visualObj = VisualObj_psy


def build_visual_objects(targets: Iterable[Any]) -> list[VisualObj_psy]:
    return [
        VisualObj_psy(
            (target.x, target.y),
            (target.width, target.height),
            target.frequency_hz,
            name=target.name,
            log_blink=True,
        )
        for target in targets
    ]


def partition_visual_objects(
    targets: Iterable[Any],
    visual_objects: Iterable[VisualObj_psy],
    side: str,
) -> tuple[list[VisualObj_psy], list[VisualObj_psy]]:
    """Split rendered targets into the flashing and static screen sides."""
    active: list[VisualObj_psy] = []
    inactive: list[VisualObj_psy] = []
    for target, visual_object in zip(targets, visual_objects, strict=True):
        (active if target.screen == side else inactive).append(visual_object)
    return active, inactive


def run_realtime_test(duration_s: float = 1.5) -> Path:
    """Run one full-screen 15/12/10 Hz hardware timing trial."""
    adder = ObjAdder()
    window = display_psy()
    adder.add_obj.connect(window.add_visual_obj)
    adder.remove_all_obj.connect(lambda: window.visual_objs.clear() or window.update())
    adder.add_visual_point.connect(window.add_visual_point)
    adder.delete_visual_point.connect(window.delete_visual_point)
    adder.start_trial.connect(window.start_visual_trial)
    adder.close_requested.connect(window.request_close)
    done = threading.Event()

    objects = [
        VisualObj_psy((50, 50), (200, 200), 15.0, name="visualObj[0]", log_blink=True),
        VisualObj_psy((800, 50), (200, 200), 12.0, name="visualObj[1]", log_blink=True),
        VisualObj_psy((425, 550), (200, 200), 10.0, name="visualObj[2]", log_blink=True),
    ]

    def submit() -> None:
        adder.add_visual_point.emit(150, 150)
        adder.start_trial.emit(objects, duration_s, None, None, done)
        done.wait(timeout=duration_s + 10.0)
        adder.close_requested.emit()

    threading.Thread(target=submit, daemon=True).start()
    try:
        window.run(adder)
    finally:
        output = window.close()
    return output


def _main() -> int:
    parser = argparse.ArgumentParser(description="Run a PsychoPy SSVEP timing test")
    parser.add_argument("--duration", type=float, default=1.5)
    args = parser.parse_args()
    output = run_realtime_test(args.duration)
    print(f"Timing records saved to: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
