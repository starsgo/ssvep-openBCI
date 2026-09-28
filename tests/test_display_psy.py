import math
import sys
import types
import unittest
from collections import defaultdict
from threading import Event
from unittest.mock import patch

from ui.display_psy import (
    ObjAdder,
    VisualObj_psy,
    _Trial,
    display_psy,
    brightness_at_phase,
    phase_at,
    top_left_to_center,
    trial_should_end,
    wrapped_phase_error,
)


class PhaseMathTests(unittest.TestCase):
    def test_initial_phase_is_zero_and_maximum_brightness(self):
        self.assertEqual(phase_at(0.0, 15.0), 0.0)
        self.assertEqual(brightness_at_phase(0.0), 1.0)

    def test_cosine_brightness_range_and_half_cycle(self):
        self.assertAlmostEqual(brightness_at_phase(math.pi), 0.0)
        samples = [brightness_at_phase(i * math.pi / 16.0) for i in range(33)]
        self.assertTrue(all(0.0 <= value <= 1.0 for value in samples))

    def test_phase_tracks_elapsed_time_after_a_dropped_frame(self):
        frequency = 12.0
        elapsed = 7.0 / 60.0
        self.assertAlmostEqual(
            phase_at(elapsed, frequency),
            (2.0 * math.pi * frequency * elapsed) % (2.0 * math.pi),
        )

    def test_wrapped_phase_error_uses_shortest_distance(self):
        self.assertAlmostEqual(wrapped_phase_error(0.05, 2 * math.pi - 0.05), 0.1)

    def test_trial_ends_on_first_refresh_at_duration(self):
        onset = 100.0
        self.assertFalse(trial_should_end(onset + 1.499, onset, 1.5))
        self.assertTrue(trial_should_end(onset + 1.5, onset, 1.5))


class CompatibilityTests(unittest.TestCase):
    def test_escape_leaves_fullscreen_without_closing_renderer(self):
        event_module = types.ModuleType("event")
        event_module.getKeys = lambda **_kwargs: ["escape"]
        psychopy_module = types.ModuleType("psychopy")
        psychopy_module.event = event_module

        renderer = display_psy.__new__(display_psy)
        renderer._fullscr = True
        renderer.win = type(
            "FakeWindow",
            (),
            {
                "fullscr": True,
                "winHandle": type("FakeHandle", (), {"has_exit": False})(),
            },
        )()

        with patch.dict(sys.modules, {"psychopy": psychopy_module}):
            wants_close = renderer._window_wants_close()

        self.assertFalse(wants_close)
        self.assertFalse(renderer.win.fullscr)
        self.assertFalse(renderer._fullscr)

    def test_mouse_position_is_published_in_top_left_coordinates(self):
        received = []

        class FakeMouse:
            @staticmethod
            def getPos():
                return (-860.0, 440.0)

        renderer = display_psy.__new__(display_psy)
        renderer.win = type("FakeWindow", (), {"size": (1920, 1080)})()
        renderer._mouse = FakeMouse()
        renderer._gaze_callback = received.append

        renderer._publish_mouse_position()

        self.assertEqual(received, [(100, 100)])

    def test_legacy_top_left_coordinates_are_preserved(self):
        self.assertEqual(
            top_left_to_center((50, 50), (200, 200), (1800, 900)),
            (-750.0, 300.0),
        )

    def test_visual_object_keeps_legacy_constructor_shape(self):
        obj = VisualObj_psy((0, 0), (200, 200), 15.0, name="target")
        self.assertEqual(obj.position, (0, 0))
        self.assertEqual(obj.size, (200, 200))
        self.assertEqual(obj.blink_frequency, 15.0)

    def test_queued_signal_is_not_executed_in_producer(self):
        adder = ObjAdder()
        received = []
        adder.start_trial.connect(lambda objects, duration: received.append((objects, duration)))
        objects = [object(), object(), object()]
        adder.start_trial.emit(objects, 1.5)
        self.assertEqual(received, [])
        adder.dispatcher.drain()
        self.assertEqual(received, [(objects, 1.5)])

    def test_cue_uses_nonzero_circle_line_width(self):
        captured = {}

        class FakeVisual:
            @staticmethod
            def Circle(win, **kwargs):
                captured.update(kwargs)
                return object()

        renderer = display_psy.__new__(display_psy)
        renderer.win = type("FakeWindow", (), {"size": (1920, 1080)})()
        renderer._visual = FakeVisual()
        renderer._cue_diameter = 50.0
        renderer.add_visual_point(150, 150)

        self.assertEqual(captured["lineWidth"], 1)
        self.assertEqual(captured["lineColor"], "red")
        self.assertEqual(captured["fillColor"], "red")

    def test_prompt_text_is_centered(self):
        captured = {}

        class FakeVisual:
            @staticmethod
            def TextStim(win, **kwargs):
                captured.update(kwargs)
                return object()

        renderer = display_psy.__new__(display_psy)
        renderer.win = object()
        renderer._visual = FakeVisual()
        renderer.set_prompt_text("注视目标0")

        self.assertEqual(captured["text"], "注视目标0")
        self.assertEqual(captured["pos"], (0, 0))

    def test_non_selected_targets_are_drawn_black(self):
        class FakeStim:
            fillColor = None

            def draw(self):
                pass

        inactive = VisualObj_psy((0, 0), (10, 10), 12.0, name="inactive")
        inactive._stim = FakeStim()

        display_psy._draw_black([inactive])

        self.assertEqual(inactive._stim.fillColor, -1.0)

    def test_trial_targets_share_onset_and_callbacks_run_on_flip(self):
        class FakeStim:
            fillColor = None

            def draw(self):
                pass

        class FakeWindow:
            def __init__(self):
                self.flip_times = iter((1.0, 1.0167))
                self.callbacks = []
                self.recordFrameIntervals = True

            def callOnFlip(self, callback, *args):
                self.callbacks.append((callback, args))

            def flip(self):
                callbacks, self.callbacks = self.callbacks, []
                for callback, args in callbacks:
                    callback(*args)
                return next(self.flip_times)

        events = []
        done = Event()
        objects = [
            VisualObj_psy((0, 0), (10, 10), frequency, name=f"target-{frequency}")
            for frequency in (15.0, 12.0, 10.0)
        ]
        for obj in objects:
            obj._stim = FakeStim()

        renderer = display_psy.__new__(display_psy)
        renderer.win = FakeWindow()
        renderer._cue_stim = None
        renderer._callback_errors = []
        renderer._last_flip_time = 0.9833
        renderer._drop_threshold = 0.02
        renderer._records = defaultdict(list)
        renderer._trial_summaries = []
        renderer.visual_objs = list(objects)
        trial = _Trial(
            trial_id=1,
            objects=objects,
            duration_s=1.5,
            on_start=lambda: events.append("blink"),
            on_stop=lambda: events.append("pause"),
            done_event=done,
        )

        renderer._draw_first_trial_frame(trial)
        self.assertEqual(events, ["blink"])
        self.assertEqual({record.flip_time_s for _, record in trial.records}, {1.0})
        renderer._last_flip_time = 1.0
        renderer._finish_trial(trial)
        self.assertEqual(events, ["blink", "pause"])
        self.assertTrue(done.is_set())


if __name__ == "__main__":
    unittest.main()
