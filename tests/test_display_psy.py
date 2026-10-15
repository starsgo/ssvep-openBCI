import math
import sys
import types
import unittest
from collections import defaultdict
from threading import Event
from unittest.mock import Mock, patch

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
    equilateral_vertices,
    format_prediction_text,
)


class PhaseMathTests(unittest.TestCase):
    def test_prediction_text_accepts_bottom_region(self):
        self.assertEqual(format_prediction_text(15., 'bottom'), 'predit:15Hz 下方')
    def test_initial_phase_is_zero_and_black(self):
        self.assertEqual(phase_at(0.0, 15.0), 0.0)
        self.assertEqual(brightness_at_phase(0.0), 0.0)

    def test_cosine_brightness_range_and_half_cycle(self):
        self.assertAlmostEqual(brightness_at_phase(math.pi), 1.0)
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
    def test_triangle_is_equilateral_and_fits_non_square_bounding_boxes(self):
        for size in ((200, 200), (200, 80), (100, 300)):
            with self.subTest(size=size):
                vertices = equilateral_vertices(size)
                edges = [math.dist(vertices[index], vertices[(index+1) % 3]) for index in range(3)]
                self.assertAlmostEqual(edges[0], edges[1])
                self.assertAlmostEqual(edges[1], edges[2])
                self.assertTrue(all(abs(x) <= size[0]/2. and abs(y) <= size[1]/2.
                                    for x, y in vertices))

    def test_triangle_uses_polygon_and_white_number_at_target_center(self):
        visual = Mock()
        module = types.ModuleType('psychopy')
        module.visual = visual
        obj = VisualObj_psy((483, 410), (200, 200), 10., shape='triangle', color='#8B0000', label='2')
        win = types.SimpleNamespace(size=(1920, 1080))
        with patch.dict(sys.modules, {'psychopy': module}):
            obj.attach(win)
        visual.Rect.assert_not_called()
        self.assertEqual(visual.ShapeStim.call_args.kwargs['vertices'], equilateral_vertices((200, 200)))
        self.assertEqual(visual.ShapeStim.call_args.kwargs['pos'], (-377., 30.))
        self.assertEqual(visual.TextStim.call_args.kwargs['text'], '2')
        self.assertEqual(visual.TextStim.call_args.kwargs['pos'], (-377., 30.))
        self.assertEqual(visual.TextStim.call_args.kwargs['color'], 'white')

    def test_dark_red_triangle_interpolates_to_white_and_returns_to_dark_red(self):
        obj = VisualObj_psy((0, 0), (200, 200), 10., shape='triangle', color='#8B0000')
        obj._stim = Mock()
        red = (139. / 127.5 - 1., -1., -1.)
        for brightness in (0., .5, 1., 0.):
            obj.draw(brightness)
            expected = tuple(base + (1.-base)*brightness for base in red)
            for actual, wanted in zip(obj._stim.fillColor, expected):
                self.assertAlmostEqual(actual, wanted)

    def test_labels_appear_outside_flicker_and_hide_for_all_targets_during_flicker(self):
        objects = [VisualObj_psy((index*220, 0), (200, 200), 10., label=str(index)) for index in range(3)]
        for obj in objects:
            obj._stim, obj._label_stim = Mock(), Mock()
        renderer = display_psy.__new__(display_psy)
        renderer.win = Mock()
        renderer.win.flip.side_effect = (1., 1.0167)
        renderer.visual_objs = objects
        renderer._static_brightness = 0.
        renderer._draw_cue = renderer._draw_status = renderer._draw_prompt = Mock()
        renderer._append_frame_records = Mock()
        renderer._draw_legacy_objects(0.)
        self.assertTrue(all(obj._label_stim.draw.call_count == 1 for obj in objects))
        trial = _Trial(1, objects[:2], 1.5, None, None, None, static_objects=objects[2:])
        renderer._draw_first_trial_frame(trial)
        trial.onset_time = 1.
        renderer._draw_trial_frame(trial, 1.0167)
        self.assertTrue(all(obj._label_stim.draw.call_count == 1 for obj in objects))
        renderer._draw_legacy_objects(1.0334)
        self.assertTrue(all(obj._label_stim.draw.call_count == 2 for obj in objects))

    def test_three_guides_join_center_to_top_midpoint_and_lower_corners(self):
        renderer = display_psy.__new__(display_psy)
        renderer.win = types.SimpleNamespace(size=(1920, 1080))
        renderer._visual = Mock()
        renderer._visual.Line.side_effect = [Mock(), Mock(), Mock()]
        renderer.set_region_guides()
        calls = renderer._visual.Line.call_args_list
        self.assertEqual([call.kwargs['end'] for call in calls],
                         [(0, 540.), (-960., -540.), (960., -540.)])
        self.assertTrue(all(call.kwargs['start'] == (0, 0) and call.kwargs['lineWidth'] == 1
                            for call in calls))
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

    def test_green_target_uses_one_channel_and_starts_black(self):
        class FakeStim:
            fillColor = None

            def draw(self):
                pass

        target = VisualObj_psy(
            (0, 0), (10, 10), 15.0, name="green-target", color_mode="green"
        )
        target._stim = FakeStim()
        for brightness, expected in [
            (0.0, (-1.0, -1.0, -1.0)),
            (0.5, (-1.0, 0.0, -1.0)),
            (1.0, (-1.0, 1.0, -1.0)),
        ]:
            target.draw(brightness)
            self.assertEqual(target._stim.fillColor, expected)

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
        self.assertEqual({record.applied_brightness for _, record in trial.records}, {0.0})
        self.assertTrue(all(obj._stim.fillColor == -1.0 for obj in objects))
        renderer._last_flip_time = 1.0
        renderer._finish_trial(trial)
        self.assertEqual(events, ["blink", "pause"])
        self.assertTrue(done.is_set())


if __name__ == "__main__":
    unittest.main()
