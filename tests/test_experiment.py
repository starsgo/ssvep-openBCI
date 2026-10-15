import csv
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from analyse.test_four import PredictionResult
from experiment import (
    generate_target_indexes,
    load_targets,
    unique_target_frequencies,
)
from gaze import GazeTracker, gaze_region
from gaze.api import ApiGazeTracker
from main import TimerThread
from ui.display_psy import build_visual_objects, partition_visual_objects


class TargetConfigurationTests(unittest.TestCase):
    def test_default_config_loads_ordered_dynamic_targets(self):
        targets = load_targets(Path(__file__).parents[1] / "object.json")
        self.assertEqual(
            [target.frequency_hz for target in targets],
            [15.0, 12.0, 10.0, 15.0, 12.0, 10.0],
        )
        self.assertEqual(unique_target_frequencies(targets), [15.0, 12.0, 10.0])
        self.assertEqual(targets[0].center, (150.0, 300.0))
        self.assertEqual(
            [target.screen for target in targets],
            ["left", "left", "left", "right", "right", "right"],
        )

    def test_generated_indexes_follow_target_count(self):
        indexes = generate_target_indexes(5, count=40)
        self.assertEqual(len(indexes), 40)
        self.assertTrue(all(0 <= index < 5 for index in indexes))

    def test_config_requires_targets_in_both_halves(self):
        document = {
            "targets": [
                {
                    "name": "only-left",
                    "screen": "left",
                    "x": 10,
                    "y": 10,
                    "width": 100,
                    "height": 100,
                    "frequency_hz": 15,
                }
            ]
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "object.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "both screen halves"):
                load_targets(path)

    def test_frequency_must_be_unique_within_each_side(self):
        document = {
            "targets": [
                {
                    "name": "left-a",
                    "screen": "left",
                    "x": 10,
                    "y": 10,
                    "width": 100,
                    "height": 100,
                    "frequency_hz": 15,
                },
                {
                    "name": "left-b",
                    "screen": "left",
                    "x": 200,
                    "y": 10,
                    "width": 100,
                    "height": 100,
                    "frequency_hz": 15,
                },
                {
                    "name": "right-a",
                    "screen": "right",
                    "x": 1100,
                    "y": 10,
                    "width": 100,
                    "height": 100,
                    "frequency_hz": 15,
                },
            ]
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "object.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unique within each screen side"):
                load_targets(path)


class GazeTrackerTests(unittest.TestCase):
    def test_region_uses_screen_midpoint(self):
        self.assertEqual(gaze_region((-100, 500), 1920), "left")
        self.assertEqual(gaze_region((959, 500), 1920), "left")
        self.assertEqual(gaze_region((960, 500), 1920), "right")

    def test_tracker_accepts_negative_x_from_left_monitor(self):
        tracker = GazeTracker(lambda: (-800, 400), poll_interval=0.001)
        tracker.start()
        self.assertTrue(tracker.ready_event.wait(0.25))
        tracker.stop()
        tracker.join(timeout=0.25)
        self.assertEqual(tracker.get_current_position(), (-800, 400))
        self.assertEqual(tracker.get_current_region(), "left")

    def test_tracker_exposes_latest_valid_position(self):
        positions = iter(((100, 200), (1500, 300)))
        latest = [(1500, 300)]

        def provider():
            try:
                latest[0] = next(positions)
            except StopIteration:
                pass
            return latest[0]

        tracker = GazeTracker(provider, poll_interval=0.001)
        tracker.start()
        deadline = time.monotonic() + 0.25
        while tracker.get_current_position() != (1500, 300) and time.monotonic() < deadline:
            time.sleep(0.002)
        tracker.stop()
        tracker.join(timeout=0.25)
        self.assertEqual(tracker.get_current_position(), (1500, 300))
        self.assertEqual(tracker.get_current_region(), "right")


class ExperimentFlowTests(unittest.TestCase):
    class Signal:
        def __init__(self):
            self.values = []

        def emit(self, *values):
            self.values.append(values)

    def make_timer(self, prediction_results_file=None):
        targets = load_targets(Path(__file__).parents[1] / "object.json")
        signals = {
            name: self.Signal()
            for name in (
                "set_static_targets",
                "delete_visual_point",
                "set_status_text",
                "set_prompt_text",
                "add_visual_point",
                "start_trial",
            )
        }
        adder = SimpleNamespace(**signals)
        timer = TimerThread(
            adder,
            SimpleNamespace(status="b"),
            SimpleNamespace(
                get_current_position=lambda: (100, 100),
                region_for=lambda position: gaze_region(position, 1920),
            ),
            targets,
            build_visual_objects(targets),
            [0, 1, 2],
            prediction_results_file=prediction_results_file,
        )
        return timer, signals

    def test_region_partition_keeps_other_half_static(self):
        timer, _ = self.make_timer()
        active, inactive = partition_visual_objects(
            timer.targets, timer.visual_objects, "right"
        )
        self.assertEqual(
            [obj.name for obj in active],
            ["visualObj[3]", "visualObj[4]", "visualObj[5]"],
        )
        self.assertEqual(
            [obj.name for obj in inactive],
            ["visualObj[0]", "visualObj[1]", "visualObj[2]"],
        )

    def test_offscreen_api_gaze_still_starts_flicker_on_corresponding_side(self):
        for point, side in (((-.1, 1.2), 'left'), ((1.1, -.2), 'right')):
            with self.subTest(point=point):
                timer, signals = self.make_timer()
                tracker = ApiGazeTracker()
                self.assertTrue(tracker.update_from_api({
                    'schema': 'slippage.gaze.v1',
                    'coordinate_space': 'normalized_top_left',
                    'valid': True, 'u': point[0], 'v': point[1], 'age_ms': 0.,
                }))
                timer.gaze_tracker = tracker
                timer.pause_duration = timer.cue_duration = timer.gaze_duration = 0
                timer.blink_duration = 0

                def submit_trial(*values):
                    signals['start_trial'].values.append(values)
                    values[4].set()
                    timer.stop()

                signals['start_trial'].emit = submit_trial
                timer.start()
                timer.join(timeout=.5)
                timer.stop()
                timer.join(timeout=.5)
                self.assertFalse(timer.is_alive())
                self.assertEqual(len(signals['start_trial'].values), 1)
                active = signals['start_trial'].values[0][0]
                expected, _ = partition_visual_objects(timer.targets, timer.visual_objects, side)
                self.assertEqual([obj.name for obj in active], [obj.name for obj in expected])
                self.assertEqual(timer.results.metadata_for(1).gaze_side, side)

    def test_prediction_only_updates_matching_rest_phase(self):
        timer, signals = self.make_timer()
        prediction = PredictionResult(1, 375, np.ones(3), 1, 12.0)

        timer._register_trial_metadata(1, 1, (100, 100), "left")
        timer._begin_rest(1)
        timer.on_prediction(prediction)
        timer._end_rest()
        timer.on_prediction(PredictionResult(2, 375, np.ones(3), 0, 15.0))

        displayed = [values[0] for values in signals["set_status_text"].values]
        self.assertEqual(
            displayed,
            ["predit:processing...", "predit:12Hz 左边", None],
        )

    def test_prediction_frequency_and_gaze_side_resolve_true_object_index(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = Path(temp_dir) / "prediction_results.csv"
            timer, _ = self.make_timer(result_path)
            timer._register_trial_metadata(1, 3, (1500, 300), "right")
            timer.on_prediction(PredictionResult(1, 375, np.ones(3), 1, 12.0))
            timer._register_trial_metadata(2, 0, (100, 300), "left")
            timer.on_prediction(PredictionResult(2, 375, np.ones(3), 0, 15.0))

            with result_path.open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            accuracy = json.loads(
                (Path(temp_dir) / "prediction_accuracy.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["actual_index"], "3")
        self.assertEqual(rows[0]["actual_side"], "right")
        self.assertEqual(rows[0]["actual_x"], "1250.0")
        self.assertEqual(rows[0]["actual_width"], "200.0")
        self.assertEqual(rows[0]["predicted_target_index"], "4")
        self.assertEqual(rows[0]["predicted_target_name"], "visualObj[4]")
        self.assertEqual(rows[0]["gaze_side"], "right")
        self.assertEqual(rows[0]["correct"], "False")
        self.assertEqual(rows[1]["predicted_target_index"], "0")
        self.assertEqual(rows[1]["correct"], "True")
        self.assertEqual(accuracy["total_predictions"], 2)
        self.assertEqual(accuracy["correct_predictions"], 1)
        self.assertEqual(accuracy["accuracy_percent"], 50.0)

    def test_index_zero_reaches_visual_obj_zero_and_submits_a_trial(self):
        targets = load_targets(Path(__file__).parents[1] / "object.json")
        signals = {
            name: self.Signal()
            for name in (
                "set_static_targets",
                "delete_visual_point",
                "set_status_text",
                "set_prompt_text",
                "add_visual_point",
                "start_trial",
            )
        }
        ready = threading.Event()
        ready.set()
        timer_holder = []

        def submit_trial(*values):
            signals["start_trial"].values.append(values)
            values[4].set()
            timer_holder[0].stop()

        signals["start_trial"].emit = submit_trial
        timer = TimerThread(
            SimpleNamespace(**signals),
            SimpleNamespace(status="b", send_event=lambda event: None),
            SimpleNamespace(
                ready_event=ready,
                get_current_position=lambda: (100, 100),
                region_for=lambda position: gaze_region(position, 1920),
            ),
            targets,
            build_visual_objects(targets),
            [0],
            blink_duration=0,
            pause_duration=0,
            cue_duration=0,
            gaze_duration=0,
        )
        timer_holder.append(timer)
        timer.start()
        timer.join(timeout=0.5)

        self.assertFalse(timer.is_alive())
        self.assertEqual(signals["add_visual_point"].values[0], targets[0].center)
        self.assertEqual(
            [values[0] for values in signals["set_prompt_text"].values],
            [None, "注视目标0", None],
        )
        active_objects = signals["start_trial"].values[0][0]
        self.assertEqual(
            [obj.name for obj in active_objects],
            ["visualObj[0]", "visualObj[1]", "visualObj[2]"],
        )
        metadata = timer.results.metadata_for(1)
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.actual_index, 0)
        self.assertEqual(metadata.gaze_side, "left")


if __name__ == "__main__":
    unittest.main()
