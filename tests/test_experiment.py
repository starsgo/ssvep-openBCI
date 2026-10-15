import csv
import json
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from analyse.test_four import BlinkSegmentProcessor, PredictionResult
from experiment import (
    TargetSpec,
    generate_target_indexes,
    load_targets,
    unique_target_frequencies,
)
from gaze import GazePointProvider, GazeTracker, gaze_region
from gaze.api import ApiGazeTracker
from main import TimerThread
from ui.display_psy import build_visual_objects, partition_visual_objects


def legacy_targets():
    """Keep regression cases for the supported original two-region layout."""
    positions = ((50, 200), (700, 200), (600, 650), (1250, 200), (1700, 200), (1400, 650))
    return [TargetSpec(f'visualObj[{index}]', 'left' if index < 3 else 'right',
                       float(x), float(y), 200., 200., (15., 12., 10.)[index % 3])
            for index, (x, y) in enumerate(positions)]


class TargetConfigurationTests(unittest.TestCase):
    def test_default_config_loads_ordered_dynamic_targets(self):
        targets = load_targets(Path(__file__).parents[1] / "object.json")
        self.assertEqual(
            [target.frequency_hz for target in targets],
            [15.0, 12.0, 10.0] * 3,
        )
        self.assertEqual(unique_target_frequencies(targets), [15.0, 12.0, 10.0])
        self.assertEqual(targets[0].center, (427.0, 240.0))
        self.assertEqual(
            [target.screen for target in targets],
            ["left"] * 3 + ["right"] * 3 + ["bottom"] * 3,
        )

    def test_shape_color_labels_and_triangle_frequencies(self):
        targets = load_targets(Path(__file__).parents[1] / 'object.json')
        objects = build_visual_objects(targets)
        self.assertEqual([target.label for target in targets], list(map(str, range(9))))
        triangles = [index for index, target in enumerate(targets) if target.shape == 'triangle']
        self.assertEqual(triangles, [2, 4, 6])
        self.assertEqual([targets[index].frequency_hz for index in triangles], [10., 12., 15.])
        for index, (target, obj) in enumerate(zip(targets, objects)):
            self.assertEqual(target.color, '#8B0000' if index in triangles else '#000000')
            self.assertEqual((obj.shape, obj.color, obj.label), (target.shape, target.color, target.label))
            self.assertEqual(gaze_region(target.center, 1920, 1080), target.screen)
            for corner in ((target.x, target.y), (target.x+target.width, target.y),
                           (target.x, target.y+target.height),
                           (target.x+target.width, target.y+target.height)):
                self.assertEqual(gaze_region(corner, 1920, 1080), target.screen)
            for other in targets[index+1:]:
                overlap = (min(target.x+target.width, other.x+other.width) > max(target.x, other.x)
                           and min(target.y+target.height, other.y+other.height) > max(target.y, other.y))
                self.assertFalse(overlap, (target.name, other.name))

    def test_legacy_configuration_without_style_fields_still_loads(self):
        document = {'targets': [dict(name=t.name, screen=t.screen, x=t.x, y=t.y,
                                    width=t.width, height=t.height, frequency_hz=t.frequency_hz)
                                for t in legacy_targets()]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'legacy.json'
            path.write_text(json.dumps(document), encoding='utf-8')
            loaded = load_targets(path)
        self.assertTrue(all(t.shape == 'rectangle' and t.color == '#000000' and t.label is None
                            for t in loaded))

    def test_invalid_shape_or_color_is_rejected(self):
        document = json.loads((Path(__file__).parents[1] / 'object.json').read_text(encoding='utf-8'))
        for key, value in (('shape', 'unknown'), ('color', '#XX0000')):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                copy = json.loads(json.dumps(document))
                copy['targets'][0][key] = value
                path = Path(directory) / 'invalid.json'
                path.write_text(json.dumps(copy), encoding='utf-8')
                with self.assertRaises(ValueError):
                    load_targets(path)

    def test_generated_indexes_follow_target_count(self):
        indexes = generate_target_indexes(5, count=40)
        self.assertEqual(len(indexes), 40)
        self.assertTrue(all(0 <= index < 5 for index in indexes))

    def test_config_can_keep_a_single_nonempty_region(self):
        document = {
            "region_layout": "three_sectors", "targets": [
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
            targets = load_targets(path)
            self.assertEqual(len(targets), 1)
            self.assertEqual(targets[0].region_layout, 'three_sectors')

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
    def test_three_regions_match_center_rays_and_boundary_ties(self):
        for point, expected in (((427, 240), 'left'), ((1493, 240), 'right'),
                                ((960, 900), 'bottom'), ((0, 0), 'left'),
                                ((960, 0), 'right'), ((960, 540), 'bottom'),
                                ((480, 809), 'left'), ((480, 810), 'bottom'),
                                ((1440, 809), 'right'), ((1440, 810), 'bottom')):
            with self.subTest(point=point):
                self.assertEqual(gaze_region(point, 1920, 1080), expected)
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

    def make_timer(self, prediction_results_file=None, *, targets=None):
        targets = legacy_targets() if targets is None else targets
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

    def test_three_region_gaze_and_nogaze_select_and_classify_bottom_targets(self):
        targets = load_targets(Path(__file__).parents[1] / 'object.json')
        for nogaze, point in ((False, (960, 900)), (True, (100, 100))):
            with self.subTest(nogaze=nogaze), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'results.csv'
                timer, signals = self.make_timer(path, targets=targets)
                timer.nogaze = nogaze
                timer.target_indexes = [6]
                timer.pause_duration = timer.cue_duration = timer.gaze_duration = 0
                timer.gaze_tracker = GazePointProvider(Mock(get_current_position=Mock(return_value=point)))
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
                self.assertEqual([obj.name for obj in signals['start_trial'].values[0][0]],
                                 ['visualObj[6]', 'visualObj[7]', 'visualObj[8]'])
                self.assertEqual(timer.results.metadata_for(1).flicker_side, 'bottom')
                timer.on_prediction(PredictionResult(1, 375, np.ones(3), 0, 15.))
                with path.open(encoding='utf-8') as stream:
                    row = next(csv.DictReader(stream))
                self.assertEqual(row['predicted_target_index'], '6')
                self.assertEqual(row['correct'], 'True')
                self.assertEqual(row['gaze_side'], 'left' if nogaze else 'bottom')

    def test_removed_middle_target_keeps_labels_and_region_candidate_set(self):
        document = json.loads((Path(__file__).parents[1] / 'object.json').read_text(encoding='utf-8'))
        document['targets'] = [target for target in document['targets'] if target['label'] != 4]
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'object.json'
            config.write_text(json.dumps(document), encoding='utf-8')
            targets = load_targets(config)
            results = Path(directory) / 'results.csv'
            timer, signals = self.make_timer(results, targets=targets)
            timer.target_indexes = [4]  # This entry now has display label 5.
            timer.nogaze = True
            timer.pause_duration = timer.cue_duration = timer.gaze_duration = 0
            timer.gaze_tracker = GazePointProvider(Mock(get_current_position=Mock(return_value=(1493, 240))))
            processor = BlinkSegmentProcessor(Path(directory)/'recording.txt',
                output_dir=Path(directory)/'classification',
                target_frequencies=unique_target_frequencies(targets),
                candidate_frequencies=timer.classification_frequencies,
                on_prediction=timer.on_prediction)
            def submit_trial(*values):
                signals['start_trial'].values.append(values)
                timepoints = np.arange(375) / 250.
                processor._samples = list(np.array([
                    np.sin(2*np.pi*10.*timepoints+phase)
                    for phase in np.linspace(0., 1., 8)
                ]).T)
                processor._process_current_segment(1)
                values[4].set()
                timer.stop()
            signals['start_trial'].emit = submit_trial
            timer.start()
            timer.join(timeout=.5)
            timer.stop()
            timer.join(timeout=.5)
            self.assertFalse(timer.is_alive())
            self.assertIn(('注视目标5',), signals['set_prompt_text'].values)
            self.assertEqual(timer.classification_frequencies(1), (15., 10.))
            timer._register_trial_metadata(2, 0, (100, 100), 'left', flicker_side='left')
            self.assertEqual(timer.classification_frequencies(2), (15., 12., 10.))
            self.assertEqual(timer.classification_frequencies(1), (15., 10.))
            self.assertEqual(timer._predictions[1].prediction_frequency, 10.)
            self.assertEqual(timer._predictions[1].target_frequencies, (15., 10.))
            with results.open(encoding='utf-8') as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row['actual_index'], '5')
            self.assertEqual(row['predicted_target_index'], '5')
            self.assertEqual(row['actual_config_index'], '4')
            self.assertEqual(row['predicted_config_index'], '4')
            self.assertEqual(row['correct'], 'True')

    def test_empty_region_is_skipped_without_advancing_eeg_segment_index(self):
        document = json.loads((Path(__file__).parents[1] / 'object.json').read_text(encoding='utf-8'))
        document['targets'] = [target for target in document['targets'] if target['screen'] != 'bottom']
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'object.json'
            config.write_text(json.dumps(document), encoding='utf-8')
            targets = load_targets(config)
        timer, signals = self.make_timer(targets=targets)
        self.assertTrue(timer.three_regions)
        timer.target_indexes = [0, 1]
        timer.pause_duration = timer.cue_duration = timer.gaze_duration = 0
        source = Mock(get_current_position=Mock(side_effect=[(960, 900), (100, 100)]))
        timer.gaze_tracker = GazePointProvider(source)
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
        self.assertIn(('该区域无目标，跳过闪烁',), signals['set_status_text'].values)
        self.assertEqual(timer.results.metadata_for(1).actual_index, 1)
        self.assertIsNone(timer.results.metadata_for(2))

    def test_gaze_phase_keeps_latest_point_when_deadline_frame_is_missing(self):
        timer, _ = self.make_timer()
        timer.gaze_duration = .02
        timer.gaze_tracker.get_current_position = Mock(side_effect=[(10, 20), (1500, 300), None])
        clock = [0.]
        def advance(seconds):
            clock[0] += seconds
            return True
        with patch('main.time.monotonic', side_effect=lambda: clock[0]), \
             patch.object(timer, '_wait', side_effect=advance):
            self.assertEqual(timer._collect_gaze(), (1500, 300))
            self.assertAlmostEqual(clock[0], .02)
            timer.gaze_tracker.get_current_position = Mock(return_value=None)
            self.assertIsNone(timer._collect_gaze())

    def test_never_acquired_gaze_flickers_left_without_waiting(self):
        timer, signals = self.make_timer()
        timer.pause_duration = timer.cue_duration = timer.gaze_duration = 0
        timer.blink_duration = 0
        timer.gaze_tracker = GazePointProvider()
        timer.target_indexes = [3]
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
        self.assertEqual(len(signals['add_visual_point'].values), 1)
        self.assertEqual(len(signals['start_trial'].values), 1)
        metadata = timer.results.metadata_for(1)
        self.assertEqual(metadata.gaze_side, 'left')
        self.assertEqual(metadata.gaze_position, (0, 0))
        self.assertEqual(metadata.gaze_point_source, 'default')
        self.assertNotIn(('等待眼动坐标',), signals['set_status_text'].values)

    def test_retained_history_from_before_gaze_phase_decides_flicker(self):
        timer, signals = self.make_timer()
        source = Mock()
        source.get_current_position.return_value = (1919, 540)
        point = GazePointProvider(source)
        self.assertEqual(point.get_current_position(), (1919, 540))
        source.get_current_position.return_value = None
        timer.gaze_tracker = point
        timer.pause_duration = timer.cue_duration = 0
        timer.gaze_duration = .03
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
        metadata = timer.results.metadata_for(1)
        self.assertEqual(metadata.gaze_position, (1919, 540))
        self.assertEqual(metadata.flicker_side, 'right')
        self.assertEqual(metadata.gaze_point_source, 'history')

    def test_nogaze_flickers_target_side_while_recording_opposite_gaze(self):
        for target_index, measured, expected_side in ((3, (100, 100), 'right'),
                                                       (0, (1500, 300), 'left'),
                                                       (3, None, 'right')):
            with self.subTest(target_index=target_index), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'results.csv'
                timer, signals = self.make_timer(path)
                timer.nogaze = True
                timer.target_indexes = [target_index]
                timer.gaze_tracker = GazePointProvider(Mock(get_current_position=Mock(return_value=measured)))
                timer.pause_duration = timer.cue_duration = timer.gaze_duration = 0
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
                expected, _ = partition_visual_objects(timer.targets, timer.visual_objects, expected_side)
                self.assertEqual([obj.name for obj in active], [obj.name for obj in expected])
                metadata = timer.results.metadata_for(1)
                self.assertEqual(metadata.gaze_position, (0, 0) if measured is None else measured)
                self.assertNotEqual(metadata.gaze_side, expected_side)
                self.assertEqual(metadata.flicker_side, expected_side)
                self.assertEqual(metadata.region_source, 'target')
                timer.on_prediction(PredictionResult(1, 375, np.ones(3), 0, 15.0))
                with path.open(encoding='utf-8') as stream:
                    row = next(csv.DictReader(stream))
                self.assertEqual(row['predicted_target_index'], str(target_index))
                self.assertEqual(row['correct'], 'True')
                self.assertEqual(row['gaze_side_correct'], 'False')
                self.assertEqual(row['flicker_side'], expected_side)

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
        targets = legacy_targets()
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
