import csv
import json
import io
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from analyse.fbcca import classify_segment
from analyse.test_four import (
    BlinkSegmentProcessor,
    SEGMENT_SAMPLES,
    parse_eeg_line,
    process_blink_segment,
)


class OnlineClassificationTests(unittest.TestCase):
    frequencies = [15.0, 12.0, 10.0]

    @staticmethod
    def synthetic_eeg(frequency, sample_count=SEGMENT_SAMPLES):
        timepoints = np.arange(sample_count) / 250.0
        return np.vstack(
            [
                np.sin(2 * np.pi * frequency * timepoints + phase)
                + 0.25 * np.sin(4 * np.pi * frequency * timepoints + phase / 2)
                for phase in np.linspace(0.0, 1.0, 8)
            ]
        )

    def test_fbcca_classifies_three_notebook_frequencies_from_375_samples(self):
        for frequency in self.frequencies:
            with self.subTest(frequency=frequency):
                result = classify_segment(
                    self.synthetic_eeg(frequency), self.frequencies, fs=250
                )
                self.assertEqual(result.scores.shape, (3,))
                self.assertEqual(result.prediction_frequency, frequency)

    def test_preprocessing_caps_at_375_and_accepts_374(self):
        for sample_count, expected in ((450, 375), (375, 375), (374, 374)):
            with self.subTest(sample_count=sample_count):
                samples = self.synthetic_eeg(15.0, sample_count).T
                processed = process_blink_segment(samples)
                self.assertEqual(processed.shape, (8, expected))
                self.assertTrue(np.all(np.isfinite(processed)))

    def test_too_short_segment_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "too short"):
            process_blink_segment(np.zeros((8, 8)))

    def test_four_active_channels_in_eight_value_record(self):
        samples = np.zeros((SEGMENT_SAMPLES, 8))
        samples[:, :4] = self.synthetic_eeg(10.0)[:4].T
        samples[:, 1:4] = -8388608  # Invalid ADC channels in the user's sample.
        samples[100, 0] = 8388607  # Isolated clipping must not dominate filtering.

        processed = process_blink_segment(samples)
        self.assertEqual(processed.shape, (8, SEGMENT_SAMPLES))
        self.assertTrue(np.all(processed[1:4] == 0))
        self.assertTrue(np.all(processed[4:] == 0))
        self.assertEqual(
            classify_segment(processed, self.frequencies).prediction_frequency, 10.0
        )

    def test_channel_selection_defaults_to_eight_and_four_mode_uses_first_four(self):
        samples = self.synthetic_eeg(12.0).T
        self.assertEqual(process_blink_segment(samples).shape, (8, SEGMENT_SAMPLES))
        four_channels = process_blink_segment(samples, channel_count=4)
        self.assertEqual(four_channels.shape, (4, SEGMENT_SAMPLES))
        self.assertEqual(
            classify_segment(four_channels, self.frequencies).prediction_frequency,
            12.0,
        )
        with self.assertRaisesRegex(ValueError, "selected 8 channels"):
            process_blink_segment(samples[:, :4])

    def test_segment_without_usable_channels_is_rejected(self):
        samples = np.zeros((SEGMENT_SAMPLES, 8))
        samples[:, :4] = -8388608
        with self.assertRaisesRegex(ValueError, "no usable EEG channels"):
            process_blink_segment(samples)

    def test_four_mode_accepts_four_or_eight_value_records(self):
        predictions = []
        eeg = self.synthetic_eeg(12.0)[:4].T
        with tempfile.TemporaryDirectory() as temp_dir:
            processor = BlinkSegmentProcessor(
                Path(temp_dir) / "recording.txt",
                target_frequencies=self.frequencies,
                on_prediction=predictions.append,
                output_dir=temp_dir,
                channel_count=4,
            )
            self.assertEqual(
                parse_eeg_line(f"[0.0] {{'fn': 1, 'eeg': {eeg[0].tolist()}}}").shape,
                (4,),
            )
            for index, sample in enumerate(eeg):
                marker = " [MARKER] blink" if index == 0 else ""
                processor._consume_line(
                    f"[{index / 250.0}] {{'fn': {index % 256}, "
                    f"'eeg': {sample.tolist()}}}{marker}\n"
                )
            processor._consume_line("[1.5] [MARKER] pause\n")

            self.assertEqual(len(predictions), 1)
            self.assertEqual(predictions[0].prediction_frequency, 12.0)
            self.assertEqual(
                np.load(Path(temp_dir) / "eeg_segments.npy").shape, (1, 4, 375)
            )
            self.assertEqual(
                np.load(Path(temp_dir) / "eeg_channel_counts.npy").tolist(), [4]
            )

            processor._consume_line("[2.0] [MARKER] blink\n")
            for sample in self.synthetic_eeg(15.0).T:
                processor._consume_line(f"[2.0] {{'eeg': {sample.tolist()}}}\n")
            processor._consume_line("[3.5] [MARKER] pause\n")

            self.assertEqual(len(predictions), 2)
            self.assertEqual(predictions[1].prediction_frequency, 15.0)
            self.assertEqual(
                np.load(Path(temp_dir) / "eeg_segments.npy").shape, (2, 4, 375)
            )
            self.assertEqual(
                np.load(Path(temp_dir) / "eeg_channel_counts.npy").tolist(), [4, 4]
            )

    def test_complete_marker_pair_saves_one_dynamic_prediction(self):
        predictions = []
        eeg = self.synthetic_eeg(12.0).T
        with tempfile.TemporaryDirectory() as temp_dir:
            processor = BlinkSegmentProcessor(
                Path(temp_dir) / "recording.txt",
                target_frequencies=self.frequencies,
                on_prediction=predictions.append,
                output_dir=temp_dir,
            )
            output = io.StringIO()
            with redirect_stdout(output):
                for index, sample in enumerate(eeg):
                    marker = " [MARKER] blink" if index == 0 else ""
                    processor._consume_line(
                        f"[{index / 250.0}] {{'fn': {index % 256}, "
                        f"'eeg': {sample.tolist()}}}{marker}\n"
                    )
                processor._consume_line("[1.5] {'fn': 0, 'eeg': [0,0,0,0,0,0,0,0]} [MARKER] pause\n")

            self.assertEqual(output.getvalue(), "")
            self.assertEqual(len(predictions), 1)
            self.assertEqual(predictions[0].sample_count, 375)
            self.assertEqual(predictions[0].prediction_frequency, 12.0)
            self.assertEqual(np.load(Path(temp_dir) / "fbcca_scores.npy").shape, (1, 3))
            with (Path(temp_dir) / "classification_results.csv").open(
                encoding="utf-8"
            ) as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["prediction_frequency_hz"], "12.0")

    def test_per_segment_candidates_exclude_deleted_frequencies_and_save_variable_lengths(self):
        candidates = {1: (15., 10.), 2: (12.,), 3: (10., 12., 15.)}
        predictions = []
        with tempfile.TemporaryDirectory() as directory:
            processor = BlinkSegmentProcessor(Path(directory)/'recording.txt',
                output_dir=directory, target_frequencies=self.frequencies,
                candidate_frequencies=lambda index: candidates[index],
                on_prediction=predictions.append)
            for index, frequency in enumerate((10., 12., 15.), 1):
                processor._consume_line('[MARKER] blink\n')
                for sample in self.synthetic_eeg(frequency).T:
                    processor._consume_line(f"[0.] {{'eeg': {sample.tolist()}}}\n")
                processor._consume_line('[MARKER] pause\n')
            self.assertEqual([p.prediction_frequency for p in predictions], [10., 12., 15.])
            self.assertEqual([p.target_frequencies for p in predictions], list(candidates.values()))
            self.assertEqual([p.prediction_index for p in predictions], [1, 0, 2])
            scores = np.load(Path(directory)/'fbcca_scores.npy')
            self.assertEqual(scores.shape, (3, 3))
            self.assertTrue(np.isnan(scores[0, 1]))
            self.assertTrue(np.isnan(scores[1, (0, 2)]).all())
            self.assertTrue(np.isfinite(scores[2]).all())
            self.assertEqual(np.load(Path(directory)/'fbcca_prediction_indexes.npy').tolist(), [2, 1, 0])
            self.assertEqual(np.load(Path(directory)/'fbcca_target_frequencies.npy').tolist(), self.frequencies)
            with (Path(directory)/'classification_results.csv').open(encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([json.loads(row['target_frequencies_hz']) for row in rows],
                             [list(value) for value in candidates.values()])

    def test_outside_region_signal_cannot_select_an_excluded_frequency(self):
        with tempfile.TemporaryDirectory() as directory:
            predictions = []
            processor = BlinkSegmentProcessor(Path(directory)/'recording.txt',
                output_dir=directory, target_frequencies=self.frequencies,
                candidate_frequencies=lambda index: (12., 10.), on_prediction=predictions.append)
            processor._samples = list(self.synthetic_eeg(15.).T)
            processor._process_current_segment(1)
            self.assertEqual(len(predictions), 1)
            self.assertIn(predictions[0].prediction_frequency, (12., 10.))
            self.assertEqual(predictions[0].scores.shape, (2,))

    def test_missing_or_empty_region_candidates_never_fall_back_to_global_set(self):
        def missing(index):
            raise ValueError('Missing region metadata')
        for callback in (lambda index: (), missing):
            with self.subTest(callback=callback), tempfile.TemporaryDirectory() as directory:
                predictions = []
                processor = BlinkSegmentProcessor(Path(directory)/'recording.txt',
                    output_dir=directory, candidate_frequencies=callback, on_prediction=predictions.append)
                with patch('analyse.test_four.classify_segment') as classify:
                    processor._process_current_segment(1)
                classify.assert_not_called()
                self.assertEqual(predictions, [])


if __name__ == "__main__":
    unittest.main()
