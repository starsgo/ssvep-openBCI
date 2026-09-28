import csv
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from analyse.fbcca import classify_segment
from analyse.test_four import (
    BlinkSegmentProcessor,
    SEGMENT_SAMPLES,
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


if __name__ == "__main__":
    unittest.main()
