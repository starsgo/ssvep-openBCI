"""Real-time parsing, preprocessing, and classification of SSVEP segments."""

from __future__ import annotations

import ast
import csv
import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

from .fbcca import ClassificationResult, classify_segment


NUM_CHANNELS = 8
SAMPLE_RATE = 250
SEGMENT_SECONDS = 1.5
SEGMENT_SAMPLES = round(SEGMENT_SECONDS * SAMPLE_RATE)
TARGET_FREQS = np.array([15.0, 12.0, 10.0])


@dataclass(frozen=True)
class PredictionResult:
    """One completed online ``blink -> pause`` classification."""

    segment_index: int
    sample_count: int
    scores: np.ndarray
    prediction_index: int
    prediction_frequency: float


def parse_eeg_line(line: str) -> Optional[np.ndarray]:
    """Extract the eight EEG channel values from one SerialReader output line."""
    start = line.find("{")
    end = line.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        frame = ast.literal_eval(line[start : end + 1])
        eeg = np.asarray(frame["eeg"], dtype=float)
    except (KeyError, TypeError, ValueError, SyntaxError):
        return None
    return eeg if eeg.shape == (NUM_CHANNELS,) else None


def process_blink_segment(samples: Sequence[np.ndarray]) -> np.ndarray:
    """Preprocess at most the first 1.5 seconds of one independent segment."""
    eeg = np.asarray(samples, dtype=float)
    if eeg.ndim != 2 or eeg.shape[1] != NUM_CHANNELS:
        raise ValueError(f"expected samples x {NUM_CHANNELS} channels, got {eeg.shape}")
    eeg = eeg[:SEGMENT_SAMPLES].T

    band_b, band_a = butter(
        4, [5.0 / (SAMPLE_RATE / 2.0), 90.0 / (SAMPLE_RATE / 2.0)], btype="band"
    )
    notch_b, notch_a = iirnotch(50.0, 30.0, SAMPLE_RATE)
    minimum_samples = max(
        3 * max(len(band_a), len(band_b)),
        3 * max(len(notch_a), len(notch_b)),
    )
    if eeg.shape[1] <= minimum_samples:
        raise ValueError(
            f"segment is too short: got {eeg.shape[1]} samples, "
            f"need more than {minimum_samples}"
        )

    # Match the active test-four notebook: filter, de-mean, and standardize
    # each channel. No PCA or variance-times-1000 scaling is applied online.
    eeg = filtfilt(band_b, band_a, eeg, axis=1)
    eeg = filtfilt(notch_b, notch_a, eeg, axis=1)
    eeg -= np.mean(eeg, axis=1, keepdims=True)
    standard_deviations = np.std(eeg, axis=1, keepdims=True)
    eeg = np.divide(
        eeg,
        standard_deviations,
        out=np.zeros_like(eeg),
        where=standard_deviations > 1e-8,
    )
    return eeg.astype(np.float32)


def process_fbcca_segment(
    processed: np.ndarray,
    target_frequencies: Sequence[float] = TARGET_FREQS,
) -> tuple[np.ndarray, int, float]:
    """Compatibility wrapper returning scores, index, and frequency."""
    result = classify_segment(processed, target_frequencies, fs=SAMPLE_RATE)
    return result.scores, result.prediction_index, result.prediction_frequency


class BlinkSegmentProcessor(threading.Thread):
    """Tail a recording and classify every complete ``blink -> pause`` segment."""

    def __init__(
        self,
        txt_file: str | Path,
        on_segment: Optional[Callable[[np.ndarray], None]] = None,
        on_prediction: Optional[Callable[[PredictionResult], None]] = None,
        target_frequencies: Sequence[float] = TARGET_FREQS,
        poll_interval: float = 0.05,
        start_at_end: bool = True,
        output_dir: str | Path | None = None,
    ) -> None:
        super().__init__(daemon=True)
        self.txt_file = Path(txt_file)
        self.on_segment = on_segment
        self.on_prediction = on_prediction
        self.target_frequencies = np.asarray(target_frequencies, dtype=float)
        if self.target_frequencies.ndim != 1 or self.target_frequencies.size < 1:
            raise ValueError("at least one target frequency is required")
        if np.any(self.target_frequencies <= 0) or not np.all(
            np.isfinite(self.target_frequencies)
        ):
            raise ValueError("target frequencies must be finite and positive")
        self.poll_interval = float(poll_interval)
        self.start_at_end = bool(start_at_end)
        self.output_dir = (
            Path(output_dir)
            if output_dir is not None
            else Path(__file__).parent / "realtime_results"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.output_dir / "runtime.log"
        self.running = True
        self._collecting = False
        self._samples: list[np.ndarray] = []
        self._segment_index = 0
        self._processed_segments: list[np.ndarray] = []
        self._predictions: list[PredictionResult] = []

    def run(self) -> None:
        while self.running and not self.txt_file.exists():
            time.sleep(self.poll_interval)
        if not self.running:
            return
        with self.txt_file.open("r", encoding="utf-8") as txt:
            if self.start_at_end:
                txt.seek(0, 2)
            while self.running:
                line = txt.readline()
                if not line:
                    time.sleep(self.poll_interval)
                    continue
                self._consume_line(line)

    def stop(self) -> None:
        self.running = False

    def _consume_line(self, line: str) -> None:
        if "[MARKER] blink" in line:
            self._collecting = True
            self._samples = []
        elif "[MARKER] pause" in line:
            if self._collecting:
                self._segment_index += 1
                self._process_current_segment(self._segment_index)
            self._collecting = False
            self._samples = []
            return

        if self._collecting:
            eeg = parse_eeg_line(line)
            if eeg is not None:
                self._samples.append(eeg)

    def _process_current_segment(self, segment_index: int) -> None:
        try:
            processed = process_blink_segment(self._samples)
            classified: ClassificationResult = classify_segment(
                processed, self.target_frequencies, fs=SAMPLE_RATE
            )
        except Exception as error:
            self.log(f"Skipped blink segment {segment_index}: {type(error).__name__}: {error}")
            return

        prediction = PredictionResult(
            segment_index=segment_index,
            sample_count=processed.shape[1],
            scores=classified.scores,
            prediction_index=classified.prediction_index,
            prediction_frequency=classified.prediction_frequency,
        )
        self._processed_segments.append(processed)
        self._predictions.append(prediction)
        self._save_results()
        self._call_safely(self.on_segment, processed, "on_segment")
        self._call_safely(self.on_prediction, prediction, "on_prediction")

    def _call_safely(self, callback: Optional[Callable], value: object, name: str) -> None:
        if callback is None:
            return
        try:
            callback(value)
        except Exception as error:
            self.log(f"{name} callback failed: {type(error).__name__}: {error}")

    def _save_results(self) -> None:
        min_timepoints = min(segment.shape[1] for segment in self._processed_segments)
        eeg_segments = np.stack(
            [segment[:, :min_timepoints] for segment in self._processed_segments]
        )
        np.save(self.output_dir / "eeg_segments.npy", eeg_segments)
        np.save(
            self.output_dir / "fbcca_scores.npy",
            np.stack([prediction.scores for prediction in self._predictions]),
        )
        np.save(
            self.output_dir / "fbcca_prediction_indexes.npy",
            np.asarray([prediction.prediction_index for prediction in self._predictions]),
        )
        np.save(
            self.output_dir / "fbcca_prediction_frequencies.npy",
            np.asarray(
                [prediction.prediction_frequency for prediction in self._predictions]
            ),
        )

        csv_file = self.output_dir / "classification_results.csv"
        with csv_file.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=(
                    "segment_index",
                    "sample_count",
                    "target_frequencies_hz",
                    "scores",
                    "prediction_index",
                    "prediction_frequency_hz",
                ),
            )
            writer.writeheader()
            for prediction in self._predictions:
                writer.writerow(
                    {
                        "segment_index": prediction.segment_index,
                        "sample_count": prediction.sample_count,
                        "target_frequencies_hz": json.dumps(
                            self.target_frequencies.tolist()
                        ),
                        "scores": json.dumps(prediction.scores.tolist()),
                        "prediction_index": prediction.prediction_index,
                        "prediction_frequency_hz": prediction.prediction_frequency,
                    }
                )

        latest = self._predictions[-1]
        self.log(
            f"Processed blink segment {latest.segment_index}: "
            f"shape={self._processed_segments[-1].shape}; "
            f"FBCCA scores={latest.scores.tolist()}; "
            f"prediction={latest.prediction_frequency:g} Hz"
        )

    def log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.log_file.open("a", encoding="utf-8") as log:
            log.write(f"[{timestamp}] {message}\n")
