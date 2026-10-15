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


SUPPORTED_CHANNEL_COUNTS = (4, 8)
ADC_LIMITS = (-8388608, 8388607)
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
    target_frequencies: tuple[float, ...] = ()


def parse_eeg_line(line: str) -> Optional[np.ndarray]:
    """Extract four or eight EEG channel values from one recording line."""
    start = line.find("{")
    end = line.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        frame = ast.literal_eval(line[start : end + 1])
        eeg = np.asarray(frame["eeg"], dtype=float)
    except (KeyError, TypeError, ValueError, SyntaxError):
        return None
    return eeg if eeg.ndim == 1 and eeg.size in SUPPORTED_CHANNEL_COUNTS else None


def process_blink_segment(
    samples: Sequence[np.ndarray], channel_count: int = 8
) -> np.ndarray:
    """Preprocess the selected channels from one 1.5-second segment."""
    if channel_count not in SUPPORTED_CHANNEL_COUNTS:
        raise ValueError(f"channel_count must be 4 or 8, got {channel_count}")
    eeg = np.asarray(samples, dtype=float)
    if eeg.ndim != 2 or eeg.shape[1] not in SUPPORTED_CHANNEL_COUNTS:
        raise ValueError(
            f"expected samples x 4 or 8 channels, got {eeg.shape}"
        )
    if eeg.shape[1] < channel_count:
        raise ValueError(
            f"selected {channel_count} channels, but recording has {eeg.shape[1]}"
        )
    eeg = eeg[:SEGMENT_SAMPLES, :channel_count].T

    # A four-channel device may still write eight values per frame, padding the
    # unused channels with zero. ADC rail values mark clipped samples. Discard
    # channels with too little usable data and repair isolated clipped samples
    # before filtering so they cannot dominate the SSVEP spectrum.
    for channel in eeg:
        invalid = ~np.isfinite(channel) | np.isin(channel, ADC_LIMITS)
        valid = ~invalid
        if np.count_nonzero(valid) < max(2, int(np.ceil(0.8 * channel.size))):
            channel[:] = 0
        elif np.any(invalid):
            positions = np.arange(channel.size)
            channel[invalid] = np.interp(
                positions[invalid], positions[valid], channel[valid]
            )

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
    if not np.any(standard_deviations > 1e-8):
        raise ValueError("no usable EEG channels in segment")
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
        channel_count: int = 8,
        candidate_frequencies: Optional[Callable[[int], Sequence[float]]] = None,
    ) -> None:
        super().__init__(daemon=True)
        if channel_count not in SUPPORTED_CHANNEL_COUNTS:
            raise ValueError(f"channel_count must be 4 or 8, got {channel_count}")
        self.txt_file = Path(txt_file)
        self.on_segment = on_segment
        self.on_prediction = on_prediction
        self.target_frequencies = np.asarray(target_frequencies, dtype=float)
        self.candidate_frequencies = candidate_frequencies
        self.channel_count = channel_count
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
            frequencies = np.asarray(self.target_frequencies if self.candidate_frequencies is None
                                     else self.candidate_frequencies(segment_index), dtype=float)
            if frequencies.ndim != 1 or frequencies.size == 0:
                raise ValueError('No candidate frequencies for this flashing region')
            if not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0):
                raise ValueError('Candidate frequencies must be finite and positive')
            if not np.all(np.any(np.isclose(frequencies[:, None], self.target_frequencies,
                                          rtol=0., atol=1e-9), axis=1)):
                raise ValueError('Candidate frequencies are not configured targets')
            processed = process_blink_segment(
                self._samples, channel_count=self.channel_count
            )
            classified: ClassificationResult = classify_segment(
                processed, frequencies, fs=SAMPLE_RATE
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
            target_frequencies=tuple(float(value) for value in frequencies),
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
        max_channels = max(segment.shape[0] for segment in self._processed_segments)
        eeg_segments = np.stack(
            [
                np.pad(
                    segment[:, :min_timepoints],
                    ((0, max_channels - segment.shape[0]), (0, 0)),
                )
                for segment in self._processed_segments
            ]
        )
        np.save(self.output_dir / "eeg_segments.npy", eeg_segments)
        np.save(
            self.output_dir / "eeg_channel_counts.npy",
            np.asarray([segment.shape[0] for segment in self._processed_segments]),
        )
        # Regions may contain different numbers of targets. Keep a rectangular
        # global frequency axis; NaN explicitly marks candidates not evaluated.
        scores = np.full((len(self._predictions), len(self.target_frequencies)), np.nan,
                         dtype=np.float32)
        global_indexes = []
        for row, prediction in enumerate(self._predictions):
            frequencies = prediction.target_frequencies or tuple(self.target_frequencies)
            columns = [int(np.flatnonzero(np.isclose(self.target_frequencies, frequency,
                                                    rtol=0., atol=1e-9))[0])
                       for frequency in frequencies]
            scores[row, columns] = prediction.scores
            global_indexes.append(int(np.flatnonzero(np.isclose(
                self.target_frequencies, prediction.prediction_frequency, rtol=0., atol=1e-9
            ))[0]))
        np.save(self.output_dir / 'fbcca_scores.npy', scores)
        np.save(self.output_dir / 'fbcca_target_frequencies.npy', self.target_frequencies)
        np.save(
            self.output_dir / "fbcca_prediction_indexes.npy",
            np.asarray(global_indexes),
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
                    "prediction_global_index",
                ),
            )
            writer.writeheader()
            for index, prediction in enumerate(self._predictions):
                writer.writerow(
                    {
                        "segment_index": prediction.segment_index,
                        "sample_count": prediction.sample_count,
                        "target_frequencies_hz": json.dumps(
                            list(prediction.target_frequencies or tuple(self.target_frequencies))
                        ),
                        "scores": json.dumps(prediction.scores.tolist()),
                        "prediction_index": prediction.prediction_index,
                        "prediction_frequency_hz": prediction.prediction_frequency,
                        "prediction_global_index": global_indexes[index],
                    }
                )

        latest = self._predictions[-1]
        self.log(
            f"Processed blink segment {latest.segment_index}: "
            f"shape={self._processed_segments[-1].shape}; "
            f"FBCCA scores={latest.scores.tolist()}; "
            f"candidates={list(latest.target_frequencies)}; "
            f"prediction={latest.prediction_frequency:g} Hz"
        )

    def log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.log_file.open("a", encoding="utf-8") as log:
            log.write(f"[{timestamp}] {message}\n")
