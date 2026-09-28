"""Online filter-bank CCA for one SSVEP stimulation segment.

The implementation mirrors the active code in ``fbcca.ipynb`` while exposing
an importable API for the real-time processor. The notebook remains unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.signal import cheb1ord, cheby1, filtfilt
from sklearn.cross_decomposition import CCA


PASSBAND_HZ = np.array([6, 14, 22, 30, 38, 46, 54, 62, 70, 78], dtype=float)
STOPBAND_HZ = np.array([4, 10, 16, 24, 32, 40, 48, 56, 64, 72], dtype=float)


@dataclass(frozen=True)
class ClassificationResult:
    """FBCCA scores and the winning configured target frequency."""

    scores: np.ndarray
    prediction_index: int
    prediction_frequency: float


def filterbank(eeg: np.ndarray, fs: float, idx_fb: int = 1) -> np.ndarray:
    """Apply one notebook-compatible Chebyshev filter bank."""
    data = np.asarray(eeg, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"EEG must have shape (channels, samples), got {data.shape}")
    if not 1 <= idx_fb <= len(PASSBAND_HZ):
        raise ValueError(f"filter bank index must be 1-{len(PASSBAND_HZ)}, got {idx_fb}")
    if fs <= 0:
        raise ValueError("sample rate must be positive")

    nyquist = fs / 2.0
    wp = [PASSBAND_HZ[idx_fb - 1] / nyquist, 90.0 / nyquist]
    ws = [STOPBAND_HZ[idx_fb - 1] / nyquist, 100.0 / nyquist]
    order, wn = cheb1ord(wp, ws, 3, 40)
    b, a = cheby1(order, 0.5, wn, btype="bandpass")
    padlen = 3 * max(len(a), len(b))
    if data.shape[1] <= padlen:
        raise ValueError(
            f"segment is too short for filter bank {idx_fb}: "
            f"got {data.shape[1]} samples, need more than {padlen}"
        )
    return filtfilt(b, a, data, axis=1)


def _reference(frequency: float, sample_count: int, fs: float, harmonics: int) -> np.ndarray:
    timepoints = np.arange(sample_count, dtype=float) / fs
    waves: list[np.ndarray] = []
    for harmonic in range(1, harmonics + 1):
        waves.append(np.sin(2.0 * np.pi * harmonic * frequency * timepoints))
        waves.append(np.cos(2.0 * np.pi * harmonic * frequency * timepoints))
    return np.asarray(waves).T


def classify_segment(
    eeg: np.ndarray,
    frequencies: Sequence[float],
    fs: float = 250.0,
    *,
    num_harmonics: int = 4,
    num_filter_banks: int = 5,
) -> ClassificationResult:
    """Classify one already-preprocessed SSVEP segment with FBCCA."""
    data = np.asarray(eeg, dtype=np.float64)
    target_frequencies = np.asarray(frequencies, dtype=float)
    if data.ndim != 2 or data.shape[0] < 1:
        raise ValueError(f"EEG must have shape (channels, samples), got {data.shape}")
    if target_frequencies.ndim != 1 or target_frequencies.size < 1:
        raise ValueError("at least one target frequency is required")
    if not np.all(np.isfinite(target_frequencies)) or np.any(target_frequencies <= 0):
        raise ValueError("target frequencies must be finite and positive")
    if num_harmonics < 1:
        raise ValueError("num_harmonics must be positive")
    if not 1 <= num_filter_banks <= len(PASSBAND_HZ):
        raise ValueError(f"num_filter_banks must be 1-{len(PASSBAND_HZ)}")

    coefficients = np.asarray(
        [index ** (-1.25) + 0.25 for index in range(1, num_filter_banks + 1)],
        dtype=float,
    )
    filtered = [filterbank(data, fs, index) for index in range(1, num_filter_banks + 1)]
    scores: list[float] = []

    for frequency in target_frequencies:
        reference = _reference(float(frequency), data.shape[1], fs, num_harmonics)
        correlations: list[float] = []
        for filtered_eeg in filtered:
            good_channels = np.std(filtered_eeg, axis=1) > 1e-8
            usable = filtered_eeg[good_channels]
            if usable.shape[0] < 1:
                correlations.append(0.0)
                continue
            cca = CCA(n_components=1, max_iter=1000)
            cca.fit(usable.T, reference)
            eeg_component, reference_component = cca.transform(usable.T, reference)
            correlation = float(
                np.corrcoef(eeg_component[:, 0], reference_component[:, 0])[0, 1]
            )
            correlations.append(0.0 if not np.isfinite(correlation) else correlation)
        scores.append(float(np.sum(coefficients * np.square(correlations))))

    score_array = np.asarray(scores, dtype=np.float32)
    prediction_index = int(np.argmax(score_array))
    return ClassificationResult(
        scores=score_array,
        prediction_index=prediction_index,
        prediction_frequency=float(target_frequencies[prediction_index]),
    )
