"""Evaluate FBCCA, ITCCA and TRCA on one marker-segmented SSVEP file."""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.linalg import eigh
from scipy.signal import butter, cheb1ord, cheby1, iirnotch, sosfiltfilt, tf2sos
from sklearn.cross_decomposition import CCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


matplotlib.use("Agg")
import matplotlib.pyplot as plt


FREQUENCIES = np.asarray([15.0, 12.0, 10.0])
SAMPLE_RATE = 250.0
NUM_HARMONICS = 4
NUM_FILTER_BANKS = 5


def parse_eeg(line: str) -> np.ndarray | None:
    start = line.find("{")
    end = line.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        frame = ast.literal_eval(line[start : end + 1])
        eeg = np.asarray(frame["eeg"], dtype=np.float64)
    except (KeyError, TypeError, ValueError, SyntaxError):
        return None
    return eeg if eeg.shape == (8,) and np.all(np.isfinite(eeg)) else None


def load_segments(path: Path) -> list[np.ndarray]:
    segments: list[np.ndarray] = []
    current: list[np.ndarray] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        eeg = parse_eeg(line)
        if "[MARKER] blink" in line:
            current = []
            if eeg is not None:
                current.append(eeg)
        elif "[MARKER] pause" in line:
            if current:
                segments.append(np.asarray(current, dtype=np.float64).T)
            current = None
        elif current is not None and eeg is not None:
            current.append(eeg)
    return segments


def load_repeated_labels(path: Path, count: int) -> np.ndarray:
    base = np.asarray(
        [int(line.strip()) for line in path.read_text().splitlines() if line.strip()],
        dtype=int,
    )
    if base.size == 0:
        raise ValueError("label file is empty")
    labels = np.resize(base, count)
    if np.any((labels < 0) | (labels >= len(FREQUENCIES))):
        raise ValueError(f"labels must be 0-{len(FREQUENCIES) - 1}")
    return labels


def load_visual_validity(path: Path, count: int) -> np.ndarray:
    summary = json.loads(path.read_text(encoding="utf-8"))
    trials = summary.get("trials", [])
    if len(trials) < count:
        raise ValueError(
            f"timing summary has {len(trials)} trials but EEG has {count} segments"
        )
    return np.asarray([bool(trial["valid"]) for trial in trials[:count]])


def standardize_trials(segments: list[np.ndarray]) -> tuple[np.ndarray, int]:
    common_samples = min(segment.shape[1] for segment in segments)
    data = np.stack([segment[:, :common_samples] for segment in segments])
    notch_b, notch_a = iirnotch(50.0, 30.0, SAMPLE_RATE)
    notch_sos = tf2sos(notch_b, notch_a)
    band_sos = butter(
        4,
        [5.0, 90.0],
        btype="bandpass",
        fs=SAMPLE_RATE,
        output="sos",
    )
    data = sosfiltfilt(notch_sos, data, axis=2)
    data = sosfiltfilt(band_sos, data, axis=2)
    data -= data.mean(axis=2, keepdims=True)
    std = data.std(axis=2, keepdims=True)
    data = np.divide(data, std, out=np.zeros_like(data), where=std > 1e-10)
    return data, common_samples


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a -= a.mean()
    b -= b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return 0.0 if denominator <= 1e-12 else float(np.dot(a, b) / denominator)


def cca_correlation(x: np.ndarray, reference: np.ndarray) -> float:
    good = np.std(x, axis=1) > 1e-10
    x = x[good]
    if x.shape[0] == 0:
        return 0.0
    cca = CCA(n_components=1, max_iter=1000)
    x_component, ref_component = cca.fit_transform(x.T, reference)
    value = correlation(x_component[:, 0], ref_component[:, 0])
    return 0.0 if not np.isfinite(value) else value


def filter_bank_sos(index: int) -> np.ndarray:
    passband = np.asarray([6, 14, 22, 30, 38, 46, 54, 62, 70, 78])
    stopband = np.asarray([4, 10, 16, 24, 32, 40, 48, 56, 64, 72])
    nyquist = SAMPLE_RATE / 2.0
    wp = [passband[index - 1] / nyquist, 90.0 / nyquist]
    ws = [stopband[index - 1] / nyquist, 100.0 / nyquist]
    order, wn = cheb1ord(wp, ws, 3, 40)
    return cheby1(order, 0.5, wn, btype="bandpass", output="sos")


def fbcca_predictions(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sample_count = data.shape[2]
    time = np.arange(sample_count) / SAMPLE_RATE
    references = []
    for frequency in FREQUENCIES:
        columns = []
        for harmonic in range(1, NUM_HARMONICS + 1):
            columns.append(np.sin(2 * np.pi * harmonic * frequency * time))
            columns.append(np.cos(2 * np.pi * harmonic * frequency * time))
        references.append(np.asarray(columns).T)
    coefficients = np.asarray(
        [index ** (-1.25) + 0.25 for index in range(1, NUM_FILTER_BANKS + 1)]
    )
    banks = [filter_bank_sos(index) for index in range(1, NUM_FILTER_BANKS + 1)]
    scores = np.zeros((data.shape[0], len(FREQUENCIES)), dtype=np.float64)
    for trial_index, trial in enumerate(data):
        filtered = [sosfiltfilt(sos, trial, axis=1) for sos in banks]
        for target_index, reference in enumerate(references):
            correlations = np.asarray(
                [cca_correlation(bank, reference) for bank in filtered]
            )
            scores[trial_index, target_index] = np.sum(
                coefficients * np.square(correlations)
            )
    return np.argmax(scores, axis=1), scores


def itcca_leave_one_out(data: np.ndarray, labels: np.ndarray) -> np.ndarray:
    predictions = []
    for test_index, test in enumerate(data):
        scores = []
        for label in range(len(FREQUENCIES)):
            train = data[(labels == label) & (np.arange(len(labels)) != test_index)]
            template = train.mean(axis=0)
            scores.append(cca_correlation(test, template.T))
        predictions.append(int(np.argmax(np.square(scores))))
    return np.asarray(predictions, dtype=int)


def itcca_train_predict(
    train_data: np.ndarray,
    train_labels: np.ndarray,
    test_data: np.ndarray,
) -> np.ndarray:
    templates = [
        train_data[train_labels == label].mean(axis=0)
        for label in range(len(FREQUENCIES))
    ]
    predictions = []
    for test in test_data:
        scores = [cca_correlation(test, template.T) for template in templates]
        predictions.append(int(np.argmax(np.square(scores))))
    return np.asarray(predictions, dtype=int)


def train_trca_class(class_trials: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = class_trials - class_trials.mean(axis=2, keepdims=True)
    summed = centered.sum(axis=0)
    q_matrix = np.einsum("nct,ndt->cd", centered, centered)
    s_matrix = summed @ summed.T - q_matrix
    scale = max(float(np.trace(q_matrix)) / q_matrix.shape[0], 1.0)
    regularized_q = q_matrix + 1e-6 * scale * np.eye(q_matrix.shape[0])
    _, eigenvectors = eigh(s_matrix, regularized_q)
    spatial_filter = eigenvectors[:, -1]
    spatial_filter /= np.linalg.norm(spatial_filter) + 1e-12
    return spatial_filter, centered.mean(axis=0)


def trca_leave_one_out(data: np.ndarray, labels: np.ndarray) -> np.ndarray:
    predictions = []
    indexes = np.arange(len(labels))
    for test_index, test in enumerate(data):
        scores = []
        test = test - test.mean(axis=1, keepdims=True)
        for label in range(len(FREQUENCIES)):
            train = data[(labels == label) & (indexes != test_index)]
            spatial_filter, template = train_trca_class(train)
            scores.append(
                correlation(spatial_filter @ test, spatial_filter @ template)
            )
        predictions.append(int(np.argmax(scores)))
    return np.asarray(predictions, dtype=int)


def trca_train_predict(
    train_data: np.ndarray,
    train_labels: np.ndarray,
    test_data: np.ndarray,
) -> np.ndarray:
    model = [
        train_trca_class(train_data[train_labels == label])
        for label in range(len(FREQUENCIES))
    ]
    predictions = []
    for test in test_data:
        test = test - test.mean(axis=1, keepdims=True)
        scores = [
            correlation(spatial_filter @ test, spatial_filter @ template)
            for spatial_filter, template in model
        ]
        predictions.append(int(np.argmax(scores)))
    return np.asarray(predictions, dtype=int)


def result_for(labels: np.ndarray, predictions: np.ndarray) -> dict:
    matrix = confusion_matrix(labels, predictions, labels=np.arange(len(FREQUENCIES)))
    per_class = np.divide(
        np.diag(matrix),
        matrix.sum(axis=1),
        out=np.zeros(len(FREQUENCIES), dtype=float),
        where=matrix.sum(axis=1) > 0,
    )
    return {
        "accuracy": float(np.mean(labels == predictions)),
        "correct": int(np.sum(labels == predictions)),
        "total": int(len(labels)),
        "per_frequency_accuracy": {
            str(int(frequency)): float(value)
            for frequency, value in zip(FREQUENCIES, per_class)
        },
        "confusion_matrix": matrix.tolist(),
    }


def save_confusion_plot(results: dict, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    for axis, (name, result) in zip(axes, results.items()):
        matrix = np.asarray(result["confusion_matrix"])
        image = axis.imshow(matrix, cmap="Blues")
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
        axis.set_title(f"{name}: {result['accuracy']:.1%}")
        axis.set_xlabel("predicted frequency (Hz)")
        axis.set_ylabel("true frequency (Hz)")
        axis.set_xticks(range(len(FREQUENCIES)), FREQUENCIES.astype(int))
        axis.set_yticks(range(len(FREQUENCIES)), FREQUENCIES.astype(int))
        fig.colorbar(image, ax=axis, shrink=0.8)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("eeg_file", type=Path)
    parser.add_argument("label_file", type=Path)
    parser.add_argument("timing_summary", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    segments = load_segments(args.eeg_file)
    labels = load_repeated_labels(args.label_file, len(segments))
    validity = load_visual_validity(args.timing_summary, len(segments))
    valid_trial_ids = np.flatnonzero(validity) + 1
    valid_segments = [segment for segment, valid in zip(segments, validity) if valid]
    valid_labels = labels[validity]
    data, common_samples = standardize_trials(valid_segments)

    fbcca_pred, fbcca_scores = fbcca_predictions(data)
    itcca_pred = itcca_leave_one_out(data, valid_labels)
    trca_pred = trca_leave_one_out(data, valid_labels)
    predictions = {
        "FBCCA": fbcca_pred,
        "ITCCA": itcca_pred,
        "TRCA": trca_pred,
    }
    results = {name: result_for(valid_labels, pred) for name, pred in predictions.items()}

    train_indexes, test_indexes = train_test_split(
        np.arange(len(valid_labels)),
        test_size=0.5,
        random_state=42,
        stratify=valid_labels,
    )
    holdout_predictions = {
        "FBCCA": fbcca_pred[test_indexes],
        "ITCCA": itcca_train_predict(
            data[train_indexes], valid_labels[train_indexes], data[test_indexes]
        ),
        "TRCA": trca_train_predict(
            data[train_indexes], valid_labels[train_indexes], data[test_indexes]
        ),
    }
    holdout_results = {
        name: result_for(valid_labels[test_indexes], pred)
        for name, pred in holdout_predictions.items()
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "eeg_file": str(args.eeg_file.resolve()),
        "marker_segments": len(segments),
        "valid_visual_trials": int(validity.sum()),
        "excluded_visual_trials": int((~validity).sum()),
        "sample_rate_hz": SAMPLE_RATE,
        "common_samples": common_samples,
        "analysis_duration_s": common_samples / SAMPLE_RATE,
        "frequencies_hz": FREQUENCIES.tolist(),
        "evaluation": {
            "leave_one_out": "ITCCA/TRCA train on 87 and test on 1 per fold",
            "stratified_holdout": "44 training trials and 44 test trials; random_state=42",
        },
        "leave_one_out_results": results,
        "stratified_holdout_results": holdout_results,
    }
    (args.output_dir / "classification_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "trial_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "valid_trial_order",
                "original_trial_id",
                "true_label",
                "true_hz",
                "fbcca",
                "itcca",
                "trca",
            ]
        )
        for index, label in enumerate(valid_labels):
            writer.writerow(
                [
                    index + 1,
                    int(valid_trial_ids[index]),
                    int(label),
                    FREQUENCIES[label],
                    FREQUENCIES[fbcca_pred[index]],
                    FREQUENCIES[itcca_pred[index]],
                    FREQUENCIES[trca_pred[index]],
                ]
            )
    np.save(args.output_dir / "fbcca_scores.npy", fbcca_scores)
    save_confusion_plot(results, args.output_dir / "confusion_matrices.png")
    save_confusion_plot(
        holdout_results, args.output_dir / "confusion_matrices_holdout.png"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
