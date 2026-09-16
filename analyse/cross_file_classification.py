"""Train template-based SSVEP classifiers on one file and test on another."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analyse.single_file_classification import (
    FREQUENCIES,
    SAMPLE_RATE,
    fbcca_predictions,
    itcca_train_predict,
    load_repeated_labels,
    load_segments,
    load_visual_validity,
    result_for,
    save_confusion_plot,
    standardize_trials,
    trca_train_predict,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_eeg", type=Path)
    parser.add_argument("train_labels", type=Path)
    parser.add_argument("train_timing", type=Path)
    parser.add_argument("test_eeg", type=Path)
    parser.add_argument("test_labels", type=Path)
    parser.add_argument("test_timing", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-invalid-train", action="store_true")
    parser.add_argument("--include-invalid-test", action="store_true")
    args = parser.parse_args()

    train_segments_all = load_segments(args.train_eeg)
    test_segments_all = load_segments(args.test_eeg)
    train_labels_all = load_repeated_labels(args.train_labels, len(train_segments_all))
    test_labels_all = load_repeated_labels(args.test_labels, len(test_segments_all))
    train_validity = load_visual_validity(args.train_timing, len(train_segments_all))
    test_validity = load_visual_validity(args.test_timing, len(test_segments_all))

    train_selection = (
        np.ones(len(train_segments_all), dtype=bool)
        if args.include_invalid_train
        else train_validity
    )
    test_selection = (
        np.ones(len(test_segments_all), dtype=bool)
        if args.include_invalid_test
        else test_validity
    )
    train_trial_ids = np.flatnonzero(train_selection) + 1
    test_trial_ids = np.flatnonzero(test_selection) + 1
    train_segments = [x for x, selected in zip(train_segments_all, train_selection) if selected]
    test_segments = [x for x, selected in zip(test_segments_all, test_selection) if selected]
    train_y = train_labels_all[train_selection]
    test_y = test_labels_all[test_selection]

    # One shared sample length is required by the learned templates. Filtering and
    # normalization remain trial-local, so no test signal contributes to training.
    combined, common_samples = standardize_trials(train_segments + test_segments)
    train_x = combined[: len(train_segments)]
    test_x = combined[len(train_segments) :]

    fbcca_pred, fbcca_scores = fbcca_predictions(test_x)
    itcca_pred = itcca_train_predict(train_x, train_y, test_x)
    trca_pred = trca_train_predict(train_x, train_y, test_x)
    predictions = {
        "FBCCA": fbcca_pred,
        "ITCCA": itcca_pred,
        "TRCA": trca_pred,
    }
    results = {name: result_for(test_y, pred) for name, pred in predictions.items()}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "evaluation": "cross-file: train only on train_eeg, evaluate only on test_eeg",
        "train_eeg": str(args.train_eeg.resolve()),
        "test_eeg": str(args.test_eeg.resolve()),
        "train_marker_segments": len(train_segments_all),
        "train_valid_trials": int(train_validity.sum()),
        "train_excluded_trials": int((~train_validity).sum()),
        "train_trials_used": int(train_selection.sum()),
        "include_invalid_train": args.include_invalid_train,
        "train_class_counts": np.bincount(train_y, minlength=len(FREQUENCIES)).tolist(),
        "test_marker_segments": len(test_segments_all),
        "test_valid_trials": int(test_validity.sum()),
        "test_excluded_trials": int((~test_validity).sum()),
        "test_trials_used": int(test_selection.sum()),
        "include_invalid_test": args.include_invalid_test,
        "test_class_counts": np.bincount(test_y, minlength=len(FREQUENCIES)).tolist(),
        "sample_rate_hz": SAMPLE_RATE,
        "common_samples": common_samples,
        "analysis_duration_s": common_samples / SAMPLE_RATE,
        "frequencies_hz": FREQUENCIES.tolist(),
        "results": results,
    }
    (args.output_dir / "classification_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "test_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.writer(output)
        writer.writerow(
            ["valid_test_order", "original_trial_id", "true_label", "true_hz",
             "fbcca_hz", "itcca_hz", "trca_hz"]
        )
        for index, label in enumerate(test_y):
            writer.writerow(
                [index + 1, int(test_trial_ids[index]), int(label), FREQUENCIES[label],
                 FREQUENCIES[fbcca_pred[index]], FREQUENCIES[itcca_pred[index]],
                 FREQUENCIES[trca_pred[index]]]
            )
    np.save(args.output_dir / "fbcca_scores.npy", fbcca_scores)
    np.save(args.output_dir / "train_trial_ids.npy", train_trial_ids)
    save_confusion_plot(results, args.output_dir / "confusion_matrices.png")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

cd D:\documents\vibe4

python -B -m analyse.cross_file_classification `
  test_6_ssvep.txt `
  labels_test_6_ssvep.txt `
  record\20260914_204523\timing_summary.json `
  test_8_ssvep.txt `
  labels_test_8_ssvep.txt `
  record\20260914_211802\timing_summary.json `
  --include-invalid-train `
  --include-invalid-test `
  --output-dir analyse\cross_file_results\train6_test8_all

'''
