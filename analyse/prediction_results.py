"""Resolve online predictions to object INDEX values and save accuracy."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from analyse.test_four import PredictionResult
from experiment import TargetSpec
from gaze import Region


@dataclass(frozen=True)
class TrialMetadata:
    segment_index: int
    actual_index: int
    gaze_position: tuple[int, int]
    gaze_side: Region
    flicker_side: Region
    region_source: Literal["gaze", "target"]
    gaze_point_source: Literal["measurement", "history", "default"]


class PredictionResultRecorder:
    FIELDS = (
        "segment_index",
        "sample_count",
        "actual_index",
        "actual_name",
        "actual_side",
        "actual_x",
        "actual_y",
        "actual_width",
        "actual_height",
        "actual_frequency_hz",
        "gaze_x",
        "gaze_y",
        "gaze_side",
        "gaze_side_correct",
        "predicted_fbcca_index",
        "predicted_frequency_hz",
        "predicted_target_index",
        "predicted_target_name",
        "correct",
        "scores",
        "flicker_side",
        "region_source",
        "gaze_point_source",
        "actual_config_index",
        "predicted_config_index",
    )

    def __init__(
        self, targets: list[TargetSpec], results_file: str | Path | None
    ) -> None:
        self.targets = targets
        self.results_file = None if results_file is None else Path(results_file)
        self._metadata: dict[int, TrialMetadata] = {}
        self._outcomes: dict[int, bool] = {}

    def register(
        self,
        segment_index: int,
        actual_index: int,
        gaze_position: tuple[int, int],
        gaze_side: Region,
        *,
        flicker_side: Region | None = None,
        region_source: Literal["gaze", "target"] = "gaze",
        gaze_point_source: Literal["measurement", "history", "default"] = "measurement",
    ) -> TrialMetadata:
        metadata = TrialMetadata(
            segment_index,
            actual_index,
            (int(gaze_position[0]), int(gaze_position[1])),
            gaze_side,
            gaze_side if flicker_side is None else flicker_side,
            region_source,
            gaze_point_source,
        )
        self._metadata[segment_index] = metadata
        return metadata

    def metadata_for(self, segment_index: int) -> TrialMetadata | None:
        return self._metadata.get(segment_index)

    def resolve_target_index(self, frequency: float, side: Region) -> int:
        matches = [
            index
            for index, target in enumerate(self.targets)
            if target.screen == side
            and math.isclose(target.frequency_hz, frequency, rel_tol=0.0, abs_tol=1e-9)
        ]
        if len(matches) != 1:
            raise ValueError("frequency and flashing region must identify exactly one target")
        return matches[0]

    def record(self, prediction: PredictionResult) -> None:
        metadata = self.metadata_for(prediction.segment_index)
        if (
            metadata is None
            or self.results_file is None
            or prediction.segment_index in self._outcomes
        ):
            return

        predicted_index = self.resolve_target_index(
            prediction.prediction_frequency, metadata.flicker_side
        )
        actual = self.targets[metadata.actual_index]
        predicted = self.targets[predicted_index]
        correct = predicted_index == metadata.actual_index
        row = {
            "segment_index": prediction.segment_index,
            "sample_count": prediction.sample_count,
            "actual_index": metadata.actual_index if actual.label is None else actual.label,
            "actual_name": actual.name,
            "actual_side": actual.screen,
            "actual_x": actual.x,
            "actual_y": actual.y,
            "actual_width": actual.width,
            "actual_height": actual.height,
            "actual_frequency_hz": actual.frequency_hz,
            "gaze_x": metadata.gaze_position[0],
            "gaze_y": metadata.gaze_position[1],
            "gaze_side": metadata.gaze_side,
            "gaze_side_correct": metadata.gaze_side == actual.screen,
            "predicted_fbcca_index": prediction.prediction_index,
            "predicted_frequency_hz": prediction.prediction_frequency,
            "predicted_target_index": predicted_index if predicted.label is None else predicted.label,
            "predicted_target_name": predicted.name,
            "correct": correct,
            "scores": json.dumps(prediction.scores.tolist()),
            "flicker_side": metadata.flicker_side,
            "region_source": metadata.region_source,
            "gaze_point_source": metadata.gaze_point_source,
            "actual_config_index": metadata.actual_index,
            "predicted_config_index": predicted_index,
        }
        self._append(row)
        self._outcomes[prediction.segment_index] = correct
        self._save_accuracy()

    def _append(self, row: dict[str, object]) -> None:
        path = self.results_file
        assert path is not None
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _save_accuracy(self) -> None:
        path = self.results_file
        assert path is not None
        total = len(self._outcomes)
        correct = sum(self._outcomes.values())
        summary = {
            "total_predictions": total,
            "correct_predictions": correct,
            "incorrect_predictions": total - correct,
            "accuracy": correct / total,
            "accuracy_percent": correct / total * 100.0,
        }
        path.with_name("prediction_accuracy.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
