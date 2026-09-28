"""Experiment target configuration."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from gaze import Region


@dataclass(frozen=True)
class TargetSpec:
    name: str
    screen: Region
    x: float
    y: float
    width: float
    height: float
    frequency_hz: float

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.width / 2.0, self.y + self.height / 2.0


def load_targets(
    path: str | Path = "object.json",
    *,
    window_size: tuple[int, int] = (1920, 1080),
) -> list[TargetSpec]:
    """Load and validate the ordered target list used by UI and classifier."""
    config_path = Path(path)
    try:
        document = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"target config does not exist: {config_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid target config JSON: {error}") from error

    raw_targets = document.get("targets") if isinstance(document, dict) else None
    if not isinstance(raw_targets, list) or not raw_targets:
        raise ValueError("object.json must contain a non-empty 'targets' array")

    targets: list[TargetSpec] = []
    for index, raw in enumerate(raw_targets):
        if not isinstance(raw, dict):
            raise ValueError(f"target {index} must be an object")
        try:
            target = TargetSpec(
                name=str(raw["name"]).strip(),
                screen=str(raw["screen"]).strip().lower(),
                x=float(raw["x"]),
                y=float(raw["y"]),
                width=float(raw["width"]),
                height=float(raw["height"]),
                frequency_hz=float(raw["frequency_hz"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid target {index}: {error}") from error
        if not target.name:
            raise ValueError(f"target {index} name must not be empty")
        if target.screen not in {"left", "right"}:
            raise ValueError(
                f"target {target.name} screen must be 'left' or 'right'"
            )
        if target.width <= 0 or target.height <= 0:
            raise ValueError(f"target {target.name} size must be positive")
        if target.frequency_hz <= 0:
            raise ValueError(f"target {target.name} frequency must be positive")
        if target.x < 0 or target.y < 0:
            raise ValueError(f"target {target.name} must be inside the window")
        if target.x + target.width > window_size[0] or target.y + target.height > window_size[1]:
            raise ValueError(f"target {target.name} exceeds the {window_size} window")
        targets.append(target)

    names = [target.name for target in targets]
    if len(set(names)) != len(names):
        raise ValueError("target names must be unique")
    side_frequencies = [(target.screen, target.frequency_hz) for target in targets]
    if len(set(side_frequencies)) != len(side_frequencies):
        raise ValueError("target frequencies must be unique within each screen side")
    regions = {target.screen for target in targets}
    if regions != {"left", "right"}:
        raise ValueError("the target configuration must contain targets in both screen halves")
    return targets


def generate_target_indexes(target_count: int, count: int = 40) -> list[int]:
    if target_count < 1:
        raise ValueError("target_count must be positive")
    if count < 1:
        raise ValueError("count must be positive")
    return [random.randrange(target_count) for _ in range(count)]


def unique_target_frequencies(targets: list[TargetSpec]) -> list[float]:
    return list(dict.fromkeys(target.frequency_hz for target in targets))
