"""Experiment target configuration."""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from gaze import Region


def color_rgb(color: str) -> tuple[float, float, float]:
    """Parse a configured #RRGGBB color into RGB values in [0, 1]."""
    if not isinstance(color, str) or re.fullmatch(r"#[0-9a-fA-F]{6}", color) is None:
        raise ValueError("target color must use #RRGGBB format")
    return tuple(int(color[index:index+2], 16) / 255.0 for index in (1, 3, 5))


@dataclass(frozen=True)
class TargetSpec:
    name: str
    screen: Region
    x: float
    y: float
    width: float
    height: float
    frequency_hz: float
    shape: str = "rectangle"
    color: str = "#000000"
    label: str | None = None
    region_layout: str = "halves"

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
    region_layout = document.get('region_layout', 'three_sectors' if any(
        isinstance(target, dict) and target.get('screen') == 'bottom' for target in raw_targets
    ) else 'halves')
    if region_layout not in {'halves', 'three_sectors'}:
        raise ValueError("region_layout must be 'halves' or 'three_sectors'")

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
                shape=str(raw.get("shape", "rectangle")).strip().lower(),
                color=str(raw.get("color", "#000000")).strip().upper(),
                label=None if raw.get("label") is None else str(raw["label"]),
                region_layout=region_layout,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid target {index}: {error}") from error
        if not target.name:
            raise ValueError(f"target {index} name must not be empty")
        if target.screen not in {"left", "right", "bottom"}:
            raise ValueError(
                f"target {target.name} screen must be 'left', 'right' or 'bottom'"
            )
        if target.screen == 'bottom' and region_layout != 'three_sectors':
            raise ValueError("bottom targets require the three_sectors layout")
        if target.width <= 0 or target.height <= 0:
            raise ValueError(f"target {target.name} size must be positive")
        if target.shape not in {"rectangle", "square", "triangle"}:
            raise ValueError(f"unsupported target shape: {target.shape}")
        color_rgb(target.color)
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
    labels = [target.label for target in targets if target.label is not None]
    if len(set(labels)) != len(labels):
        raise ValueError("target labels must be unique")
    side_frequencies = [(target.screen, target.frequency_hz) for target in targets]
    if len(set(side_frequencies)) != len(side_frequencies):
        raise ValueError("target frequencies must be unique within each screen side")
    return targets


def generate_target_indexes(target_count: int, count: int = 40) -> list[int]:
    if target_count < 1:
        raise ValueError("target_count must be positive")
    if count < 1:
        raise ValueError("count must be positive")
    return [random.randrange(target_count) for _ in range(count)]


def unique_target_frequencies(targets: list[TargetSpec]) -> list[float]:
    return list(dict.fromkeys(target.frequency_hz for target in targets))
