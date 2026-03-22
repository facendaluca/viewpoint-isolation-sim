from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BoundedParams:
    """The explicit, safe parameters examiners can edit via UI widgets."""

    steps: int = 150
    top_k: int = 5
    seed: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BoundedParams:
        """Safely extract known keys from a raw dict, falling back to defaults."""
        return cls(
            steps=int(data.get("steps", 150)),
            top_k=int(data.get("top_k", 5)),
            seed=int(data.get("seed", 0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "top_k": self.top_k,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class Preset:
    name: str
    scenario: str
    params: BoundedParams


PRESETS: list[Preset] = [
    Preset("Quick Test", "experiment_baseline", BoundedParams(steps=50, top_k=3, seed=42)),
    Preset("Full Baseline", "experiment_baseline", BoundedParams(steps=150, top_k=5, seed=0)),
    Preset(
        "High Variance Exploratory",
        "experiment_baseline",
        BoundedParams(steps=150, top_k=10, seed=123),
    ),
]


def get_preset_by_name(name: str) -> Preset | None:
    for p in PRESETS:
        if p.name == name:
            return p
    return None


def parse_advanced_json(json_str: str) -> dict[str, Any]:
    """Parse JSON overrides, ensuring it resolves to a dict. Raises ValueError on failure."""
    if not json_str.strip():
        return {}
    try:
        loaded = json.loads(json_str)
        if not isinstance(loaded, dict):
            raise ValueError("Advanced JSON overrides must be a JSON object (dictionary).")
        return loaded
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON syntax: {e}") from e


def build_final_overrides(form_params: BoundedParams, advanced_json: str) -> dict[str, Any]:
    """
    Merge widget parameters with valid advanced JSON overrides.
    Advanced JSON takes precedence if keys overlap, but UI typically avoids overlap.
    Raises ValueError if JSON is invalid.
    """
    base = form_params.to_dict()
    overrides = parse_advanced_json(advanced_json)
    base.update(overrides)
    return base
