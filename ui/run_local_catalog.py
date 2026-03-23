"""
Static domain catalogue for the Run Locally page.

Owns field specs, bounded parameter model, and the curated preset list.
No Streamlit, no parsing, no validation — pure data.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IntFieldSpec:
    key: str
    label: str
    min_value: int
    max_value: int
    default: int
    step: int
    help_text: str

    @property
    def bounds_text(self) -> str:
        return f"{self.min_value}–{self.max_value}"


@dataclass(frozen=True)
class BoundedParams:
    steps: int
    top_k: int
    seed: int

    def to_dict(self) -> dict[str, int]:
        return {"steps": self.steps, "top_k": self.top_k, "seed": self.seed}


@dataclass(frozen=True)
class Preset:
    id: str
    label: str
    description: str
    scenario: str
    defaults: BoundedParams


FIELD_SPECS: dict[str, IntFieldSpec] = {
    "steps": IntFieldSpec(
        key="steps",
        label="Simulation steps",
        min_value=25,
        max_value=500,
        default=150,
        step=25,
        help_text="Number of simulation steps to execute for one run.",
    ),
    "top_k": IntFieldSpec(
        key="top_k",
        label="Recommendation pool (top_k)",
        min_value=1,
        max_value=10,
        default=5,
        step=1,
        help_text="How many candidate videos are considered at each step.",
    ),
    "seed": IntFieldSpec(
        key="seed",
        label="Random seed",
        min_value=0,
        max_value=99999,
        default=0,
        step=1,
        help_text="Deterministic seed used for the run.",
    ),
}

BOUND_PARAM_KEYS = frozenset(FIELD_SPECS.keys())

PRESETS: tuple[Preset, ...] = (
    Preset(
        id="main_comparison",
        label="Main comparison",
        description="Balanced examiner preset for the main comparison scenario.",
        scenario="experiment_compare",
        defaults=BoundedParams(steps=150, top_k=5, seed=0),
    ),
    Preset(
        id="quick_baseline",
        label="Quick baseline",
        description="Smaller baseline run for quick smoke tests and demonstrations.",
        scenario="experiment_baseline",
        defaults=BoundedParams(steps=50, top_k=2, seed=0),
    ),
    Preset(
        id="drift_enabled",
        label="Drift-enabled baseline",
        description="Baseline scenario with viewpoint drift enabled.",
        scenario="experiment_baseline_drift",
        defaults=BoundedParams(steps=50, top_k=2, seed=0),
    ),
)

PRESET_BY_ID: dict[str, Preset] = {p.id: p for p in PRESETS}
DEFAULT_PRESET_ID = "main_comparison"


def get_preset(preset_id: str) -> Preset:
    return PRESET_BY_ID.get(preset_id, PRESET_BY_ID[DEFAULT_PRESET_ID])


def infer_preset_id(selected_scenario: str) -> str:
    scenario = selected_scenario.strip()
    for preset in PRESETS:
        if preset.scenario == scenario:
            return preset.id
    return DEFAULT_PRESET_ID
