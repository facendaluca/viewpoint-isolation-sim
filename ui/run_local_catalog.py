"""
Static domain catalogue for the Run Locally page.
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
        id="E1_baseline_single_watcher",
        label="E1 - baseline single watcher",
        description=(
            "Single-agent baseline condition for checking stance-distance ",
            "convergence and lock-in under heuristic policy.",
        ),
        scenario="E1_baseline_single_watcher",
        defaults=BoundedParams(steps=150, top_k=5, seed=0),
    ),
    Preset(
        id="E2_baseline_multi_phenotype_cohort",
        label="E2 - Baseline multi-phenotype cohort",
        description=(
            "Cohort condition comparing watcher, sampler, and avoider "
            "phenotypes under the same recommendation environment."
        ),
        scenario="E2_baseline_multi_phenotype_cohort",
        defaults=BoundedParams(steps=200, top_k=5, seed=0),
    ),
    Preset(
        id="E3_explore_low",
        label="E3 - Low exploration",
        description=(
            "Low-curiosity condition for testing whether reduced exploration "
            "accelerates narrowing and lock-in."
        ),
        scenario="E3_explore_low",
        defaults=BoundedParams(steps=200, top_k=5, seed=0),
    ),
    Preset(
        id="E4_explore_base",
        label="E4 - Base exploration",
        description=(
            "Reference exploration setting used as the midpoint comparison "
            "for the exploration sweep."
        ),
        scenario="E4_explore_base",
        defaults=BoundedParams(steps=200, top_k=5, seed=0),
    ),
    Preset(
        id="E5_explore_high",
        label="E5 - High exploration",
        description=(
            "Higher-curiosity condition for testing whether stronger exploration "
            "delays or weakens lock-in."
        ),
        scenario="E5_explore_high",
        defaults=BoundedParams(steps=200, top_k=5, seed=0),
    ),
    Preset(
        id="E6_sentiment_strict_vs_lenient",
        label="E6 - Sentiment strict vs lenient",
        description=(
            "Two-agent sentiment tolerance comparison for testing whether "
            "stricter negative-content filtering changes exposure dynamics."
        ),
        scenario="E6_sentiment_strict_vs_lenient",
        defaults=BoundedParams(steps=200, top_k=5, seed=0),
    ),
)

PRESET_BY_ID: dict[str, Preset] = {preset.id: preset for preset in PRESETS}
DEFAULT_PRESET_ID = "E1_baseline_single_watcher"


def get_preset(preset_id: str) -> Preset:
    return PRESET_BY_ID.get(preset_id, PRESET_BY_ID[DEFAULT_PRESET_ID])


def infer_preset_id(selected_scenario: str) -> str:
    scenario = selected_scenario.strip()
    for preset in PRESETS:
        if preset.scenario == scenario:
            return preset.id
    return DEFAULT_PRESET_ID
