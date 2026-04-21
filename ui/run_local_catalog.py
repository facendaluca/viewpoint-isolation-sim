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
class FloatFieldSpec:
    key: str
    label: str
    min_value: float
    max_value: float
    default: float
    step: float
    help_text: str
    format_str: str = "%.2f"

    @property
    def bounds_text(self) -> str:
        return f"{self.min_value:.2f}–{self.max_value:.2f}"


@dataclass(frozen=True)
class BoundedParams:
    steps: int
    top_k: int
    rank_alpha: float
    drift_alpha: float
    lock_in_threshold: float
    persistence_window: int

    def to_dict(self) -> dict[str, int | float]:
        return {
            "steps": self.steps,
            "top_k": self.top_k,
            "rank_alpha": self.rank_alpha,
            "drift_alpha": self.drift_alpha,
            "lock_in_threshold": self.lock_in_threshold,
            "persistence_window": self.persistence_window,
        }


@dataclass(frozen=True)
class Preset:
    id: str
    label: str
    description: str
    scenario: str
    defaults: BoundedParams


INT_FIELD_SPECS: dict[str, IntFieldSpec] = {
    "steps": IntFieldSpec(
        key="steps",
        label="Simulation steps",
        min_value=25,
        max_value=500,
        default=200,
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
    "persistence_window": IntFieldSpec(
        key="persistence_window",
        label="Persistence window",
        min_value=1,
        max_value=50,
        default=10,
        step=1,
        help_text="Minimum consecutive steps required before a lock-in episode is counted.",
    ),
}

FLOAT_FIELD_SPECS: dict[str, FloatFieldSpec] = {
    "rank_alpha": FloatFieldSpec(
        key="rank_alpha",
        label="Ranking alpha",
        min_value=0.0,
        max_value=1.0,
        default=0.30,
        step=0.05,
        help_text="Exposure weighting between interest and engagement during ranking.",
    ),
    "drift_alpha": FloatFieldSpec(
        key="drift_alpha",
        label="Viewpoint drift alpha",
        min_value=0.0,
        max_value=0.20,
        default=0.02,
        step=0.01,
        help_text="Strength of viewpoint drift after interactions when drift is enabled.",
    ),
    "lock_in_threshold": FloatFieldSpec(
        key="lock_in_threshold",
        label="Lock-in threshold",
        min_value=0.0,
        max_value=1.0,
        default=0.20,
        step=0.05,
        help_text="Viewpoint-distance threshold used to detect operational lock-in.",
    ),
}

BOUND_PARAM_KEYS = frozenset((*INT_FIELD_SPECS.keys(), *FLOAT_FIELD_SPECS.keys()))

PRESETS: tuple[Preset, ...] = (
    Preset(
        id="E1_baseline_single_watcher",
        label="E1 - baseline single watcher",
        description=(
            "Single-agent baseline condition for checking stance-distance "
            "convergence and lock-in under heuristic policy."
        ),
        scenario="E1_baseline_single_watcher",
        defaults=BoundedParams(
            steps=200,
            top_k=5,
            rank_alpha=0.30,
            drift_alpha=0.02,
            lock_in_threshold=0.20,
            persistence_window=10,
        ),
    ),
    Preset(
        id="E2_baseline_multi_phenotype_cohort",
        label="E2 - Baseline multi-phenotype cohort",
        description=(
            "Cohort condition comparing watcher, sampler, and avoider "
            "phenotypes under the same recommendation environment."
        ),
        scenario="E2_baseline_multi_phenotype_cohort",
        defaults=BoundedParams(
            steps=200,
            top_k=5,
            rank_alpha=0.30,
            drift_alpha=0.02,
            lock_in_threshold=0.20,
            persistence_window=10,
        ),
    ),
    Preset(
        id="E3_explore_low",
        label="E3 - Low exploration",
        description=(
            "Low-curiosity condition for testing whether reduced exploration "
            "accelerates narrowing and lock-in."
        ),
        scenario="E3_explore_low",
        defaults=BoundedParams(
            steps=200,
            top_k=5,
            rank_alpha=0.30,
            drift_alpha=0.02,
            lock_in_threshold=0.20,
            persistence_window=10,
        ),
    ),
    Preset(
        id="E4_explore_base",
        label="E4 - Base exploration",
        description=(
            "Reference exploration setting used as the midpoint comparison "
            "for the exploration sweep."
        ),
        scenario="E4_explore_base",
        defaults=BoundedParams(
            steps=200,
            top_k=5,
            rank_alpha=0.30,
            drift_alpha=0.02,
            lock_in_threshold=0.20,
            persistence_window=10,
        ),
    ),
    Preset(
        id="E5_explore_high",
        label="E5 - High exploration",
        description=(
            "Higher-curiosity condition for testing whether stronger exploration "
            "delays or weakens lock-in."
        ),
        scenario="E5_explore_high",
        defaults=BoundedParams(
            steps=200,
            top_k=5,
            rank_alpha=0.30,
            drift_alpha=0.02,
            lock_in_threshold=0.20,
            persistence_window=10,
        ),
    ),
    Preset(
        id="E6_sentiment_strict_vs_lenient",
        label="E6 - Sentiment strict vs lenient",
        description=(
            "Two-agent sentiment tolerance comparison for testing whether "
            "stricter negative-content filtering changes exposure dynamics."
        ),
        scenario="E6_sentiment_strict_vs_lenient",
        defaults=BoundedParams(
            steps=200,
            top_k=5,
            rank_alpha=0.30,
            drift_alpha=0.02,
            lock_in_threshold=0.20,
            persistence_window=10,
        ),
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
