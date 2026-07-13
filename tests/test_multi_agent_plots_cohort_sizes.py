from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg", force=True)

from fyp_sim.plotting.multi_agent_metrics import cohort_group_label
from fyp_sim.plotting.multi_agent_plots import plot_phenotype_lockin_outcomes


def _lockin_outcomes_frame(phenotypes: list[str]) -> pd.DataFrame:
    rows = []
    for i, phenotype in enumerate(phenotypes):
        for seed in ("s00000", "s00001"):
            rows.append(
                {
                    "seed": seed,
                    "phenotype": phenotype,
                    "time_to_lock_in": 10 + i,
                    "n_lockin_episodes": 1 + i,
                    "total_lockin_steps": 20 + i,
                    "locked_in": True,
                    "time_to_lock_in_plot": float(10 + i),
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    "phenotypes",
    [
        ["strict", "lenient"],  # E6-style two-agent cohort (regression: IndexError)
        ["watcher", "sampler", "avoider"],  # E2-style three-phenotype cohort
        ["watcher", "sampler", "avoider", "extra"],  # larger cohort
    ],
)
def test_lockin_outcomes_plot_handles_any_cohort_size(tmp_path: Path, phenotypes: list[str]):
    out_path = tmp_path / "figure_g_phenotype_lockin_outcomes.png"

    plot_phenotype_lockin_outcomes(
        lockin_outcomes=_lockin_outcomes_frame(phenotypes),
        out_path=out_path,
        subtitle="test subtitle",
    )

    assert out_path.exists()


def _write_resolved_config(tmp_path: Path, agents: list[dict]) -> Path:
    (tmp_path / "config_resolved.json").write_text(json.dumps({"agents": agents}))
    return tmp_path


@pytest.mark.parametrize(
    ("agents", "expected"),
    [
        # E2-style cohort: the arms really are phenotypes.
        (
            [
                {"agent_id": "watcher", "phenotype": "watcher", "sentiment_threshold": -0.2},
                {"agent_id": "sampler", "phenotype": "sampler", "sentiment_threshold": -0.2},
                {"agent_id": "avoider", "phenotype": "avoider", "sentiment_threshold": -0.2},
            ],
            "Phenotype",
        ),
        # E6-style cohort: one phenotype, arms differ only by sentiment threshold.
        (
            [
                {"agent_id": "strict", "phenotype": "watcher", "sentiment_threshold": 0.5},
                {"agent_id": "baseline", "phenotype": "watcher", "sentiment_threshold": -0.2},
                {"agent_id": "lenient", "phenotype": "watcher", "sentiment_threshold": -1.0},
            ],
            "Sentiment-threshold arm",
        ),
        # Same phenotype and threshold everywhere: fall back to a neutral noun.
        (
            [
                {"agent_id": "a", "phenotype": "watcher", "sentiment_threshold": -0.2},
                {"agent_id": "b", "phenotype": "watcher", "sentiment_threshold": -0.2},
            ],
            "Cohort arm",
        ),
    ],
)
def test_cohort_group_label_names_what_actually_varies(
    tmp_path: Path, agents: list[dict], expected: str
):
    run_dir = _write_resolved_config(tmp_path, agents)
    assert cohort_group_label(run_dir) == expected


def test_cohort_group_label_defaults_to_phenotype_without_config(tmp_path: Path):
    assert cohort_group_label(tmp_path) == "Phenotype"
