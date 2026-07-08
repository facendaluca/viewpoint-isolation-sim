from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg", force=True)

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
