from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fyp_sim.plotting.sweep_plots import (
    SWEEP_HEATMAP_SPECS,
    load_sweep_summary,
    plot_sweep_heatmaps,
)


def _write_sweep_summary(run_dir: Path, df: pd.DataFrame) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(run_dir / "summary.csv", index=False)


def _grid_summary() -> pd.DataFrame:
    # Minimal 2x2 sweep grid with the metric columns the heatmaps read.
    return pd.DataFrame(
        {
            "top_k": [1, 1, 5, 5],
            "rank_alpha": [0.0, 0.5, 0.0, 0.5],
            "mean_vii_mean": [0.30, 0.42, 0.25, 0.38],
            "lock_in_rate_mean": [0.0, 0.6, 0.0, 0.4],
        }
    )


def test_plot_sweep_heatmaps_writes_png_and_pdf(tmp_path: Path) -> None:
    run_dir = tmp_path / "sweep_run"
    _write_sweep_summary(run_dir, _grid_summary())

    written = plot_sweep_heatmaps(run_dir)

    assert len(written) == 2 * len(SWEEP_HEATMAP_SPECS)
    for path in written:
        assert path.exists()
        assert path.stat().st_size > 0

    names = sorted(p.name for p in written)
    assert names == [
        "figure_h_sweep_mean_vii.pdf",
        "figure_h_sweep_mean_vii.png",
        "figure_i_sweep_lockin_rate.pdf",
        "figure_i_sweep_lockin_rate.png",
    ]


def test_missing_summary_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="summary.csv"):
        plot_sweep_heatmaps(tmp_path / "no_such_run")


def test_non_sweep_summary_raises_with_missing_columns(tmp_path: Path) -> None:
    run_dir = tmp_path / "batch_run"
    # A batch-style summary has per-seed rows and no rank_alpha grid.
    df = pd.DataFrame({"seed": [0, 1], "mean_vii": [0.3, 0.4]})
    _write_sweep_summary(run_dir, df)

    with pytest.raises(ValueError, match="does not look like a sweep summary"):
        load_sweep_summary(run_dir)


def test_duplicate_grid_cells_raise(tmp_path: Path) -> None:
    run_dir = tmp_path / "dup_run"
    df = _grid_summary()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    _write_sweep_summary(run_dir, df)

    with pytest.raises(ValueError, match="more than one row per"):
        load_sweep_summary(run_dir)
