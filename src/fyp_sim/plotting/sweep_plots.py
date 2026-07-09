from __future__ import annotations

from pathlib import Path

import pandas as pd

from .heatmaps import heatmap

SWEEP_AXES = ("top_k", "rank_alpha")

# The two figures the exploration sweep needs: how isolated the user ends up
# on average, and how often runs lock in at all. Keyed by the summary.csv
# column, giving the output filename stem and the figure title.
SWEEP_HEATMAP_SPECS: dict[str, tuple[str, str]] = {
    "mean_vii_mean": (
        "figure_h_sweep_mean_vii",
        "Mean VII across the sweep grid (mean over seeds)",
    ),
    "lock_in_rate_mean": (
        "figure_i_sweep_lockin_rate",
        "Lock-in rate across the sweep grid (mean over seeds)",
    ),
}


def load_sweep_summary(run_dir: Path) -> pd.DataFrame:
    summary_path = run_dir / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Expected summary.csv at: {summary_path}")

    df = pd.read_csv(summary_path)

    missing = [c for c in (*SWEEP_AXES, *SWEEP_HEATMAP_SPECS) if c not in df.columns]
    if missing:
        raise ValueError(
            "summary.csv does not look like a sweep summary. "
            f"run_dir={run_dir}. Missing columns: {sorted(missing)}. "
            f"Found columns: {sorted(df.columns.tolist())}"
        )

    # A heatmap needs exactly one value per grid cell. Duplicates usually mean
    # the sweep varied a third parameter, which these figures cannot show.
    if df.duplicated(subset=list(SWEEP_AXES)).any():
        raise ValueError(
            "summary.csv has more than one row per (top_k, rank_alpha) cell, "
            f"so it cannot be drawn as a single heatmap. run_dir={run_dir}"
        )

    return df


def plot_sweep_heatmaps(run_dir: Path) -> list[Path]:
    """Write the sweep heatmap figures for a sweep run directory.

    Returns the paths of every file written (PNG and PDF per figure).
    """
    df = load_sweep_summary(run_dir)
    out_dir = run_dir / "plots"

    written: list[Path] = []
    for value, (stem, title) in SWEEP_HEATMAP_SPECS.items():
        out_path = out_dir / f"{stem}.png"
        heatmap(df, value=value, out_path=out_path, title=title)
        written.append(out_path)
        written.append(out_path.with_suffix(".pdf"))

    return written
