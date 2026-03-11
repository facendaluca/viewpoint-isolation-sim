from __future__ import annotations

from pathlib import Path

import pandas as pd

from .heatmaps import heatmap


def plot_sweep_heatmaps(*, sweep_path: Path, out_dir: Path) -> None:
    df = pd.read_csv(sweep_path)

    if "alpha" in df.columns and "rank_alpha" not in df.columns:
        raise ValueError(
            "Detected legacy 'alpha' column. "
            "Please regenerate outputs with new config schema or rename column manually."
        )

    df["top_k"] = df["top_k"].astype(int)
    df["rank_alpha"] = df["rank_alpha"].astype(float)

    heatmap(
        df,
        value="mean_vii_mean",
        out_path=out_dir / "heatmap_mean_vii.png",
        title="Mean VII across sweep",
    )
    heatmap(
        df,
        value="lock_in_rate_mean",
        out_path=out_dir / "heatmap_lock_in_rate.png",
        title="Lock-in rate across sweep",
    )
