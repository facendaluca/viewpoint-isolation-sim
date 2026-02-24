from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd
import seaborn as sns

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


def heatmap(
    df: pd.DataFrame,
    *,
    value: str,
    out_path: Path,
    title: str,
) -> None:
    """
    Create and save a heatmap from a sweep-style dataframe.

    Expects:
    - top_k (int)
    - rank_alpha (float)
    - <value> (numeric)
    """
    if "top_k" not in df.columns or "rank_alpha" not in df.columns:
        raise ValueError("df must contain columns: 'top_k and 'rank_alpha'")
    if value not in df.columns:
        raise ValueError(f"df missing required column for heatmap value: {value!r}")

    pivot = df.pivot(index="top_k", columns="rank_alpha", values=value).sort_index()

    fig, ax = plt.subplots()
    sns.heatmap(
        pivot,
        ax=ax,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        cbar=True,
    )
    ax.set_title(title)
    ax.set_xlabel("rank_alpha")
    ax.set_ylabel("top_k")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
