from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def heatmap(
    df: pd.DataFrame,
    *,
    value: str,
    out_path: Path,
    title: str,
) -> None:
    pivot = df.pivot(index="top_k", columns="alpha", values=value).sort_index()
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
    ax.set_xlabel("alpha")
    ax.set_ylabel("top_k")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    sweep_path = Path("results/sweep_summary.csv")
    out_dir = Path("outputs/plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(sweep_path)

    # Ensure numeric types
    df["top_k"] = df["top_k"].astype(int)
    df["alpha"] = df["alpha"].astype(float)

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

    print(f"Wrote plots to: {out_dir}")


if __name__ == "__main__":
    main()
