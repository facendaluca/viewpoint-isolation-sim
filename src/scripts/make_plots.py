from __future__ import annotations

from pathlib import Path

import pandas as pd

from fyp_sim.plotting import heatmap


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
