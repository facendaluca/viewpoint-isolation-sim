from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd
import seaborn as sns

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

_REQUIRED_SUMMARY_COLS = {
    "agent",
    "seed",
    "mean_vii",
    "mean_watch_time",
    "watch_rate",
    "sample_rate",
    "avoid_rate",
}


def make_compare_plot(
    run_dir: Path, *, out_path: Path | None = None, seed: int | None = None
) -> Path:
    """
    Create a simple comparison plot for a compare run directory produced by run_compare.py.

    Expects:
    - run_dir/summary.csv
    - run_dir/logs/<agent>/run_seed_<seed>.csv

    Outputs:
    - run_dir/compare_plot.png (unless out_path is provided)
    """
    summary_path = run_dir / "summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(f"summary.csv not found at {summary_path}")

    df = pd.read_csv(summary_path)

    missing = _REQUIRED_SUMMARY_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in summary.csv: {summary_path}")

    # Prefer consistent ordering
    agent_order = [a for a in ["heuristic", "llm"] if a in set(df["agent"])]
    if not agent_order:
        agent_order = sorted(df["agent"].unique().tolist())
    df["agent"] = pd.Categorical(df["agent"], categories=agent_order, ordered=True)

    # Choose a seed for the time-series panel
    seeds = sorted(df["seed"].unique().tolist())
    seed_for_series = seeds[0] if seeds else 0
    if seed is not None:
        seed_for_series = int(seed)

    # Load one seed's step logs for time-series VII plot
    ts_frames: list[pd.DataFrame] = []
    for agent in agent_order:
        log_path = run_dir / "logs" / agent / f"run_seed_{seed_for_series}.csv"
        if log_path.exists():
            dlog = pd.read_csv(log_path)
            dlog["agent"] = agent
            ts_frames.append(dlog)

    df_ts = pd.concat(ts_frames, ignore_index=True) if ts_frames else pd.DataFrame()

    # Aggregate across seeds for bar plots
    agg = (
        df.groupby("agent", observed=True)
        .agg(
            mean_vii_mean=("mean_vii", "mean"),
            mean_vii_std=("mean_vii", "std"),
            mean_watch_time_mean=("mean_watch_time", "mean"),
            mean_watch_time_std=("mean_watch_time", "std"),
            watch_rate_mean=("watch_rate", "mean"),
            sample_rate_mean=("sample_rate", "mean"),
            avoid_rate_mean=("avoid_rate", "mean"),
        )
        .reset_index()
    )

    # Single figure (2x2)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    ax_ts, ax_vii, ax_wt, ax_rates = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]

    # VII cumulative mean over time (single seed)
    if not df_ts.empty:
        sns.lineplot(
            data=df_ts,
            x="t",
            y="vii_cum",
            hue="agent",
            hue_order=agent_order,
            ax=ax_ts,
            markers=True,
            dashes=False,
        )
        ax_ts.set_title(f"VII (cumulative mean) over time (seed={seed_for_series})")
        ax_ts.set_xlabel("t")
        ax_ts.set_ylabel("vii_cum")

    else:
        ax_ts.set_title(f"VII (cumulative mean) over time (seed={seed_for_series})")
        ax_ts.text(0.5, 0.5, "No step logs found for time-series panel", ha="center", va="center")
        ax_ts.set_axis_off()

    # Mean VII across seeds (+std)
    sns.barplot(data=agg, x="agent", y="mean_vii_mean", ax=ax_vii, order=agent_order)
    for i, row in agg.iterrows():
        ax_vii.errorbar(
            i,
            row["mean_vii_mean"],
            yerr=row["mean_vii_std"] if pd.notna(row["mean_vii_std"]) else 0.0,
            fmt="None",
            capsize=4,
        )
    ax_vii.set_title("Mean VII across seeds")
    ax_vii.set_xlabel("")
    ax_vii.set_ylabel("mean_vii")

    # Mean watch time across seeds (+std)
    sns.barplot(data=agg, x="agent", y="mean_watch_time_mean", ax=ax_wt, order=agent_order)
    for i, row in agg.iterrows():
        ax_wt.errorbar(
            i,
            row["mean_watch_time_mean"],
            yerr=row["mean_watch_time_std"] if pd.notna(row["mean_watch_time_std"]) else 0.0,
            fmt="None",
            capsize=4,
        )
    ax_wt.set_title("Mean watch-time proxy across seeds")
    ax_wt.set_xlabel("")
    ax_wt.set_ylabel("mean_watch_time (s)")

    # Action rates (mean across seeds)
    rates = agg.melt(
        id_vars=["agent"],
        value_vars=["watch_rate_mean", "sample_rate_mean", "avoid_rate_mean"],
        var_name="action",
        value_name="rate",
    )
    rates["action"] = rates["action"].str.replace("_mean", "", regex=False)
    sns.barplot(data=rates, x="action", y="rate", hue="agent", ax=ax_rates, hue_order=agent_order)
    ax_rates.set_title("Action rates (mean across seeds)")
    ax_rates.set_xlabel("")
    ax_rates.set_ylabel("rate")
    ax_rates.set_ylim(0.0, 1.0)

    fig.tight_layout()

    if out_path is None:
        out_path = run_dir / "compare_plot.png"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return out_path
