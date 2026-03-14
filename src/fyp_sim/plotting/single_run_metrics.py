from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .common import (
    LockInEpisode,
    RunPlotParams,
    detect_lock_in_episodes,
    ensure_dir,
    first_seed_run_log,
    load_run_log_df,
    load_run_plot_params,
    seed_dirs,
)
from .plot_utils import format_run_subtitle

ACTION_ORDER = ("Watch", "Sample", "Avoid")
DEFAULT_ROLLING_WINDOW = 25


@dataclass(frozen=True, slots=True)
class ActionDistributionData:
    proportions: dict[str, float]
    rolling_rates: pd.DataFrame
    window: int


@dataclass(frozen=True, slots=True)
class SingleRunPlotContext:
    run_dir: Path
    out_dir: Path
    params: RunPlotParams
    primary_seed: str
    df: pd.DataFrame
    subtitle: str
    episodes: list[LockInEpisode]
    lockin_summary: dict[str, int]


def compute_action_distribution(
    df: pd.DataFrame,
    *,
    window: int = DEFAULT_ROLLING_WINDOW,
) -> ActionDistributionData:
    action_series = df["user_action"].astype(str).str.title()
    proportions = {action: float(action_series.eq(action).mean()) for action in ACTION_ORDER}

    effective_window = min(window, max(1, len(df)))
    rolling_rates = pd.DataFrame({"step_id": df["step_id"]})

    for action in ACTION_ORDER:
        rolling_rates[action] = (
            action_series.eq(action)
            .astype(float)
            .rolling(
                effective_window,
                min_periods=1,
            )
            .mean()
        )

    return ActionDistributionData(
        proportions=proportions,
        rolling_rates=rolling_rates,
        window=effective_window,
    )


def build_single_run_context(run_dir: Path) -> SingleRunPlotContext:
    out_dir = run_dir / "plots"
    ensure_dir(out_dir)

    params = load_run_plot_params(run_dir)
    primary_run_log = first_seed_run_log(run_dir)
    primary_seed = primary_run_log.parent.name

    df = load_run_log_df(primary_run_log, run_dir=run_dir)
    subtitle = format_run_subtitle(
        seed_label=primary_seed,
        threshold=params.threshold,
        persistence_window=params.persistence_window,
        steps=params.steps,
        rank_alpha=params.rank_alpha,
        drift_alpha=params.drift_alpha,
    )

    episodes, lockin_summary = detect_lock_in_episodes(
        df,
        threshold=params.threshold,
        persistence_window=params.persistence_window,
    )

    return SingleRunPlotContext(
        run_dir=run_dir,
        out_dir=out_dir,
        params=params,
        primary_seed=primary_seed,
        df=df,
        subtitle=subtitle,
        episodes=episodes,
        lockin_summary=lockin_summary,
    )


def build_lockin_summary_rows(run_dir: Path) -> list[dict[str, int | float | str]]:
    params = load_run_plot_params(run_dir)
    rows: list[dict[str, int | float | str]] = []

    for seed_dir in seed_dirs(run_dir):
        run_log_path = seed_dir / "run_log.csv"
        if not run_log_path.exists():
            raise FileNotFoundError(f"Expected run_log.csv at: {run_log_path}")

        df = load_run_log_df(run_log_path, run_dir=run_dir)
        _, summary = detect_lock_in_episodes(
            df,
            threshold=params.threshold,
            persistence_window=params.persistence_window,
        )

        rows.append(
            {
                "seed": seed_dir.name,
                "time_to_lock_in": summary["time_to_lock_in"],
                "n_lockin_episodes": summary["n_lockin_episodes"],
                "total_lockin_steps": summary["total_lockin_steps"],
                "filter_bubble_threshold": params.threshold,
                "persistence_window": params.persistence_window,
            }
        )

    return rows


def write_lockin_summary_csv(run_dir: Path) -> Path:
    rows = build_lockin_summary_rows(run_dir)
    out_path = run_dir / "lockin_summary.csv"
    pd.DataFrame(rows).sort_values("seed").to_csv(out_path, index=False)
    return out_path
