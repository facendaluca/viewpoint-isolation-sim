from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def save_figure_both_formats(fig: plt.Figure, out_path: Path) -> None:
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")


def format_lockin_summary(summary: dict[str, int]) -> str:
    time_to_lock_in = summary["time_to_lock_in"]
    ttl_text = "None" if time_to_lock_in < 0 else str(time_to_lock_in)

    return (
        f"Time to lock-in: {ttl_text}\n"
        f"Lock-in episodes: {summary['n_lockin_episodes']}\n"
        f"Total lock-in steps: {summary['total_lockin_steps']}"
    )


def format_run_subtitle(
    *,
    seed_label: str,
    threshold: float,
    persistence_window: int,
    steps: int | None,
    rank_alpha: float | None,
    drift_alpha: float | None,
) -> str:
    parts = [
        f"seed={seed_label}",
        f"threshold={threshold:.2f}",
        f"persistence_window={persistence_window}",
    ]

    if steps is not None:
        parts.append(f"steps={steps}")
    if rank_alpha is not None:
        parts.append(f"rank_alpha={rank_alpha:.2f}")
    if drift_alpha is not None:
        parts.append(f"drift_alpha={drift_alpha:.2f}")

    return ", ".join(parts)
