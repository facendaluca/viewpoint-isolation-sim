from __future__ import annotations

from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.figure import Figure

DEFAULT_DPI = 300
DEFAULT_TEXTBOX_STYLE = {
    "boxstyle": "round, pad=0.35",
    "facecolor": "white",
    "alpha": 0.9,
}


def save_figure_both_formats(fig: Figure, out_path: Path) -> None:
    fig.savefig(out_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")


def add_figure_subtitle(
    fig: Figure,
    text: str,
    *,
    y: float,
    fontsize: int = 10,
) -> None:
    fig.text(0.5, y, text, ha="center", va="center", fontsize=fontsize)


def add_annotation_box(
    ax: Axes,
    text: str,
    *,
    x: float = 0.02,
    y: float = 0.98,
    va: str = "top",
    fontsize: int = 10,
) -> None:
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        va=va,
        ha="left",
        fontsize=fontsize,
        bbox=DEFAULT_TEXTBOX_STYLE,
    )


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


def format_multi_run_subtitle(
    *,
    n_seeds: int,
    threshold: float,
    steps: int | None,
    rank_alpha: float | None,
    drift_alpha: float | None,
) -> str:
    parts = [
        f"n_seeds={n_seeds}",
        f"threshold={threshold:.2f}",
    ]

    if steps is not None:
        parts.append(f"steps={steps}")
    if rank_alpha is not None:
        parts.append(f"rank_alpha={rank_alpha:.2f}")
    if drift_alpha is not None:
        parts.append(f"drift_alpha={drift_alpha:.2f}")

    return ", ".join(parts)
