from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt

from .common import ensure_dir, load_run_plot_params, seed_dirs
from .multi_run_metrics import build_multi_run_vii_summary, write_multi_run_summary_csv
from .plot_utils import _format_multi_run_subtitle, save_figure_both_formats


def plot_multi_run_variability(run_dir: Path) -> Path | None:
    run_seed_dirs = seed_dirs(run_dir)
    if len(run_seed_dirs) < 2:
        return None

    out_dir = run_dir / "plots"
    ensure_dir(out_dir)

    params = load_run_plot_params(run_dir)
    summary = build_multi_run_vii_summary(run_dir)

    subtitle = _format_multi_run_subtitle(
        n_seeds=len(run_seed_dirs),
        threshold=params.threshold,
        steps=params.steps,
        rank_alpha=params.rank_alpha,
        drift_alpha=params.drift_alpha,
    )

    fig, ax = plt.subplots(figsize=(10.5, 5.4))

    ax.plot(
        summary["step_id"],
        summary["vii_mean"],
        linewidth=1.8,
        label="Mean cohort viewpoint distance",
    )
    ax.fill_between(
        summary["step_id"],
        summary["vii_ci_lower"],
        summary["vii_ci_upper"],
        alpha=0.18,
        label="95% CI",
    )
    ax.axhline(
        params.threshold,
        linestyle="--",
        linewidth=1.2,
        label=f"Bubble threshold ({params.threshold:.2f})",
    )

    ax.set_xlabel("Step ID")
    ax.set_ylabel("Viewpoint distance (VII_t)")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Figure D - Multi-run variability in stance distance", pad=18)
    fig.text(0.5, 0.94, subtitle, ha="center", va="center", fontsize=10)
    ax.legend(loc="upper right")

    fig.tight_layout(rect=[0, 0, 1, 0.92])

    out_path = out_dir / "figure_d_multi_run_variability.png"
    save_figure_both_formats(fig, out_path)
    plt.close(fig)

    write_multi_run_summary_csv(run_dir)
    return out_path
