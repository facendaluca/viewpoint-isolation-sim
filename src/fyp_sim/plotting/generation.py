from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fyp_sim.plotting.multi_agent_metrics import has_multi_agent_run
from fyp_sim.plotting.multi_agent_plots import plot_multi_agent_figures
from fyp_sim.plotting.multi_run_plots import plot_multi_run_variability
from fyp_sim.plotting.single_run_plots import plot_single_run_figures

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
_PLOT_OUTPUT_SUFFIXES = _IMAGE_SUFFIXES | {".pdf"}


class PlotGenerationError(ValueError):
    """Friendly, UI-safe error for plot generation workflows."""


@dataclass(frozen=True)
class PlotStatus:
    label: str
    plot_files: tuple[Path, ...]
    latest_input_mtime: float | None
    latest_plot_mtime: float | None


def _first_existing_file(run_dir: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        path = run_dir / name
        if path.is_file():
            return path
    return None


def _require_run_dir(run_dir: Path) -> Path:
    if not run_dir.exists():
        raise PlotGenerationError(f"Run directory not found: `{run_dir}`.")
    if not run_dir.is_dir():
        raise PlotGenerationError(f"Selected run path is not a directory: `{run_dir}`.")
    return run_dir


def _find_manifest_file(run_dir: Path) -> Path | None:
    # Keep legacy compatibility for older artefacts.
    return _first_existing_file(run_dir, ("manifest.json", "meta.json"))


def _find_summary_file(run_dir: Path) -> Path | None:
    return _first_existing_file(run_dir, ("summary.csv",))


def _find_seed_run_logs(run_dir: Path) -> list[Path]:
    seeds_dir = run_dir / "seeds"
    if not seeds_dir.is_dir():
        return []

    return sorted(
        (path for path in seeds_dir.glob("*/run_log.csv") if path.is_file()),
        key=lambda path: str(path.relative_to(run_dir)),
    )


def _list_plot_files(run_dir: Path, *, suffixes: set[str]) -> list[Path]:
    plots_dir = run_dir / "plots"
    if not plots_dir.is_dir():
        return []

    return sorted(
        (
            path
            for path in plots_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in suffixes
        ),
        key=lambda path: str(path.relative_to(plots_dir)),
    )


def list_generated_plot_images(run_dir: Path) -> list[Path]:
    """Return plot image files under run_dir/plots in stable relative-path order."""
    return _list_plot_files(run_dir, suffixes=_IMAGE_SUFFIXES)


def _list_plot_outputs(run_dir: Path) -> list[Path]:
    return _list_plot_files(run_dir, suffixes=_PLOT_OUTPUT_SUFFIXES)


def _existing_input_files(run_dir: Path) -> list[Path]:
    files: list[Path] = []

    manifest_path = _find_manifest_file(run_dir)
    if manifest_path is not None:
        files.append(manifest_path)

    summary_path = _find_summary_file(run_dir)
    if summary_path is not None:
        files.append(summary_path)

    files.extend(_find_seed_run_logs(run_dir))
    return files


def _required_input_files(run_dir: Path) -> list[Path]:
    manifest_path = _find_manifest_file(run_dir)
    if manifest_path is None:
        raise PlotGenerationError("Cannot generate plots because `manifest.json` is missing.")

    summary_path = _find_summary_file(run_dir)
    run_logs = _find_seed_run_logs(run_dir)

    if summary_path is None and not run_logs:
        raise PlotGenerationError(
            "Cannot generate plots because neither `summary.csv` nor any "
            "`seeds/*/run_log.csv` files were found."
        )

    files = [manifest_path]
    if summary_path is not None:
        files.append(summary_path)
    files.extend(run_logs)
    return files


def _latest_mtime(paths: list[Path]) -> float | None:
    if not paths:
        return None
    return max(path.stat().st_mtime for path in paths)


def validate_run_for_plot_generation(run_dir: Path) -> None:
    _require_run_dir(run_dir)
    _required_input_files(run_dir)


def get_plot_status(run_dir: Path) -> PlotStatus:
    """
    Return a UI-friendly status label for plots in this run.

    - None: no supported plot images exist
    - Outdated: plot images exist but are older than known inputs
    - Found: plot images exist and are not older than known inputs
    """
    run_dir = _require_run_dir(run_dir)

    plot_files = list_generated_plot_images(run_dir)
    input_files = _existing_input_files(run_dir)

    latest_input_mtime = _latest_mtime(input_files)
    latest_plot_mtime = _latest_mtime(plot_files)

    if not plot_files:
        return PlotStatus(
            label="None",
            plot_files=tuple(),
            latest_input_mtime=latest_input_mtime,
            latest_plot_mtime=None,
        )

    label = "Found"
    if (
        latest_input_mtime is not None
        and latest_plot_mtime is not None
        and latest_plot_mtime < latest_input_mtime
    ):
        label = "Outdated"

    return PlotStatus(
        label=label,
        plot_files=tuple(plot_files),
        latest_input_mtime=latest_input_mtime,
        latest_plot_mtime=latest_plot_mtime,
    )


def _delete_existing_plot_outputs(run_dir: Path) -> None:
    for path in _list_plot_outputs(run_dir):
        path.unlink()


def generate_plots_for_run(run_dir: Path, overwrite: bool = False) -> list[Path]:
    """
    Generate plots for a run directory by calling the plotting pipeline directly.

    Returns generated plot image paths in stable relative-path order.
    """
    run_dir = _require_run_dir(run_dir)
    _required_input_files(run_dir)

    existing_plot_images = list_generated_plot_images(run_dir)
    if existing_plot_images and not overwrite:
        raise PlotGenerationError(
            "Plots already exist for this run. Enable overwrite to regenerate them."
        )

    if overwrite:
        _delete_existing_plot_outputs(run_dir)

    if has_multi_agent_run(run_dir):
        plot_multi_agent_figures(run_dir)
    else:
        plot_single_run_figures(run_dir)

    # Optional figure for multi-seed runs.
    plot_multi_run_variability(run_dir)

    generated_plot_images = list_generated_plot_images(run_dir)
    if not generated_plot_images:
        raise PlotGenerationError(
            "Plot generation finished, but no supported image files were created in `plots/`."
        )

    return generated_plot_images
