from __future__ import annotations

import sys
from pathlib import Path

import pytest

import src.scripts.make_plots as make_plots_module
from fyp_sim.plotting.generation import PlotGenerationError


def _make_run_dir(base_dir: Path) -> Path:
    run_dir = base_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("test", encoding="utf-8")


def test_main_calls_generate_plots_for_run_with_cli_run_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = _make_run_dir(tmp_path / "cli_path_run")
    seen: list[tuple[Path, bool]] = []

    def fake_generate_plots_for_run(path: Path, overwrite: bool = False) -> list[Path]:
        seen.append((path, overwrite))

        plot_a = run_dir / "plots" / "figure_a_vii_trajectory.png"
        plot_b = run_dir / "plots" / "figure_b_action_distribution.png"
        _touch(plot_a)
        _touch(plot_b)
        return [plot_a, plot_b]

    monkeypatch.setattr(make_plots_module, "generate_plots_for_run", fake_generate_plots_for_run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["make_plots.py", "--run-dir", str(run_dir)],
    )

    make_plots_module.main()
    captured = capsys.readouterr()

    assert seen == [(run_dir, True)]
    assert f"Wrote 2 plot image(s) to: {run_dir / 'plots'}" in captured.out
    assert "- plots/figure_a_vii_trajectory.png" in captured.out
    assert "- plots/figure_b_action_distribution.png" in captured.out


def test_main_prints_relative_paths_for_generated_plots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = _make_run_dir(tmp_path / "relative_paths_run")

    plot_a = run_dir / "plots" / "figure_e_phenotype_vii_trajectories.png"
    plot_b = run_dir / "plots" / "nested" / "figure_g_sup_phenotype_lockin_timeline.png"
    _touch(plot_a)
    _touch(plot_b)

    def fake_generate_plots_for_run(path: Path, overwrite: bool = False) -> list[Path]:
        assert path == run_dir
        assert overwrite is True
        return [plot_a, plot_b]

    monkeypatch.setattr(make_plots_module, "generate_plots_for_run", fake_generate_plots_for_run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["make_plots.py", "--run-dir", str(run_dir)],
    )

    make_plots_module.main()
    captured = capsys.readouterr()

    assert "- plots/figure_e_phenotype_vii_trajectories.png" in captured.out
    assert "- plots/nested/figure_g_sup_phenotype_lockin_timeline.png" in captured.out


def test_main_exits_cleanly_on_plot_generation_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = _make_run_dir(tmp_path / "error_run")

    def fake_generate_plots_for_run(path: Path, overwrite: bool = False) -> list[Path]:
        assert path == run_dir
        assert overwrite is True
        raise PlotGenerationError("Cannot generate plots because `manifest.json` is missing.")

    monkeypatch.setattr(make_plots_module, "generate_plots_for_run", fake_generate_plots_for_run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["make_plots.py", "--run-dir", str(run_dir)],
    )

    with pytest.raises(SystemExit) as exc_info:
        make_plots_module.main()

    assert str(exc_info.value) == "Cannot generate plots because `manifest.json` is missing."
