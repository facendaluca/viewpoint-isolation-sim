from __future__ import annotations

import sys
from pathlib import Path

import src.scripts.make_plots as make_plots_module


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("test", encoding="utf-8")


def _make_run_dir(base_dir: Path) -> Path:
    run_dir = base_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def test_main_routes_single_agent_run_and_skips_multi_agent_outputs(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    run_dir = _make_run_dir(tmp_path / "single_agent_run")
    plots_dir = run_dir / "plots"

    def fake_has_multi_agent_run(path: Path) -> bool:
        assert path == run_dir
        return False

    def fake_plot_single_run_figures(path: Path) -> Path:
        assert path == run_dir

        _touch(plots_dir / "figure_a_vii_trajectory.png")
        _touch(plots_dir / "figure_a_vii_trajectory.pdf")
        _touch(plots_dir / "figure_b_action_distribution.png")
        _touch(plots_dir / "figure_b_action_distribution.pdf")
        _touch(plots_dir / "figure_c_lockin_episodes.png")
        _touch(plots_dir / "figure_c_lockin_episodes.pdf")
        _touch(run_dir / "lockin_summary.csv")

        return plots_dir

    def fake_plot_multi_agent_figures(path: Path) -> Path | None:
        raise AssertionError("Multi-agent plotter should not be called for a single-agent run.")

    def fake_plot_multi_run_variability(path: Path) -> Path | None:
        assert path == run_dir
        return None

    monkeypatch.setattr(make_plots_module, "has_multi_agent_run", fake_has_multi_agent_run)
    monkeypatch.setattr(make_plots_module, "plot_single_run_figures", fake_plot_single_run_figures)
    monkeypatch.setattr(
        make_plots_module, "plot_multi_agent_figures", fake_plot_multi_agent_figures
    )
    monkeypatch.setattr(
        make_plots_module,
        "plot_multi_run_variability",
        fake_plot_multi_run_variability,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["make_plots.py", "--run-dir", str(run_dir)],
    )

    make_plots_module.main()
    captured = capsys.readouterr()

    assert (plots_dir / "figure_a_vii_trajectory.png").exists()
    assert (plots_dir / "figure_a_vii_trajectory.pdf").exists()
    assert (plots_dir / "figure_b_action_distribution.png").exists()
    assert (plots_dir / "figure_b_action_distribution.pdf").exists()
    assert (plots_dir / "figure_c_lockin_episodes.png").exists()
    assert (plots_dir / "figure_c_lockin_episodes.pdf").exists()
    assert (run_dir / "lockin_summary.csv").exists()

    assert not (plots_dir / "figure_e_phenotype_vii_trajectories.png").exists()
    assert not (plots_dir / "figure_f_phenotype_action_dynamics.png").exists()
    assert not (plots_dir / "figure_g_phenotype_lockin_outcomes.png").exists()
    assert not (plots_dir / "figure_g_sup_phenotype_lockin_timeline.png").exists()
    assert not (run_dir / "phenotype_lockin_summary.csv").exists()
    assert not (run_dir / "multi_run_vii_summary.csv").exists()

    assert "Wrote single-agent plots to:" in captured.out
    assert "Wrote lock-in summary to:" in captured.out
    assert "Wrote multi-agent phenotype figures to:" not in captured.out
    assert "Wrote multi-run variability figure to:" not in captured.out


def test_main_routes_multi_agent_run_and_writes_multi_run_outputs(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    run_dir = _make_run_dir(tmp_path / "multi_agent_run")
    plots_dir = run_dir / "plots"

    def fake_has_multi_agent_run(path: Path) -> bool:
        assert path == run_dir
        return True

    def fake_plot_single_run_figures(path: Path) -> Path:
        raise AssertionError("Single-agent plotter should not be called for a multi-agent run.")

    def fake_plot_multi_agent_figures(path: Path) -> Path:
        assert path == run_dir

        _touch(plots_dir / "figure_e_phenotype_vii_trajectories.png")
        _touch(plots_dir / "figure_e_phenotype_vii_trajectories.pdf")
        _touch(plots_dir / "figure_f_phenotype_action_dynamics.png")
        _touch(plots_dir / "figure_f_phenotype_action_dynamics.pdf")
        _touch(plots_dir / "figure_g_phenotype_lockin_outcomes.png")
        _touch(plots_dir / "figure_g_phenotype_lockin_outcomes.pdf")
        _touch(plots_dir / "figure_g_sup_phenotype_lockin_timeline.png")
        _touch(plots_dir / "figure_g_sup_phenotype_lockin_timeline.pdf")
        _touch(run_dir / "phenotype_lockin_summary.csv")

        return plots_dir

    def fake_plot_multi_run_variability(path: Path) -> Path:
        assert path == run_dir

        out_path = plots_dir / "figure_d_multi_run_vii_variability.png"
        _touch(out_path)
        _touch(plots_dir / "figure_d_multi_run_vii_variability.pdf")
        _touch(run_dir / "multi_run_vii_summary.csv")
        return out_path

    monkeypatch.setattr(make_plots_module, "has_multi_agent_run", fake_has_multi_agent_run)
    monkeypatch.setattr(make_plots_module, "plot_single_run_figures", fake_plot_single_run_figures)
    monkeypatch.setattr(
        make_plots_module, "plot_multi_agent_figures", fake_plot_multi_agent_figures
    )
    monkeypatch.setattr(
        make_plots_module,
        "plot_multi_run_variability",
        fake_plot_multi_run_variability,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["make_plots.py", "--run-dir", str(run_dir)],
    )

    make_plots_module.main()
    captured = capsys.readouterr()

    assert (plots_dir / "figure_e_phenotype_vii_trajectories.png").exists()
    assert (plots_dir / "figure_e_phenotype_vii_trajectories.pdf").exists()
    assert (plots_dir / "figure_f_phenotype_action_dynamics.png").exists()
    assert (plots_dir / "figure_f_phenotype_action_dynamics.pdf").exists()
    assert (plots_dir / "figure_g_phenotype_lockin_outcomes.png").exists()
    assert (plots_dir / "figure_g_phenotype_lockin_outcomes.pdf").exists()
    assert (plots_dir / "figure_g_sup_phenotype_lockin_timeline.png").exists()
    assert (plots_dir / "figure_g_sup_phenotype_lockin_timeline.pdf").exists()
    assert (run_dir / "phenotype_lockin_summary.csv").exists()

    assert (plots_dir / "figure_d_multi_run_vii_variability.png").exists()
    assert (plots_dir / "figure_d_multi_run_vii_variability.pdf").exists()
    assert (run_dir / "multi_run_vii_summary.csv").exists()

    assert not (plots_dir / "figure_a_vii_trajectory.png").exists()
    assert not (plots_dir / "figure_b_action_distribution.png").exists()
    assert not (plots_dir / "figure_c_lockin_episodes.png").exists()
    assert not (run_dir / "lockin_summary.csv").exists()

    assert "Wrote multi-agent phenotype figures to:" in captured.out
    assert "Wrote phenotype lock-in summary to:" in captured.out
    assert "Wrote multi-run variability figure to:" in captured.out
    assert "Wrote multi-run summary to:" in captured.out
    assert "Wrote single-agent plots to:" not in captured.out


def test_main_passes_cli_run_dir_to_all_helpers(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    run_dir = _make_run_dir(tmp_path / "cli_path_run")
    seen_paths: list[Path] = []

    def fake_has_multi_agent_run(path: Path) -> bool:
        seen_paths.append(path)
        return False

    def fake_plot_single_run_figures(path: Path) -> Path:
        seen_paths.append(path)
        plots_dir = path / "plots"
        _touch(plots_dir / "figure_a_vii_trajectory.png")
        _touch(path / "lockin_summary.csv")
        return plots_dir

    def fake_plot_multi_run_variability(path: Path) -> Path | None:
        seen_paths.append(path)
        return None

    monkeypatch.setattr(make_plots_module, "has_multi_agent_run", fake_has_multi_agent_run)
    monkeypatch.setattr(make_plots_module, "plot_single_run_figures", fake_plot_single_run_figures)
    monkeypatch.setattr(
        make_plots_module,
        "plot_multi_agent_figures",
        lambda path: (_ for _ in ()).throw(
            AssertionError("Multi-agent plotter should not be called in this test.")
        ),
    )
    monkeypatch.setattr(
        make_plots_module,
        "plot_multi_run_variability",
        fake_plot_multi_run_variability,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["make_plots.py", "--run-dir", str(run_dir)],
    )

    make_plots_module.main()
    capsys.readouterr()

    assert seen_paths == [run_dir, run_dir, run_dir]
