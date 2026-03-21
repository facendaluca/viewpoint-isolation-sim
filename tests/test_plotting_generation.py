from __future__ import annotations

from pathlib import Path

import pytest

from fyp_sim.plotting.generation import PlotGenerationError, generate_plots_for_run, get_plot_status


def _make_run_dir(
    tmp_path: Path,
    *,
    with_manifest: bool = True,
    with_summary: bool = True,
    with_seed_log: bool = False,
) -> Path:
    run_dir = tmp_path / "outputs" / "runs" / "20260321" / "120000Z_baseline_abc12345"
    run_dir.mkdir(parents=True)

    if with_manifest:
        (run_dir / "manifest.json").write_text('{"mode": "baseline"}', encoding="utf-8")

    if with_summary:
        (run_dir / "summary.csv").write_text("metric,value\nfoo,1\n", encoding="utf-8")

    if with_seed_log:
        seed_dir = run_dir / "seeds" / "s00000"
        seed_dir.mkdir(parents=True)
        (seed_dir / "run_log.csv").write_text(
            "step_id,viewpoint_distance\n0,0.5\n",
            encoding="utf-8",
        )

    return run_dir


def _write_plot(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake-image-bytes")


def test_get_plot_status_returns_none_when_no_plot_images_exist(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path)

    status = get_plot_status(run_dir)

    assert status.label == "None"
    assert list(status.plot_files) == []
    assert status.latest_input_mtime is not None
    assert status.latest_plot_mtime is None


def test_get_plot_status_returns_outdated_when_inputs_are_newer(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path)
    plot_path = run_dir / "plots" / "figure_a_vii_trajectory.png"
    _write_plot(plot_path)

    summary_path = run_dir / "summary.csv"
    current = summary_path.read_text(encoding="utf-8")
    summary_path.write_text(current + "bar,2\n", encoding="utf-8")

    status = get_plot_status(run_dir)

    assert status.label == "Outdated"
    assert list(status.plot_files) == [plot_path]


def test_generate_plots_for_run_raises_for_missing_run_dir(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"

    with pytest.raises(PlotGenerationError, match="Run directory not found"):
        generate_plots_for_run(missing)


def test_generate_plots_for_run_raises_when_manifest_is_missing(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path, with_manifest=False)

    with pytest.raises(PlotGenerationError, match="manifest.json"):
        generate_plots_for_run(run_dir)


def test_generate_plots_for_run_raises_when_summary_and_seed_logs_are_missing(
    tmp_path: Path,
) -> None:
    run_dir = _make_run_dir(tmp_path, with_manifest=True, with_summary=False, with_seed_log=False)

    with pytest.raises(PlotGenerationError, match="summary.csv"):
        generate_plots_for_run(run_dir)


def test_generate_plots_for_run_raises_when_plots_exist_and_overwrite_is_false(
    tmp_path: Path,
) -> None:
    run_dir = _make_run_dir(tmp_path)
    _write_plot(run_dir / "plots" / "figure_a_vii_trajectory.png")

    with pytest.raises(PlotGenerationError, match="already exist"):
        generate_plots_for_run(run_dir, overwrite=False)


def test_generate_plots_for_run_generates_single_agent_plots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = _make_run_dir(tmp_path)
    calls: list[str] = []

    def fake_has_multi_agent_run(_: Path) -> bool:
        return False

    def fake_plot_single_run_figures(target: Path) -> Path | None:
        calls.append("single")
        out_dir = target / "plots"
        _write_plot(out_dir / "figure_a_vii_trajectory.png")
        _write_plot(out_dir / "figure_b_viii_trajectory.png")
        _write_plot(out_dir / "figure_c_lockin_episodes.png")
        return out_dir

    def fake_plot_multi_agent_figures(_: Path) -> Path | None:
        calls.append("multi")
        return None

    def fake_plot_multi_run_variability(target: Path) -> Path | None:
        calls.append("variability")
        out_path = target / "plots" / "figure_d_multi_run_variability.png"
        _write_plot(out_path)
        return out_path

    monkeypatch.setattr(
        "fyp_sim.plotting.generation.has_multi_agent_run",
        fake_has_multi_agent_run,
    )
    monkeypatch.setattr(
        "fyp_sim.plotting.generation.plot_single_run_figures",
        fake_plot_single_run_figures,
    )
    monkeypatch.setattr(
        "fyp_sim.plotting.generation.plot_multi_agent_figures",
        fake_plot_multi_agent_figures,
    )
    monkeypatch.setattr(
        "fyp_sim.plotting.generation.plot_multi_run_variability",
        fake_plot_multi_run_variability,
    )

    plot_files = generate_plots_for_run(run_dir, overwrite=False)

    assert calls == ["single", "variability"]
    assert [path.name for path in plot_files] == [
        "figure_a_vii_trajectory.png",
        "figure_b_viii_trajectory.png",
        "figure_c_lockin_episodes.png",
        "figure_d_multi_run_variability.png",
    ]


def test_generate_plots_for_run_overwrite_replaces_existing_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = _make_run_dir(tmp_path)
    old_plot = run_dir / "plots" / "figure_a_vii_trajectory.png"
    _write_plot(old_plot)

    def fake_has_multi_agent_run(_: Path) -> bool:
        return False

    def fake_plot_single_run_figures(target: Path) -> Path | None:
        out_dir = target / "plots"
        _write_plot(out_dir / "figure_b_action_distribution.png")
        return out_dir

    def fake_plot_multi_run_variability(_: Path) -> Path | None:
        return None

    monkeypatch.setattr(
        "fyp_sim.plotting.generation.has_multi_agent_run",
        fake_has_multi_agent_run,
    )
    monkeypatch.setattr(
        "fyp_sim.plotting.generation.plot_single_run_figures",
        fake_plot_single_run_figures,
    )
    monkeypatch.setattr(
        "fyp_sim.plotting.generation.plot_multi_run_variability",
        fake_plot_multi_run_variability,
    )

    plot_files = generate_plots_for_run(run_dir, overwrite=True)
    assert [path.name for path in plot_files] == ["figure_b_action_distribution.png"]
