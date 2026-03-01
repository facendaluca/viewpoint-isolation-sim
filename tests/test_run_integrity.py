from __future__ import annotations

from pathlib import Path

from ui.run_integrity import check_run_outputs


def test_check_run_outputs_missing_run_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "nope"
    issues = check_run_outputs(run_dir)
    assert len(issues) == 1
    assert issues[0].kind == "missing"
    assert issues[0].path == run_dir


def test_check_run_outputs_detects_missing_files(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    # Only seeds directory exists, but no summary.csv and no seed dirs
    (run_dir / "seeds").mkdir()

    issues = check_run_outputs(run_dir)
    kinds = {i.kind for i in issues}
    paths = {i.path.name for i in issues}

    assert "missing" in kinds  # summary.csv missing
    assert "summary.csv" in paths
    assert any("No seed subdirectories" in i.message for i in issues)


def test_check_run_outputs_happy_path(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    # Non-empty summary.csv
    (run_dir / "summary.csv").write_text("col\n1\n", encoding="utf-8")

    seeds = run_dir / "seeds"
    seeds.mkdir()
    sdir = seeds / "s00042"
    sdir.mkdir()

    # Non-empty run_log.csv
    (sdir / "run_log.csv").write_text("t,video_id\n0,1\n", encoding="utf-8")

    issues = check_run_outputs(run_dir)
    assert issues == []
