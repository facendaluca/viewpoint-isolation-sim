from __future__ import annotations

from pathlib import Path
from typing import Any

from ui.app_state import AppState, available_runs, available_scenarios, get_state, set_state


def test_app_state_roundtrip() -> None:
    s = AppState(
        selected_run_dir="outputs/runs/abc", selected_scenario="experiment_x", params={"a": 1}
    )
    d = s.to_dict()
    s2 = AppState.from_dict(d)
    assert s2 == s


def test_get_state_initialises_mapping() -> None:
    session: dict[str, Any] = {}
    s = get_state(session)
    assert isinstance(s, AppState)

    # Mutate via set_state and read back
    s2 = s.with_selected_scenario("experiment_compare")
    set_state(session, s2)
    s3 = get_state(session)
    assert s3.selected_scenario == "experiment_compare"


def test_available_runs_lists_dirs(tmp_path: Path) -> None:
    base = tmp_path / "runs"
    base.mkdir()

    # New layout: runs/YYYYMMDD/<run_id>/
    date_dir = base / "20260227"
    date_dir.mkdir()
    (date_dir / "b_run").mkdir()
    (date_dir / "a_run").mkdir()
    # mark them as "run dirs" by adding manifest.json
    (date_dir / "b_run" / "manifest.json").write_text("{}", encoding="utf-8")
    (date_dir / "a_run" / "manifest.json").write_text("{}", encoding="utf-8")

    runs = available_runs(base)
    assert runs == [str(date_dir / "a_run"), str(date_dir / "b_run")]


def test_available_scenarios_lists_json_stems(tmp_path: Path) -> None:
    cfg = tmp_path / "configs"
    cfg.mkdir()
    (cfg / "experiment_baseline.json").write_text("{}", encoding="utf-8")
    (cfg / "experiment_compare.json").write_text("{}", encoding="utf-8")
    scenarios = available_scenarios(cfg)
    assert scenarios == ["experiment_baseline", "experiment_compare"]
