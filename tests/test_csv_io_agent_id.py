from __future__ import annotations

from pathlib import Path

from fyp_sim.runners.csv_io import write_run_log_csv
from fyp_sim.simulation.engine import StepLog


def _log(*, t: int, video_id: int, agent_id: str = "") -> StepLog:
    r = StepLog(
        t=t,
        video_id=video_id,
        action="Watch",
        watch_time_s=10,
        interest=0.5,
        vii_t=0.2,
        vii_cum=0.2,
        topic_interest=0.3,
        interest_keys=3,
    )
    # Only used when agent_id=True
    r.agent_id = agent_id
    return r


def test_write_run_log_csv_default_schema_has_no_agent_id(tmp_path: Path) -> None:
    p = tmp_path / "run_log.csv"
    write_run_log_csv(p, [_log(t=0, video_id=1)], include_viewpoint=False)
    header = p.read_text(encoding="utf-8").splitlines()[0]
    assert header.startswith("t,video_id,action"), header
    assert "agent_id" not in header


def test_wrtie_run_log_csv_includes_agent_id_when_enabled(tmp_path: Path) -> None:
    p = tmp_path / "run_log.csv"
    write_run_log_csv(
        p,
        [_log(t=0, video_id=1, agent_id="watcher"), _log(t=1, video_id=2, agent_id="sampler")],
        include_viewpoint=False,
        include_agent_id=True,
    )
    header = p.read_text(encoding="utf-8").splitlines()[0]
    assert header.startswith("agent_id,t,video_id,action"), header
