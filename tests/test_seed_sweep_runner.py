from __future__ import annotations

import csv
import json
from pathlib import Path

from fyp_sim.runners.seed_sweep import run_seed_sweep


def test_run_seed_sweep_writes_logs_and_summary(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = repo_root / "configs" / "experiment_baseline.json"

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert isinstance(cfg, dict)

    # Make test fast + deterministic
    cfg["scenario"] = "experiment_baseline"
    cfg["steps"] = 5
    cfg["seeds"] = [42]
    cfg["policy"] = {"mode": "heuristic"}

    run_dir = run_seed_sweep(cfg, cfg_path=cfg_path, outputs_root=tmp_path)

    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "config_resolved.json").exists()
    assert (run_dir / "summary.csv").exists()
    assert (run_dir / "seeds" / "s00042" / "run_log.csv").exists()

    with (run_dir / "seeds" / "s00042" / "run_log.csv").open("r", newline="") as f:
        header = next(csv.reader(f))
    assert header[:8] == [
        "t",
        "video_id",
        "action",
        "watch_time_s",
        "interest",
        "topic_interest",
        "vii_t",
        "vii_cum",
    ]


def test_run_seed_sweep_multi_agent_adds_agent_id_column(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = repo_root / "configs" / "experiment_baseline.json"

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert isinstance(cfg, dict)

    cfg["scenario"] = "experiment_baseline"
    cfg["steps"] = 3
    cfg["seeds"] = [7]
    cfg["policy"] = {"mode": "heuristic"}

    # Cohort mode (2 agents)
    cfg["n_agents"] = 2
    cfg["agents"] = [
        {
            "agent_id": "watcher",
            "phenotype": "watcher",
            "viewpoint_score": 0.25,
            "sentiment_threshold": 0.0,
            "interest_vector": cfg["user"].get("interest_vector", {}),
        },
        {
            "agent_id": "sampler",
            "phenotype": "sampler",
            "viewpoint_score": 0.75,
            "sentiment_threshold": 0.0,
            "interest_vector": cfg["user"].get("interest_vector", {}),
        },
    ]

    run_dir = run_seed_sweep(cfg, cfg_path=cfg_path, outputs_root=tmp_path)

    seed_log = run_dir / "seeds" / "s00007" / "run_log.csv"
    assert seed_log.exists()

    lines = seed_log.read_text(encoding="utf-8").splitlines()
    assert lines, "run_log.csv is empty"
    assert lines[0].startswith("agent_id,t,video_id,action"), lines[0]

    # Ensure at least two different agent_id values appear in the body
    agent_ids = {ln.split(",")[0] for ln in lines[1:] if ln.strip()}
    assert {"watcher", "sampler"}.issubset(agent_ids)
