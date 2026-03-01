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
