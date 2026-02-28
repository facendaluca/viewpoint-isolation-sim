from __future__ import annotations

import json
from pathlib import Path

from fyp_sim.examiner_dashboard.backend import run_heuristic


def test_run_heuristic_creates_run_dir_and_writes_files(tmp_path: Path) -> None:
    cfg = {
        "scenario": "experiment_baseline",
        "steps": 150,
        "top_k": 5,
        "seed": 42,
    }

    run_dir = run_heuristic(cfg, output_root=tmp_path)

    assert run_dir.exists()
    # Structure: <output_root>/<YYYYMMDD>/<run_id>/
    assert run_dir.parent.parent == tmp_path
    assert len(run_dir.parent.name) == 8  # YYYYMMDD

    assert (run_dir / "config_resolved.json").exists()
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "summary.csv").exists()
    assert (run_dir / "plots").is_dir()
    assert (run_dir / "seeds").is_dir()

    seed_dir = run_dir / "seeds" / "s00042"
    assert seed_dir.is_dir()
    assert (seed_dir / "run_log.csv").exists()

    loaded_cfg = json.loads((run_dir / "config_resolved.json").read_text(encoding="utf-8"))
    assert loaded_cfg["scenario"] == "experiment_baseline"
    assert loaded_cfg["seed"] == 42

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["mode"] == "baseline"
    assert manifest["seeds"] == [42]
    assert "cfg_hash" in manifest
    assert "run_id" in manifest


def test_run_heuristic_records_cfg_path_in_manifest(tmp_path: Path) -> None:
    cfg = {
        "scenario": "experiment_baseline",
        "steps": 150,
        "top_k": 5,
        "seed": 42,
    }

    run_dir = run_heuristic(
        cfg,
        output_root=tmp_path,
        cfg_path="configs/experiment_baseline.json",
    )

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["cfg_path"] == "configs/experiment_baseline.json"
