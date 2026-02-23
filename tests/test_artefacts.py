from __future__ import annotations

import json
import re
from pathlib import Path

from fyp_sim.artefacts import create_run_artefacts


def test_create_run_artefacrs_creates_expected_structure(tmp_path: Path) -> None:
    cfg = {
        "steps": 10,
        "top_k": 3,
        "alpha": 0.25,
        "enable_interest_updates": True,
        "interest_topic_alpha": 0.1,
        "interest_tag_alpha": 0.05,
        "interest_decay": 0.02,
        "interest_normalise": False,
        "interest_prune_below": 0.001,
    }
    seeds = [1, 42]

    artefacts = create_run_artefacts(
        cfg=cfg,
        cfg_path=Path("configs/experiment_baseline.json"),
        mode="heuristic",
        seeds=seeds,
        outputs_root=tmp_path / "outputs" / "runs",
    )

    assert re.match(r"^\d{6}Z_[a-z]+_[0-9a-f]{8}$", artefacts.run_id)
    assert artefacts.root_dir.exists()
    assert artefacts.seeds_dir.exists()
    assert artefacts.plots_dir.exists()
    assert artefacts.manifest_path.exists()
    assert artefacts.summary_path.name == "summary.csv"

    manifest = json.loads(artefacts.manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_id"] == artefacts.run_id
    assert manifest["mode"] == "heuristic"
    assert manifest["seeds"] == seeds
    assert "cfg_hash" in manifest
