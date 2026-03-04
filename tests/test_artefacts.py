from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from fyp_sim.artefacts import create_run_artefacts
from fyp_sim.models import Video


def test_create_run_artefacrs_creates_expected_structure(tmp_path: Path) -> None:
    cfg = {
        "steps": 10,
        "top_k": 3,
        "rank_alpha": 0.25,
        "drift_alpha": 0.0,
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
    key_params = manifest["key_params"]
    assert key_params["rank_alpha"] == 0.25
    assert key_params["drift_alpha"] == 0.0


def test_create_run_artefacts_writes_corpus_and_hash(tmp_path: Path) -> None:
    cfg = {
        "steps": 3,
        "top_k": 2,
        "rank_alpha": 0.25,
        "drift_alpha": 0.0,
        "corpus": {"source": "generated", "seed": 123},
    }
    seeds = [1]

    corpus = [
        Video(
            video_id=2,
            topic_category="sports",
            viewpoint_score=0.8,
            sentiment_score=0.0,
            duration_s=15,
            tags=("tags2",),
        ),
        Video(
            video_id=1,
            topic_category="politics",
            viewpoint_score=0.2,
            sentiment_score=-1.0,
            duration_s=15,
            tags=("tag1", "tag3"),
        ),
    ]

    artefacts = create_run_artefacts(
        cfg=cfg,
        cfg_path=None,
        mode="baseline",
        seeds=seeds,
        outputs_root=tmp_path / "outputs" / "runs",
        corpus=corpus,
    )

    corpus_path = artefacts.root_dir / "corpus.json"
    assert corpus_path.exists()

    manifest = json.loads(artefacts.manifest_path.read_text(encoding="utf-8"))
    assert "corpus" in manifest
    assert manifest["corpus"]["path"] == "corpus.json"
    assert manifest["corpus"]["n_videos"] == 2
    assert manifest["corpus"]["seed"] == 123
    assert manifest["corpus"]["source"] == "generated"

    payload = json.loads(corpus_path.read_text(encoding="utf-8"))
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert manifest["corpus"]["hash"] == expected
