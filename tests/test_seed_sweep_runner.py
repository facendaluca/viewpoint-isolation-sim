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
    # Heuristic runs keep the old schema: no LLM metadata columns.
    assert "policy_mode" not in header


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


def test_run_seed_sweep_llm_mode_writes_llm_metadata(tmp_path: Path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = repo_root / "configs" / "experiment_baseline.json"

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["scenario"] = "experiment_baseline"
    cfg["steps"] = 4
    cfg["seeds"] = [3]
    cfg["policy"] = {"mode": "llm", "llm": {"model": "fake", "rerank_slate": True}}

    # Offline stand-in for the real LLM client
    class FakeClient:
        def complete(self, prompt: str, *, timeout_s: float) -> str:  # noqa: ARG002
            return '{"action": "Sample", "confidence": 0.5}'

    import fyp_sim.runners.seed_sweep as seed_sweep_module
    from fyp_sim.agents.deciders import LLMDecider

    monkeypatch.setattr(
        seed_sweep_module,
        "build_decider",
        lambda cfg: LLMDecider(prompt_id="decision_v1", client=FakeClient(), timeout_s=1.0),
    )

    run_dir = run_seed_sweep(cfg, cfg_path=cfg_path, outputs_root=tmp_path)

    log_path = run_dir / "seeds" / "s00003" / "run_log.csv"
    with log_path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))

    assert rows
    for row in rows:
        assert row["policy_mode"] == "llm"
        assert row["llm_prompt_id"] == "decision_v1"
        assert row["llm_valid"] == "True"
        assert row["llm_fallback_reason"] == ""
        assert row["llm_action"] == "Sample"
        assert row["llm_confidence"] == "0.5"
