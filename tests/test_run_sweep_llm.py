from __future__ import annotations

import csv
import json
from pathlib import Path

from fyp_sim.agents.deciders import LLMDecider


class _FakeClient:
    def complete(self, prompt: str, *, timeout_s: float) -> str:  # noqa: ARG002
        return '{"action":"Sample","confidence":0.7}'


def test_run_sweep_honours_llm_policy_and_writes_per_seed_traces(
    tmp_path: Path, monkeypatch
) -> None:
    import src.scripts.run_sweep as module

    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / "configs" / "experiment_sweep.json").read_text(encoding="utf-8"))
    cfg["steps"] = 3
    cfg["persistence_window"] = 2
    cfg["seeds"] = [0]
    cfg["top_k_grid"] = [1, 2]
    cfg["rank_alpha_grid"] = [0.3]
    cfg["policy"] = {
        "mode": "llm",
        "llm": {
            "model": "fake",
            "prompt_id": "decision_v3.2",
            "rerank_slate": True,
        },
    }
    cfg_path = tmp_path / "sweep.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    output_root = tmp_path / "outputs"

    monkeypatch.setattr(
        module,
        "build_decider",
        lambda cfg: LLMDecider(
            prompt_id="decision_v3.2", client=_FakeClient(), timeout_s=1.0
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["run_sweep", str(cfg_path), "--out", str(output_root)],
    )

    module.main()

    diagnostics_path = next(output_root.rglob("llm_diagnostics.json"))
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert diagnostics["llm_expected_call_count"] == 9
    assert diagnostics["llm_call_count"] == 9
    assert diagnostics["llm_valid_count"] == 9
    assert diagnostics["llm_fallback_count"] == 0

    run_dir = diagnostics_path.parent
    with (run_dir / "per_seed_summary.csv").open(newline="") as file:
        assert len(list(csv.DictReader(file))) == 2
    logs = sorted(run_dir.rglob("run_log.csv"))
    assert len(logs) == 2
    with logs[0].open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 3
    assert all(row["llm_prompt_id"] == "decision_v3.2" for row in rows)
