from __future__ import annotations

import json
from pathlib import Path

from fyp_sim.agents.deciders import LLMDecider


class _FakeClient:
    def complete(self, prompt: str, *, timeout_s: float) -> str:  # noqa: ARG002
        return '{"action":"Watch","confidence":0.8}'


def test_run_compare_isolates_outputs_and_counts_every_rerank_call(
    tmp_path: Path, monkeypatch
) -> None:
    import src.scripts.run_compare as module

    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / "configs" / "experiment_compare.json").read_text(encoding="utf-8"))
    cfg["steps"] = 4
    cfg["persistence_window"] = 2
    cfg["seeds"] = [0]
    cfg_path = tmp_path / "compare.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    output_root = tmp_path / "outputs"

    monkeypatch.setattr(
        module,
        "build_llm_decider",
        lambda cfg: LLMDecider(
            prompt_id="decision_v3.2", client=_FakeClient(), timeout_s=1.0
        ),
    )
    monkeypatch.setattr(module, "make_compare_plot", lambda run_dir, seed: run_dir / "plot.png")

    for _ in range(2):
        monkeypatch.setattr(
            "sys.argv",
            ["run_compare", "--config", str(cfg_path), "--out", str(output_root)],
        )
        module.main()

    diagnostics_files = sorted(output_root.rglob("llm_diagnostics.json"))
    assert len(diagnostics_files) == 2
    assert diagnostics_files[0].parent != diagnostics_files[1].parent

    for path in diagnostics_files:
        diagnostics = json.loads(path.read_text(encoding="utf-8"))
        expected_calls = cfg["steps"] * cfg["top_k"]
        assert diagnostics["llm_expected_call_count"] == expected_calls
        assert diagnostics["llm_call_count"] == expected_calls
        assert diagnostics["llm_valid_count"] == expected_calls
        assert diagnostics["llm_fallback_count"] == 0
        assert diagnostics["llm_prompt_id"] == "decision_v3.2"
        assert (path.parent / "comparison_diagnostics.json").exists()
