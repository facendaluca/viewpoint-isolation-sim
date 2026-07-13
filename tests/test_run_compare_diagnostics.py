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


def test_candidate_trace_has_no_observer_effect_and_writes_matched_diagnostics(
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
        lambda cfg: LLMDecider(prompt_id="decision_v3.2", client=_FakeClient(), timeout_s=1.0),
    )
    monkeypatch.setattr(module, "make_compare_plot", lambda run_dir, seed: run_dir / "plot.png")

    for extra in ([], ["--no-candidate-trace"]):
        monkeypatch.setattr(
            "sys.argv",
            ["run_compare", "--config", str(cfg_path), "--out", str(output_root), *extra],
        )
        module.main()

    run_dirs = sorted(path.parent for path in output_root.rglob("llm_diagnostics.json"))
    assert len(run_dirs) == 2
    trace_on_dirs = [d for d in run_dirs if (d / "matched_policy_diagnostics.json").exists()]
    assert len(trace_on_dirs) == 1
    trace_on_dir = trace_on_dirs[0]
    trace_off_dir = next(d for d in run_dirs if d != trace_on_dir)

    # Tracing must not change the simulation: selected-step logs are
    # byte-identical with the trace on and off.
    for arm in ("llm", "heuristic"):
        on_log = (trace_on_dir / "logs" / arm / "run_seed_0.csv").read_text(encoding="utf-8")
        off_log = (trace_off_dir / "logs" / arm / "run_seed_0.csv").read_text(encoding="utf-8")
        assert on_log == off_log

    assert not (trace_off_dir / "matched_policy_diagnostics.json").exists()
    assert not (trace_off_dir / "candidate_trace").exists()

    expected_rows = cfg["steps"] * cfg["top_k"]
    trace_csv = trace_on_dir / "candidate_trace" / "candidate_trace_seed_0.csv"
    assert trace_csv.exists()
    assert len(trace_csv.read_text(encoding="utf-8").splitlines()) == expected_rows + 1

    matched = json.loads(
        (trace_on_dir / "matched_policy_diagnostics.json").read_text(encoding="utf-8")
    )
    assert matched["comparison_type"] == "matched_candidate_context"
    assert matched["gates"] == {
        "rows_equal_llm_calls": True,
        "confusion_totals_equal_rows": True,
    }
    aggregate = matched["aggregate"]
    assert aggregate["expected_rows"] == aggregate["observed_rows"] == expected_rows
    assert aggregate["confusion_total"] == expected_rows

    comparison = json.loads(
        (trace_on_dir / "comparison_diagnostics.json").read_text(encoding="utf-8")
    )
    assert comparison["comparison_type"] == "unpaired_architecture_divergence"
    assert "unpaired_action_difference_rate" in comparison["aggregate"]
    assert "action_difference_rate" not in comparison["aggregate"]
    assert "action_difference_steps" not in comparison["per_seed"][0]

    manifest = json.loads((trace_on_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["candidate_trace_enabled"] is True
    assert manifest["matched_policy_diagnostics_path"] == "matched_policy_diagnostics.json"
