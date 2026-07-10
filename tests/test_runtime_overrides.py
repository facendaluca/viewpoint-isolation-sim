from __future__ import annotations

from fyp_sim.runtime_overrides import apply_runtime_overrides


def test_runtime_overrides_are_non_mutating_and_auditable() -> None:
    original = {"steps": 200, "seeds": [0, 1], "policy": {"mode": "heuristic"}}

    resolved, audit = apply_runtime_overrides(
        original,
        steps=40,
        seeds=[3, 4],
        policy_mode="llm",
        llm_model="model",
        prompt_id="decision_v3.2",
    )

    assert original == {"steps": 200, "seeds": [0, 1], "policy": {"mode": "heuristic"}}
    assert resolved["steps"] == 40
    assert resolved["seeds"] == [3, 4]
    assert resolved["policy"]["mode"] == "llm"
    assert resolved["policy"]["llm"]["prompt_id"] == "decision_v3.2"
    assert audit["policy.llm.prompt_id"] == "decision_v3.2"
