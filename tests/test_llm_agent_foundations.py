from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from fyp_sim.agents.deciders import LLMClient, LLMDecider
from fyp_sim.llm.prompting import load_prompt_template, render_decision_prompt
from fyp_sim.models import User, UserAction, UserPhenotype, Video


@dataclass(slots=True)
class FakeClient(LLMClient):
    output: str

    def complete(self, prompt: str, *, timeout_s: float) -> str:  # noqa: ARG002
        return self.output


@dataclass(slots=True)
class FixedFallback:
    action: UserAction

    def decide_next_action(self, user: User, video: Video) -> UserAction:  # noqa: ARG002
        return self.action


def _make_user_video() -> tuple[User, Video]:
    user = User(
        phenotype=UserPhenotype.WATCHER,
        viewpoint_score=0.2,
        interest_vector={"sports": 0.8, "politics": 0.1},
        sentiment_threshold=0.0,
    )
    video = Video(
        1,
        "sports",
        0.1,
        0.0,
        15,
        tags=("short",),
    )
    return user, video


def test_llm_valid_then_fallback_on_invalid_output():
    user, video = _make_user_video()

    # 1) Valid LLM output -> uses LLM decision
    valid_client = FakeClient(output='{"action": "Watch", "confidence": 0.9}')
    decider_valid = LLMDecider(
        prompt_id="decision_v1",
        client=valid_client,
        timeout_s=1.0,
        fallback=FixedFallback(UserAction.AVOID),
    )
    assert decider_valid.decide_next_action(user, video) == UserAction.WATCH
    assert decider_valid.last_meta.valid is True
    assert decider_valid.last_meta.policy_mode == "llm"
    assert decider_valid.last_meta.llm_action == "Watch"
    assert decider_valid.last_meta.llm_confidence == 0.9

    # 2) Invalid LLM output -> falls back
    invalid_client = FakeClient(output="not json at all")
    decider_invalid = LLMDecider(
        prompt_id="decision_v1",
        client=invalid_client,
        timeout_s=1.0,
        fallback=FixedFallback(UserAction.SAMPLE),
    )
    assert decider_invalid.decide_next_action(user, video) == UserAction.SAMPLE
    assert decider_invalid.last_meta.valid is False
    assert decider_invalid.last_meta.fallback_reason == "invalid_output"


def test_decision_v2_4_prompt_renders() -> None:
    user, video = _make_user_video()
    prompt = render_decision_prompt("decision_v2.4", user=user, video=video)

    assert "PROMPT_ID: decision_v2.4" in prompt
    assert "phenotype: watcher" in prompt
    assert "topic_category: sports" in prompt
    assert "computed_interest_score: 0.8" in prompt
    assert "{user." not in prompt
    assert "{video." not in prompt
    assert '{"action":"Sample","confidence":0.68' in prompt


def test_configured_llm_prompt_ids_exist() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    for path in sorted((repo_root / "configs").glob("*.json")):
        cfg = json.loads(path.read_text(encoding="utf-8"))
        llm_cfg = ((cfg.get("policy") or {}).get("llm") or {})
        prompt_id = llm_cfg.get("prompt_id")
        if prompt_id:
            assert load_prompt_template(str(prompt_id))
