from __future__ import annotations

from dataclasses import dataclass

from fyp_sim.agents.deciders import LLMClient, LLMDecider
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

    # 2) Invalid LLM output -> falls back
    invalid_client = FakeClient(output="not json at all")
    decider_invalid = LLMDecider(
        prompt_id="decision_v1",
        client=invalid_client,
        timeout_s=1.0,
        fallback=FixedFallback(UserAction.SAMPLE),
    )
    assert decider_invalid.decide_next_action(user, video) == UserAction.SAMPLE
