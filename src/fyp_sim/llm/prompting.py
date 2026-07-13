from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from fyp_sim.models import User, Video
from fyp_sim.policy import interest_score

_PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


@lru_cache(maxsize=32)
def load_prompt_template(prompt_id: str) -> str:
    path = _PROMPT_DIR / f"{prompt_id}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


@dataclass(frozen=True, slots=True)
class _UserPromptView:
    phenotype: str
    viewpoint_score: float
    sentiment_threshold: float
    interest_vector: str  # stable JSON string


@dataclass(frozen=True, slots=True)
class _VideoPromptView:
    video_id: int
    topic_category: str
    viewpoint_score: float
    sentiment_score: float
    duration_s: int
    tags: str  # stable JSON string
    interest_score: float


def render_decision_prompt(prompt_id: str, *, user: User, video: Video) -> str:
    template = load_prompt_template(prompt_id)

    u = _UserPromptView(
        phenotype=str(user.phenotype.value),
        viewpoint_score=float(user.viewpoint_score),
        sentiment_threshold=float(user.sentiment_threshold),
        interest_vector=json.dumps(user.interest_vector, sort_keys=True),
    )

    v = _VideoPromptView(
        video_id=int(video.video_id),
        topic_category=str(getattr(video.topic_category, "value", video.topic_category)),
        viewpoint_score=float(video.viewpoint_score),
        sentiment_score=float(video.sentiment_score),
        duration_s=int(video.duration_s),
        tags=json.dumps(list(video.tags)),
        interest_score=float(interest_score(user, video)),
    )

    # Template uses {user.*} and {video.*}
    return template.format(
        user=u,
        video=v,
    )
