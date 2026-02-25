"""
Core domain models for the simulation.

Conventions:
    - viewpoint_score is normalised to [0.0, 1.0] for both users and videos.
    - sentiment_score is in [-1.0, 1.0] where lower means more negative content.
    - tags are free-form keywords (strings) attached to videos.
    - interest_vector is a single map over BOTH topic categories and tags (0.0-1.0 affinity).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum


def _validate_unit_interval(value: object, *, field: str, owner: str) -> None:
    """Fail fast if `value` is not a finite number in [0.0, 1.0]."""
    try:
        v = float(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{owner}.{field} must be a number in [0.0, 1.0]; got {value!r}") from e

    if not math.isfinite(v):
        raise ValueError(f"{owner}.{field} must be a finite float in [0.0, 1.0]; got {value!r}")

    if v < 0.0 or v > 1.0:
        raise ValueError(f"{owner}.{field} out of range: {v!r} (expected 0.0 <= value <= 1.0)")


class UserPhenotype(StrEnum):
    """Survey-derived viewing style used by the policy layer."""

    AVOIDER = "avoider"
    SAMPLER = "sampler"
    WATCHER = "watcher"


class UserAction(StrEnum):
    """Discrete engagement action produced by the policy layer."""

    AVOID = "Avoid"
    SAMPLE = "Sample"
    WATCH = "Watch"


@dataclass(frozen=True, slots=True)
class Video:
    """A content item in the candidate pool."""

    video_id: int
    topic_category: str
    viewpoint_score: float  # MUST be in [0.0, 1.0]
    sentiment_score: float
    duration_s: int
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_unit_interval(
            self.viewpoint_score,
            field="viewpoint_score",
            owner=f"Video(video_id={self.video_id})",
        )


@dataclass(slots=True)
class User:
    """A simulated user with a viewing phenotype and preferences."""

    phenotype: UserPhenotype
    viewpoint_score: float  # MUST be in [0.0, 1.0]
    interest_vector: dict[str, float]
    sentiment_threshold: float

    def __post_init__(self) -> None:
        _validate_unit_interval(
            self.viewpoint_score,
            field="viewpoint_score",
            owner=f"User(phenotype={self.phenotype.value})",
        )
