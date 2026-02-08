"""
Core domain models for the simulation.

Conventions:
    - viewpoint_score is normalised to [0.0, 1.0] for both users and videos.
    - sentiment_score is in [-1.0, 1.0] where lower means more negative content.
    - tags are free-form keywords (strings) attached to videos.
    - interest_vector is a single map over BOTH topic categories and tags (0.0-1.0 affinity).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


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
    topic_category: str  # Taxonomy category label, e.g. "politics"
    viewpoint_score: float  # Normalised stance score in [0.0, 1.0]
    sentiment_score: float  # Content sentiment in [-1.0, 1.0]
    duration_s: int  # Video duration in seconds (>= 0)
    tags: tuple[str, ...] = ()  # Free-form keywords, e.g. "meme", "sleep"


@dataclass(slots=True)
class User:
    """A simulated user with a viewing phenotype and preferences."""

    phenotype: UserPhenotype
    viewpoint_score: float  # Normalised stance score in [0.0, 1.0]
    interest_vector: dict[str, float]  # Topic/tag -> affinity in [0.0, 1.0]
    sentiment_threshold: float  # Avoid content with sentiment_score < threshold
