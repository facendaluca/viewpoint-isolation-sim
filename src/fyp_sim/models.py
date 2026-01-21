from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class UserPhenotype(str, Enum):
    AVOIDER = "avoider"
    SAMPLER = "sampler"
    WATCHER = "watcher"


class UserAction(str, Enum):
    AVOID = "Avoid"
    SAMPLE = "Sample"
    WATCH = "Watch"


@dataclass(frozen=True, slots=True)
class Video:
    video_id: int
    topic_category: str
    viewpoint_score: float  # Score indicating the viewpoint of the video (0.0 to 1.0
    sentiment_score: float  # Sentiment score of the video content (-1.0 to 1.0)
    duration_s: int  # Duration of the video in seconds (>= 0)
    tags: tuple[str, ...] = ()  # descriptive keywords, e.g.


@dataclass(slots=True)
class User:
    phenotype: UserPhenotype
    viewpoint_score: float  # User's viewpoint score (0.0 to 1.0)
    interest_vector: dict[str, float]  # topic -> affinity (0.0 to 1.0)
    sentiment_threshold: float
