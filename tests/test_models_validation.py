from __future__ import annotations

import pytest

from fyp_sim.models import User, UserPhenotype, Video


def test_video_viewpoint_score_out_of_range_raises() -> None:
    with pytest.raises(ValueError, match=r"viewpoint_score.*out of range"):
        Video(
            video_id=1,
            topic_category="politics",
            viewpoint_score=-0.10,  # out of range
            sentiment_score=0.0,
            duration_s=25,
            tags=(),
        )


def test_user_viewpoint_score_out_of_range_raises() -> None:
    with pytest.raises(ValueError, match=r"viewpoint_score.*out of range"):
        User(
            phenotype=UserPhenotype.SAMPLER,
            viewpoint_score=1.10,  # out of range
            interest_vector={},
            sentiment_threshold=-0.2,
        )


def test_boundary_values_are_allowed() -> None:
    Video(
        video_id=2,
        topic_category="politics",
        viewpoint_score=0.0,
        sentiment_score=0.0,
        duration_s=25,
        tags=(),
    )
    Video(
        video_id=3,
        topic_category="politics",
        viewpoint_score=1.0,
        sentiment_score=0.0,
        duration_s=25,
        tags=(),
    )
    User(
        phenotype=UserPhenotype.SAMPLER,
        viewpoint_score=0.0,
        interest_vector={"politics": 0.5},
        sentiment_threshold=-0.2,
    )
    User(
        phenotype=UserPhenotype.SAMPLER,
        viewpoint_score=1.0,
        interest_vector={"politics": 0.5},
        sentiment_threshold=-0.2,
    )
