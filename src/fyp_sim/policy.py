"""
Policy layer: maps (user, video) -> UserAction.

This encodes behavioural assumptions used by the simulation:
    - Interest is a scalar in [0, 1], derived from either topic_category or any free-form tag.
    - Sentiment gating applies first: if a video's sentiment is below the user's threshold, every phenotype avoids it regardless of interest (simple harm-avoidance model).
    - Watchers have established taste: they watch clear matches, sample adjacent or uncertain content, and avoid clearly irrelevant content.
    - Samplers explore broadly: they sample most sentiment-safe content and only fully watch when interest is strong.
    - Avoiders are selective: they watch clear matches and avoid everything outside their bracket.
"""

from __future__ import annotations

from fyp_sim.models import User, UserAction, UserPhenotype, Video

# A video counts as a "clear match" for watchers and avoiders at or above this score.
INTEREST_BRACKET_THRESHOLD = 0.5
# Below this, a watcher treats the video as clearly irrelevant and avoids it.
WATCHER_SAMPLE_FLOOR = 0.2
# Samplers only commit to a full watch above this stronger bar.
SAMPLER_WATCH_THRESHOLD = 0.7


def decide_action(user: User, video: Video) -> UserAction:
    """Decide action based on sentiment gating, interest, and phenotype."""
    interest = interest_score(user, video)

    # Sentiment gating: if content is "too negative", every phenotype avoids it
    if video.sentiment_score < user.sentiment_threshold:
        return UserAction.AVOID

    if user.phenotype is UserPhenotype.WATCHER:
        if interest >= INTEREST_BRACKET_THRESHOLD:
            return UserAction.WATCH
        if interest >= WATCHER_SAMPLE_FLOOR:
            return UserAction.SAMPLE
        return UserAction.AVOID

    if user.phenotype is UserPhenotype.SAMPLER:
        if interest >= SAMPLER_WATCH_THRESHOLD:
            return UserAction.WATCH
        return UserAction.SAMPLE

    # AVOIDER
    if interest >= INTEREST_BRACKET_THRESHOLD:
        return UserAction.WATCH
    return UserAction.AVOID


def interest_score(user: User, video: Video) -> float:
    """Interest keyed by both topic_category and free-form tags (0.0-1.0)"""
    topic_interest = user.interest_vector.get(video.topic_category, 0.0)
    tag_interest = max((user.interest_vector.get(tag, 0.0) for tag in video.tags), default=0.0)
    return max(topic_interest, tag_interest)
