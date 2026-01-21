from __future__ import annotations

from fyp_sim.models import User, UserAction, UserPhenotype, Video


def decide_action(user: User, video: Video) -> UserAction:
    """Decide user action based on phenotype, viewpoint, and sentiment."""
    topic_interest = user.interest_vector.get(video.topic_category, 0.0)
    tag_interest = max((user.interest_vector.get(t, 0.0) for t in video.tags), default=0.0)
    interest = max(topic_interest, tag_interest)

    # Sentiment gating: if content is "too negative", bias away from it
    if video.sentiment_score < user.sentiment_threshold:
        return UserAction.AVOID

    # Phenotype-driven thresholds
    if user.phenotype == UserPhenotype.WATCHER:
        return UserAction.WATCH if interest >= 0.5 else UserAction.SAMPLE
    if user.phenotype == UserPhenotype.SAMPLER:
        if interest >= 0.6:
            return UserAction.WATCH
        if interest >= 0.2:
            return UserAction.SAMPLE
        return UserAction.AVOID
    # AVOIDER
    return UserAction.SAMPLE if interest >= 0.8 else UserAction.AVOID
