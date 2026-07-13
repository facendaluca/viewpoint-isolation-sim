"""
Policy layer: maps (user, video) -> UserAction.

This encodes behavioural assumptions used by the simulation:
    - Interest is a scalar in [0, 1], derived from either topic_category or any free-form tag.
    - Sentiment gating applies first: if a video's sentiment is below the user's threshold, every phenotype avoids it regardless of interest (simple harm-avoidance model).
    - Watchers have an established taste anchored on topics: they watch any sentiment-safe video whose topic they already care about, even slightly, and refuse videos whose topic sits outside that taste no matter how appealing a single tag looks.
    - Samplers explore broadly: they sample most sentiment-safe content and only fully watch when interest is strong.
    - Avoiders are selective: they watch clear matches and avoid everything outside their bracket.

There are two action functions on purpose:
    - decide_action() is what the simulated user actually does.
    - predicted_action() is the engagement estimate the platform ranks with.

The ranker used to call decide_action() directly, so it could never serve a
video the user would refuse (the circular proxy found by the risk-01 audit).
Splitting the two means the platform ranks with a simpler interest-bracket
model that does not encode the watcher's topic-taste rule, so served videos
can now be refused. Note the honest framing: both functions read the same
User state, including interest_vector, so this is a limited (misspecified)
platform prediction model, not an enforced private-information boundary.
"""

from __future__ import annotations

from fyp_sim.models import User, UserAction, UserPhenotype, Video

# A video counts as a "clear match" for the predicted-engagement brackets and for avoiders at or above this score.
INTEREST_BRACKET_THRESHOLD = 0.5
# Below this, the platform's watcher model expects the video to be ignored rather than sampled.
WATCHER_SAMPLE_FLOOR = 0.2
# Samplers only commit to a full watch above this stronger bar.
SAMPLER_WATCH_THRESHOLD = 0.7
# A watcher's established taste: topics at or above this affinity are inside it, everything else gets refused.
WATCHER_TASTE_FLOOR = 0.2

# Version tag for the behavioural contract this module implements. A config
# hash cannot tell runs made before and after the risk-01 watcher change apart
# (the configs are byte-identical), so run manifests record this tag instead.
# v2 = the watcher topic-taste rule in decide_action plus the split between
# predicted_action (ranking) and decide_action (realised behaviour).
POLICY_CONTRACT = "watcher-topic-taste-v2"


def decide_action(user: User, video: Video) -> UserAction:
    """Decide the realised action based on sentiment gating, interest, and phenotype."""
    interest = interest_score(user, video)

    # Sentiment gating: if content is "too negative", every phenotype avoids it
    if video.sentiment_score < user.sentiment_threshold:
        return UserAction.AVOID

    if user.phenotype is UserPhenotype.WATCHER:
        # Established taste: the video's own topic must already be in the profile.
        # High interest and slight interest both earn a watch; an appealing tag
        # on an off-taste topic does not.
        topic_affinity = user.interest_vector.get(video.topic_category, 0.0)
        if topic_affinity >= WATCHER_TASTE_FLOOR:
            return UserAction.WATCH
        return UserAction.AVOID

    if user.phenotype is UserPhenotype.SAMPLER:
        if interest >= SAMPLER_WATCH_THRESHOLD:
            return UserAction.WATCH
        return UserAction.SAMPLE

    # AVOIDER
    if interest >= INTEREST_BRACKET_THRESHOLD:
        return UserAction.WATCH
    return UserAction.AVOID


def predicted_action(user: User, video: Video) -> UserAction:
    """Estimate the action the platform expects, for ranking only.

    This is the interest-bracket model the recommender believes in. It reads
    the same User state as decide_action but is too coarse to capture the
    watcher's topic-taste rule, so a tag-hooked video still looks engaging
    here even when the watcher will refuse it. That gap is a misspecified
    platform model, not withheld information. For samplers and avoiders the
    estimate matches their realised behaviour.
    """
    interest = interest_score(user, video)

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
