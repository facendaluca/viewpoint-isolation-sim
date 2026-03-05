from __future__ import annotations

import heapq
from dataclasses import dataclass

from fyp_sim.models import User, UserAction, UserPhenotype, Video
from fyp_sim.simulation import engine as base_engine


def _interest_score_once(user: User, v: Video) -> float:
    """Semantics-equivalent to policy.interest_score, but avoids generator overhead."""
    iv = user.interest_vector
    topic_best = iv.get(v.topic_category, 0.0)

    tags = v.tags
    if not tags:
        return float(topic_best)

    get = iv.get
    tag_best = 0.0
    for tag in tags:
        x = get(tag, 0.0)
        if x > tag_best:
            tag_best = x

    return float(tag_best if tag_best > topic_best else topic_best)


def _decide_action_from_interest(
    *, phenotype: UserPhenotype, sentiment_threshold: float, video_sentiment: float, interest: float
) -> UserAction:
    """Semantics-equivalent to policy.decide_action, but uses precomputed interest."""
    if video_sentiment < sentiment_threshold:
        return UserAction.AVOID

    if phenotype == UserPhenotype.WATCHER:
        return UserAction.WATCH if interest >= 0.5 else UserAction.SAMPLE

    if phenotype == UserPhenotype.SAMPLER:
        if interest >= 0.6:
            return UserAction.WATCH
        if interest >= 0.2:
            return UserAction.SAMPLE
        return UserAction.AVOID

    # AVOIDER
    return UserAction.SAMPLE if interest >= 0.8 else UserAction.AVOID


def _engagement_proxy(action: UserAction, *, duration_s: int, max_duration: int) -> float:
    """Semantics-equivalent to engine.engagement_proxy, without string conversions"""
    if action == UserAction.AVOID:
        return 0.0
    if action == UserAction.SAMPLE:
        return 0.2

    # WATCH
    if max_duration <= 0:
        return 0.8
    return 0.6 + 0.4 * (duration_s / max_duration)


def _video_score_opt(user: User, v: Video, *, rank_alpha: float, max_duration: int) -> float:
    """Semantics-equivalent to engine.video_score, but computes interest once."""
    interest = _interest_score_once(user, v)
    action = _decide_action_from_interest(
        phenotype=user.phenotype,
        sentiment_threshold=user.sentiment_threshold,
        video_sentiment=v.sentiment_score,
        interest=interest,
    )
    engagement = _engagement_proxy(action, duration_s=v.duration_s, max_duration=max_duration)
    return (1.0 - rank_alpha) * interest + rank_alpha * engagement


@dataclass(slots=True)
class WeightedTopKChooserOpt:
    """Optimised weighted Top-K chooser.

    Keeps scoring semantics identical to base_engine.choose_video_weighted_top_k(),
    but reduces complexity:
        - avoids sorting all N (O(N log N)) -> Top-K selection (O(N log K))
        - hoists max_duration out of the timestep loop
    """

    pool: list[Video]
    max_duration: int

    @classmethod
    def from_pool(cls, pool: list[Video]) -> WeightedTopKChooserOpt:
        max_d = max((v.duration_s for v in pool), default=0)
        return cls(pool=pool, max_duration=max_d)

    def __call__(
        self,
        user: User,
        pool: list[Video],
        rng,
        *,
        top_k: int,
        rank_alpha: float,
    ) -> Video:
        if top_k <= 0:
            raise ValueError("top_k must be > 0")

        k = min(top_k, len(pool))
        max_d = self.max_duration
        score_fn = _video_score_opt

        # deterministic ordering (-score, video_id)
        entries = (
            (-score_fn(user, v, rank_alpha=rank_alpha, max_duration=max_d), v.video_id, v)
            for v in pool
        )
        top = heapq.nsmallest(k, entries)

        candidates = [v for _, _, v in top]
        # Recover score from first element (-score) so we don't recompute
        weights = [max(-neg_s, 0.0) + 1e-9 for neg_s, _, _ in top]
        return rng.choices(candidates, weights=weights, k=1)[0]


def run_simulation_opt(
    *,
    user: base_engine.User,
    video_pool: list[base_engine.Video],
    steps: int,
    rng,
    top_k: int = 3,
    rank_alpha: float | None = None,
    drift_alpha: float | None = None,
    alpha: float | None = None,
    watch_time_fn=base_engine.watch_time_seconds,
    decider: base_engine.ActionDecider | None = None,
    enable_interest_updates: bool = False,
    interest_topic_alpha: float = 0.10,
    interest_tag_alpha: float = 0.05,
    interest_decay: float = 0.02,
    interest_normalise: bool = False,
    interest_prune_below: float = 0.001,
    enable_viewpoint_drift: bool = False,
    viewpoint_drift_rate: float = 0.0,
    phase_tracer=None,
) -> list[base_engine.StepLog]:
    """Baseline run_simulation() with an optimised chooser.

    Keeps all semantics in base_engine.run_simulation, only swapping the selection mechanism.
    """
    chooser = WeightedTopKChooserOpt.from_pool(video_pool)

    # Pass through everything unchanged; only chooser differs.
    return base_engine.run_simulation(
        user=user,
        video_pool=video_pool,
        steps=steps,
        rng=rng,
        top_k=top_k,
        rank_alpha=rank_alpha,
        drift_alpha=drift_alpha,
        alpha=alpha,
        chooser=chooser,
        watch_time_fn=watch_time_fn,
        decider=decider,
        enable_interest_updates=enable_interest_updates,
        interest_topic_alpha=interest_topic_alpha,
        interest_tag_alpha=interest_tag_alpha,
        interest_decay=interest_decay,
        interest_normalise=interest_normalise,
        interest_prune_below=interest_prune_below,
        enable_viewpoint_drift=enable_viewpoint_drift,
        viewpoint_drift_rate=viewpoint_drift_rate,
        phase_tracer=phase_tracer,
    )
