from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from fyp_sim.engagement import watch_time_seconds
from fyp_sim.metrics import running_mean, viewpoint_distance
from fyp_sim.models import User, Video
from fyp_sim.policy import decide_action, interest_score


def engagement_proxy(action) -> float:
    """Simple proxy fraction used for ranking (not the actual watch time)"""
    if action.value.lower() == "avoid":
        return 0.0
    if action.value.lower() == "sample":
        return 0.2
    return 0.8  # WATCH


def video_score(user: User, v: Video, *, alpha: float) -> float:
    """Score used for candidate ranking: interest + alpha * engagement_proxy"""
    a = decide_action(user, v)
    return interest_score(user, v) + alpha * engagement_proxy(a)


@dataclass(slots=True)
class StepLog:
    t: int
    video_id: int
    action: str
    watch_time_s: int
    interest: float
    vii_t: float
    vii_cum: float


def choose_video_max_interest(user: User, pool: list[Video]) -> Video:
    """Deterministic baseline: always choose the most 'interesting' video."""
    return max(pool, key=lambda v: interest_score(user, v))


def choose_video_weighted_top_k(
    user: User,
    pool: list[Video],
    rng,
    *,
    top_k: int = 3,
    alpha: float = 0.3,
) -> Video:
    """
        Rank videos by score, take top_k, then choose using weighted randomness
    Deterministic given rng seed.
    """
    if top_k <= 0:
        raise ValueError("top_k must be > 0")

    scored = [(video_score(user, v, alpha=alpha), v) for v in pool]
    # Stable ordering: score desc, then video_id asc (prevents randomness in ties)
    scored.sort(key=lambda x: (-x[0], x[1].video_id))

    k = min(top_k, len(scored))
    candidates = [v for _, v in scored[:k]]

    weights = []
    for s, _ in scored[:k]:
        # Ensure strictly positive weights so rng.choices never errors on all-zeros
        weights.append(max(s, 0.0) + 1e-9)

    return rng.choices(candidates, weights=weights, k=1)[0]


def run_simulation(
    *,
    user: User,
    video_pool: list[Video],
    steps: int,
    rng,
    chooser: Callable[[User, list[Video]], Video] = choose_video_weighted_top_k,
    watch_time_fn=watch_time_seconds,
) -> list[StepLog]:
    """Run a minimal simulation loop and return per-step logs."""
    if steps <= 0:
        raise ValueError("steps must be > 0")
    if not video_pool:
        raise ValueError("video_pool must not be empty")

    logs: list[StepLog] = []
    vii_cum = 0.0

    for t in range(steps):
        v = chooser(user, video_pool, rng)
        action = decide_action(user, v)
        wt = watch_time_fn(user, v, rng)

        vii_t = viewpoint_distance(user.viewpoint_score, v.viewpoint_score)
        vii_cum = running_mean(vii_cum, t, vii_t)

        logs.append(
            StepLog(
                t=t,
                video_id=v.video_id,
                action=action.value,
                watch_time_s=wt,
                interest=interest_score(user, v),
                vii_t=vii_t,
                vii_cum=vii_cum,
            )
        )

    return logs
