"""
Simulation engine.

This module runs a minimal, reproducible simulation loop for a single user.
At each timestep t:
    1) choose a video from a pool (baseline or Top-K weighted chooser),
    2) decide an action (policy),
    3) convert action -> watch time (implicit feedback proxy),
    4) compute viewpoint distance (VII_t) and its running mean (VII_cum),
    5) log everything for later analysis/reporting.

Design goal: keep the engine small and deterministic (given rng seed) so experiments are reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from fyp_sim.engagement import watch_time_seconds
from fyp_sim.metrics import running_mean, viewpoint_distance
from fyp_sim.models import User, Video
from fyp_sim.policy import decide_action, interest_score


def engagement_proxy(action, video: Video, *, max_duration: int) -> float:
    """Deterministic engagement heuristic used for ranking (not actual watch time).

    Idea:
        - Avoid -> 0
        - Sample -> small constant
        - Watch -> prefers longer videos (more engagement opportunity)

    This makes alpha more meaningful by preventing WATCH from always being the same value.
    """
    a = action.value.lower()
    if a == "avoid":
        return 0.0
    if a == "sample":
        return 0.2

    # WATCH: scale with duration, bounded in [0.6, 1.0]
    if max_duration <= 0:
        return 0.8
    return 0.6 + 0.4 * (video.duration_s / max_duration)


def video_score(user: User, v: Video, *, alpha: float, max_duration: int) -> float:
    """Candidate ranking score: convex combo of interest and engagement proxy.

    alpha = 0.0 -> interest only
    alpha = 1.0 -> engagement proxy only
    """
    a = decide_action(user, v)
    e = engagement_proxy(a, v, max_duration=max_duration)
    i = interest_score(user, v)
    return (1.0 - alpha) * i + alpha * e


# Step-level log row written out by scripts (e.g., CSV) for analysis and reporting.
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
    """Deterministic baseline: always choose the most 'interesting' video.

    Useful as a sanity check/baseline policy (no exploration).
    """
    return max(pool, key=lambda v: interest_score(user, v))


def choose_video_weighted_top_k(
    user: User,
    pool: list[Video],
    rng,
    *,
    top_k: int = 3,
    alpha: float = 0.3,
) -> Video:
    """Rank videos by score, take top_k, then choose using weighted randomness.

    - Deterministic given rng seed.
    - Stable tie-breaking (score desc, video_id asc) avoids accidental nondeterminism.
    """
    if top_k <= 0:
        raise ValueError("top_k must be > 0")

    max_d = max(v.duration_s for v in pool) if pool else 0

    scored = [(video_score(user, v, alpha=alpha, max_duration=max_d), v) for v in pool]
    # Stable ordering: score desc, then video_id asc (prevents randomness in ties)
    scored.sort(key=lambda x: (-x[0], x[1].video_id))

    k = min(top_k, len(scored))
    candidates = [v for _, v in scored[:k]]

    weights = []
    for s, _ in scored[:k]:
        # Ensure strictly positive weights so rng.choices never errors on all-zeros
        weights.append(max(s, 0.0) + 1e-9)

    return rng.choices(candidates, weights=weights, k=1)[0]


class ChooserFn(Protocol):
    def __call__(
        self,
        user: User,
        pool: list[Video],
        rng,
        *,
        top_k: int,
        alpha: float,
    ) -> Video: ...


def run_simulation(
    *,
    user: User,
    video_pool: list[Video],
    steps: int,
    rng,
    top_k: int = 3,
    alpha: float = 0.3,
    chooser: ChooserFn = choose_video_weighted_top_k,
    watch_time_fn=watch_time_seconds,
) -> list[StepLog]:
    """Run a minimal simulation loop and return per-step logs."""
    if steps <= 0:
        raise ValueError("steps must be > 0")
    if not video_pool:
        raise ValueError("video_pool must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError("alpha must be between 0.0 and 1.0")

    logs: list[StepLog] = []
    vii_cum = 0.0

    for t in range(steps):
        # Exposure model: choose a candidate video from the current pool (stochastic but seeded).
        v = chooser(user, video_pool, rng, top_k=top_k, alpha=alpha)
        action = decide_action(user, v)
        wt = watch_time_fn(user, v, rng)

        vii_t = viewpoint_distance(user.viewpoint_score, v.viewpoint_score)
        # VII_t is per-step distance, VII_cum tracks the running mean exposure distance over time.
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
