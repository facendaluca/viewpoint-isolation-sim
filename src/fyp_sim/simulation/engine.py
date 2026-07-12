"""
Simulation engine.

This module runs a minimal, reproducible simulation loop for a single user.
At each timestep t:
    1) choose a video from a pool (baseline or Top-K weighted chooser),
    2) decide an action (policy/agent),
    3) convert action -> watch time (implicit feedback proxy),
    4) compute viewpoint distance (VII_t) and its running mean (VII_cum),
    5) log everything for later analysis/reporting.

Design goal: keep the engine small and deterministic (given rng seed) so experiments are reproducible.
"""

# TODO: Refactor the engine: move logging into a separate module

from __future__ import annotations

import hashlib
import inspect
import json
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Protocol

from fyp_sim.agents import ActionDecider, HeuristicDecider
from fyp_sim.candidate_trace import CandidateRecord, CandidateTraceCollector
from fyp_sim.engagement import watch_time_seconds
from fyp_sim.interests import update_interest_vector
from fyp_sim.metrics import running_mean, viewpoint_distance
from fyp_sim.models import User, UserAction, Video
from fyp_sim.policy import decide_action, interest_score, predicted_action
from fyp_sim.simulation.viewpoint_drift import apply_viewpoint_drift


def engagement_proxy(action, video: Video, *, max_duration: int) -> float:
    """Deterministic engagement heuristic used for ranking (not actual watch time).

    Idea:
        - Avoid -> 0
        - Sample -> small constant
        - Watch -> prefers longer videos (more engagement opportunity)

    This makes rank_alpha more meaningful by preventing WATCH from always being the same value.
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


def video_score(user: User, v: Video, *, rank_alpha: float, max_duration: int) -> float:
    """Candidate ranking score: convex combo of interest and engagement proxy.

    rank_alpha = 0.0 -> interest only
    rank_alpha = 1.0 -> engagement proxy only

    The proxy comes from policy.predicted_action, the platform's own engagement
    estimate, not from the user's realised decide_action. A watcher can
    therefore be served a video it will refuse.
    """
    a = predicted_action(user, v)
    e = engagement_proxy(a, v, max_duration=max_duration)
    i = interest_score(user, v)
    return (1.0 - rank_alpha) * i + rank_alpha * e


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
    topic_interest: float
    interest_keys: int

    # State trace. Existing `interest`/`topic_interest` remain post-update aliases.
    interest_pre: float = 0.0
    interest_post: float = 0.0
    topic_interest_pre: float = 0.0
    topic_interest_post: float = 0.0
    interest_state_hash_pre: str = ""
    interest_state_hash_post: str = ""

    # Viewpoint drift logging (pre/post update)
    user_viewpoint_pre: float = 0.0
    user_viewpoint_post: float = 0.0
    video_viewpoint_score: float = 0.0

    # Policy/LLM metadata
    policy_mode: str = "heuristic"
    llm_prompt_id: str = ""
    llm_valid: bool = True
    llm_fallback_reason: str = ""
    llm_action: str = ""
    llm_confidence: float | None = None
    llm_prompt_tokens: int = 0
    llm_completion_tokens: int = 0
    llm_total_tokens: int = 0
    llm_token_count_estimated: bool = False

    agent_id: str = ""


def _interest_state_hash(user: User) -> str:
    payload = json.dumps(
        user.interest_vector, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


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
    rank_alpha: float = 0.3,
) -> Video:
    """Rank videos by score, take top_k, then choose using weighted randomness.

    - Deterministic given rng seed.
    - Stable tie-breaking (score desc, video_id asc) avoids accidental nondeterminism.
    """
    if top_k <= 0:
        raise ValueError("top_k must be > 0")

    max_d = max(v.duration_s for v in pool) if pool else 0

    scored = [(video_score(user, v, rank_alpha=rank_alpha, max_duration=max_d), v) for v in pool]
    # Stable ordering: score desc, then video_id asc (prevents randomness in ties)
    scored.sort(key=lambda x: (-x[0], x[1].video_id))

    k = min(top_k, len(scored))
    candidates = [v for _, v in scored[:k]]

    weights = []
    for s, _ in scored[:k]:
        # Ensure strictly positive weights so rng.choices never errors on all-zeros
        weights.append(max(s, 0.0) + 1e-9)

    return rng.choices(candidates, weights=weights, k=1)[0]


def select_video_llm_slate(
    user: User,
    pool: list[Video],
    rng,
    *,
    top_k: int,
    rank_alpha: float,
    decider: ActionDecider,
    trace: CandidateTraceCollector | None = None,
    trace_t: int | None = None,
):
    """LLM-in-loop selection: heuristic ranking preselects the top_k slate, then the
    decider's action for each candidate replaces the heuristic engagement proxy.

    Scoring stays the same convex combination as video_score, so rank_alpha keeps its
    meaning: it now weights how much the decider's action (Watch > Sample > Avoid)
    influences exposure. The decider is only asked about the slate, never the full
    corpus, which keeps LLM cost bounded. If the decider falls back to the heuristic
    for a candidate, that candidate's score matches the plain heuristic score.

    When `trace` is given, every decider call is also recorded together with the
    heuristic shadow action computed on the same frozen pre-update (user, video)
    context. Recording is passive: it adds no LLM calls, consumes no randomness,
    and never mutates user, pool, or decider state, so results are identical with
    tracing on or off.

    Returns (video, action, meta) so the caller can reuse the chosen candidate's
    decision instead of asking the decider a second time.
    """
    if top_k <= 0:
        raise ValueError("top_k must be > 0")

    max_d = max(v.duration_s for v in pool) if pool else 0

    # Cheap first pass: identical ranking to choose_video_weighted_top_k.
    scored = [(video_score(user, v, rank_alpha=rank_alpha, max_duration=max_d), v) for v in pool]
    scored.sort(key=lambda x: (-x[0], x[1].video_id))
    slate_scored = scored[: min(top_k, len(scored))]

    trace_records: list[CandidateRecord] | None = None
    if trace is not None:
        step_t = trace.next_t(trace_t)
        state_hash_pre = _interest_state_hash(user)
        viewpoint_pre = float(user.viewpoint_score)
        trace_records = []

    # Second pass: rescore the slate with the decider's own action.
    rescored = []
    for slate_rank, (heuristic_score, v) in enumerate(slate_scored, start=1):
        action = decider.decide_next_action(user, v)
        meta = getattr(decider, "last_meta", None)
        e = engagement_proxy(action, v, max_duration=max_d)
        i = interest_score(user, v)
        final_score = (1.0 - rank_alpha) * i + rank_alpha * e
        rescored.append((final_score, v, action, meta))
        if trace_records is not None:
            trace_records.append(
                CandidateRecord(
                    seed=trace.seed,
                    t=step_t,
                    slate_rank=slate_rank,
                    video_id=v.video_id,
                    selected=False,
                    interest_state_hash_pre=state_hash_pre,
                    user_viewpoint_pre=viewpoint_pre,
                    topic=str(v.topic_category),
                    tags="|".join(v.tags),
                    video_viewpoint=float(v.viewpoint_score),
                    video_sentiment=float(v.sentiment_score),
                    duration_s=int(v.duration_s),
                    interest=i,
                    heuristic_action=decide_action(user, v).value,
                    llm_action=action.value,
                    llm_action_raw=str(getattr(meta, "llm_action", "") or ""),
                    llm_valid=bool(getattr(meta, "valid", True)) if meta else True,
                    llm_fallback_reason=str(getattr(meta, "fallback_reason", "") or ""),
                    llm_confidence=getattr(meta, "llm_confidence", None) if meta else None,
                    heuristic_score=heuristic_score,
                    llm_engagement=e,
                    rerank_score=final_score,
                )
            )

    rescored.sort(key=lambda x: (-x[0], x[1].video_id))
    weights = [max(s, 0.0) + 1e-9 for s, _, _, _ in rescored]
    idx = rng.choices(range(len(rescored)), weights=weights, k=1)[0]
    _, video, action, meta = rescored[idx]
    if trace_records is not None:
        for record in trace_records:
            if record.video_id == video.video_id:
                record.selected = True
        trace.record_step(trace_records)
    return video, action, meta


class ChooserFn(Protocol):
    def __call__(
        self,
        user: User,
        pool: list[Video],
        rng,
        *,
        top_k: int,
        rank_alpha: float,
    ) -> Video: ...


class PhaseTracer(Protocol):
    """Optional hook for timing/observability around the timestep loop"""

    def phase(self, name: str) -> AbstractContextManager[None]: ...


def _resolve_drift_alpha(*, drift_alpha: float | None, viewpoint_drift_rate: float) -> float:
    """Resolve drift strength with backwards-compatible aliasing

    New name: drift_alpha
    Old name: viewpoint_drift_rate

    Rules:
        - drift_alpha wins if provided
        - otherwise fall back to viewpoint_drift_rate
    """
    if drift_alpha is None:
        return viewpoint_drift_rate
    return drift_alpha


def run_simulation(
    *,
    user: User,
    video_pool: list[Video],
    steps: int,
    rng,
    # Optional separate stream for watch-time randomness; defaults to rng so
    # existing callers keep byte-identical behaviour.
    engagement_rng=None,
    top_k: int = 3,
    rank_alpha: float | None = None,
    drift_alpha: float | None = None,
    # Deprecated: keep only to fail fast if old callers still pass it
    alpha: float | None = None,
    chooser: ChooserFn = choose_video_weighted_top_k,
    watch_time_fn=watch_time_seconds,
    decider: ActionDecider | None = None,
    # LLM-in-loop: rerank the top_k slate with the decider before choosing.
    # Ignores any custom chooser (see select_video_llm_slate).
    llm_rerank: bool = False,
    enable_interest_updates: bool = False,
    interest_topic_alpha: float = 0.10,
    interest_tag_alpha: float = 0.05,
    interest_decay: float = 0.02,
    interest_normalise: bool = False,
    interest_prune_below: float = 0.001,
    enable_viewpoint_drift: bool = False,
    viewpoint_drift_rate: float = 0.0,
    phase_tracer: PhaseTracer | None = None,
    # Optional matched-context recorder for the llm_rerank path; passive, so
    # passing None or a collector yields byte-identical simulations.
    candidate_trace: CandidateTraceCollector | None = None,
) -> list[StepLog]:
    """Run a minimal simulation loop and return per-step logs."""
    if alpha is not None:
        raise ValueError(
            "Config/argument 'alpha' has been split. Use 'rank_alpha' for ranking and "
            "'drift_alpha' or 'viewpoint_drift_rate' for viewpoint drift strength."
        )
    if rank_alpha is None:
        raise ValueError(
            "Missing required parameter 'rank_alpha'. "
            "The old 'alpha' parameter has been removed/split."
        )

    if steps <= 0:
        raise ValueError("steps must be > 0")
    if not video_pool:
        raise ValueError("video_pool must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    if rank_alpha < 0.0 or rank_alpha > 1.0:
        raise ValueError("rank_alpha must be between 0.0 and 1.0")

    resolved_drift_alpha = _resolve_drift_alpha(
        drift_alpha=drift_alpha,
        viewpoint_drift_rate=viewpoint_drift_rate,
    )
    if resolved_drift_alpha < 0.0:
        raise ValueError("resolved_drift_alpha must be >= 0.0")

    if decider is None:
        decider = HeuristicDecider()

    if engagement_rng is None:
        engagement_rng = rng

    # Pass the decider's action into the engagement mapping when supported, so the
    # logged action and its watch-time signal agree in every policy mode. Older
    # watch_time_fn callables without an 'action' parameter keep working unchanged.
    try:
        _watch_time_accepts_action = "action" in inspect.signature(watch_time_fn).parameters
    except (TypeError, ValueError):
        _watch_time_accepts_action = False

    def _watch_time(u, video, r, chosen_action):
        if _watch_time_accepts_action:
            return watch_time_fn(u, video, r, action=chosen_action)
        return watch_time_fn(u, video, r)

    logs: list[StepLog] = []
    vii_cum = 0.0

    if phase_tracer is None:
        for t in range(steps):
            # Exposure model: choose a candidate video from the current pool (stochastic but seeded).
            if llm_rerank:
                v, action, meta = select_video_llm_slate(
                    user,
                    video_pool,
                    rng,
                    top_k=top_k,
                    rank_alpha=rank_alpha,
                    decider=decider,
                    trace=candidate_trace,
                    trace_t=t,
                )
            else:
                v = chooser(user, video_pool, rng, top_k=top_k, rank_alpha=rank_alpha)
                action = decider.decide_next_action(user, v)
                meta = getattr(decider, "last_meta", None)
            interest_pre = interest_score(user, v)
            topic_interest_pre = float(user.interest_vector.get(v.topic_category, 0.0))
            interest_state_hash_pre = _interest_state_hash(user)
            wt = _watch_time(user, v, engagement_rng, action)

            if enable_interest_updates and action == UserAction.WATCH:
                update_interest_vector(
                    user=user,
                    video=v,
                    watch_time_s=wt,
                    topic_alpha=interest_topic_alpha,
                    tag_alpha=interest_tag_alpha,
                    decay=interest_decay,
                    normalise=interest_normalise,
                    prune_below=interest_prune_below,
                )

            user_viewpoint_pre = float(user.viewpoint_score)

            vii_t = viewpoint_distance(user.viewpoint_score, v.viewpoint_score)
            # VII_t is per-step distance, VII_cum tracks the running mean exposure distance over time.
            vii_cum = running_mean(vii_cum, t, vii_t)
            interest_post = interest_score(user, v)
            topic_interest_post = float(user.interest_vector.get(v.topic_category, 0.0))
            interest_keys = len(user.interest_vector)
            interest_state_hash_post = _interest_state_hash(user)

            # Optional viewpoint drift update happens after VII_t so baseline VII_t stays comparable.
            if enable_viewpoint_drift and resolved_drift_alpha > 0.0:
                user.viewpoint_score = apply_viewpoint_drift(
                    user_viewpoint_pre,
                    v.viewpoint_score,
                    drift_rate=resolved_drift_alpha,
                    action=action,
                )

            user_viewpoint_post = float(user.viewpoint_score)

            # Log policy/LLM metadata
            policy_mode = getattr(meta, "policy_mode", "heuristic") if meta else "heuristic"
            llm_prompt_id = getattr(meta, "prompt_id", "") or ""
            llm_valid = bool(getattr(meta, "valid", True)) if meta else True
            llm_fallback_reason = getattr(meta, "fallback_reason", "") or ""
            llm_action = getattr(meta, "llm_action", "") or ""
            llm_confidence = getattr(meta, "llm_confidence", None) if meta else None
            llm_prompt_tokens = int(getattr(meta, "prompt_tokens", 0)) if meta else 0
            llm_completion_tokens = int(getattr(meta, "completion_tokens", 0)) if meta else 0
            llm_total_tokens = int(getattr(meta, "total_tokens", 0)) if meta else 0
            llm_token_count_estimated = (
                bool(getattr(meta, "token_count_estimated", False)) if meta else False
            )

            logs.append(
                StepLog(
                    t=t,
                    video_id=v.video_id,
                    action=action.value,
                    watch_time_s=wt,
                    interest=interest_post,
                    vii_t=vii_t,
                    vii_cum=vii_cum,
                    policy_mode=policy_mode,
                    llm_prompt_id=llm_prompt_id,
                    llm_valid=llm_valid,
                    llm_fallback_reason=llm_fallback_reason,
                    llm_action=llm_action,
                    llm_confidence=llm_confidence,
                    llm_prompt_tokens=llm_prompt_tokens,
                    llm_completion_tokens=llm_completion_tokens,
                    llm_total_tokens=llm_total_tokens,
                    llm_token_count_estimated=llm_token_count_estimated,
                    topic_interest=topic_interest_post,
                    interest_keys=interest_keys,
                    interest_pre=interest_pre,
                    interest_post=interest_post,
                    topic_interest_pre=topic_interest_pre,
                    topic_interest_post=topic_interest_post,
                    interest_state_hash_pre=interest_state_hash_pre,
                    interest_state_hash_post=interest_state_hash_post,
                    user_viewpoint_pre=user_viewpoint_pre,
                    user_viewpoint_post=user_viewpoint_post,
                    video_viewpoint_score=float(v.viewpoint_score),
                )
            )

        return logs

    # Profiled path: record per-phase timings via the provided tracer.
    tracer = phase_tracer
    for t in range(steps):
        if llm_rerank:
            with tracer.phase("generate_feed"):
                v, action, meta = select_video_llm_slate(
                    user,
                    video_pool,
                    rng,
                    top_k=top_k,
                    rank_alpha=rank_alpha,
                    decider=decider,
                    trace=candidate_trace,
                    trace_t=t,
                )
            with tracer.phase("simulate_interaction"):
                wt = _watch_time(user, v, engagement_rng, action)
        else:
            with tracer.phase("generate_feed"):
                v = chooser(user, video_pool, rng, top_k=top_k, rank_alpha=rank_alpha)

            with tracer.phase("simulate_interaction"):
                action = decider.decide_next_action(user, v)
                meta = getattr(decider, "last_meta", None)
                interest_pre = interest_score(user, v)
                topic_interest_pre = float(user.interest_vector.get(v.topic_category, 0.0))
                interest_state_hash_pre = _interest_state_hash(user)
                wt = _watch_time(user, v, engagement_rng, action)

        if llm_rerank:
            interest_pre = interest_score(user, v)
            topic_interest_pre = float(user.interest_vector.get(v.topic_category, 0.0))
            interest_state_hash_pre = _interest_state_hash(user)

        with tracer.phase("update_state"):
            if enable_interest_updates and action == UserAction.WATCH:
                update_interest_vector(
                    user=user,
                    video=v,
                    watch_time_s=wt,
                    topic_alpha=interest_topic_alpha,
                    tag_alpha=interest_tag_alpha,
                    decay=interest_decay,
                    normalise=interest_normalise,
                    prune_below=interest_prune_below,
                )

            user_viewpoint_pre = float(user.viewpoint_score)

            vii_t = viewpoint_distance(user.viewpoint_score, v.viewpoint_score)
            vii_cum = running_mean(vii_cum, t, vii_t)
            interest_post = interest_score(user, v)
            topic_interest_post = float(user.interest_vector.get(v.topic_category, 0.0))
            interest_keys = len(user.interest_vector)
            interest_state_hash_post = _interest_state_hash(user)

            if enable_viewpoint_drift and resolved_drift_alpha > 0.0:
                user.viewpoint_score = apply_viewpoint_drift(
                    user_viewpoint_pre,
                    v.viewpoint_score,
                    drift_rate=resolved_drift_alpha,
                    action=action,
                )

            user_viewpoint_post = float(user.viewpoint_score)

        with tracer.phase("log_append"):
            policy_mode = getattr(meta, "policy_mode", "heuristic") if meta else "heuristic"
            llm_prompt_id = getattr(meta, "prompt_id", "") or ""
            llm_valid = bool(getattr(meta, "valid", True)) if meta else True
            llm_fallback_reason = getattr(meta, "fallback_reason", "") or ""
            llm_action = getattr(meta, "llm_action", "") or ""
            llm_confidence = getattr(meta, "llm_confidence", None) if meta else None
            llm_prompt_tokens = int(getattr(meta, "prompt_tokens", 0)) if meta else 0
            llm_completion_tokens = int(getattr(meta, "completion_tokens", 0)) if meta else 0
            llm_total_tokens = int(getattr(meta, "total_tokens", 0)) if meta else 0
            llm_token_count_estimated = (
                bool(getattr(meta, "token_count_estimated", False)) if meta else False
            )

            logs.append(
                StepLog(
                    t=t,
                    video_id=v.video_id,
                    action=action.value,
                    watch_time_s=wt,
                    interest=interest_post,
                    vii_t=vii_t,
                    vii_cum=vii_cum,
                    policy_mode=policy_mode,
                    llm_prompt_id=llm_prompt_id,
                    llm_valid=llm_valid,
                    llm_fallback_reason=llm_fallback_reason,
                    llm_action=llm_action,
                    llm_confidence=llm_confidence,
                    llm_prompt_tokens=llm_prompt_tokens,
                    llm_completion_tokens=llm_completion_tokens,
                    llm_total_tokens=llm_total_tokens,
                    llm_token_count_estimated=llm_token_count_estimated,
                    topic_interest=topic_interest_post,
                    interest_keys=interest_keys,
                    interest_pre=interest_pre,
                    interest_post=interest_post,
                    topic_interest_pre=topic_interest_pre,
                    topic_interest_post=topic_interest_post,
                    interest_state_hash_pre=interest_state_hash_pre,
                    interest_state_hash_post=interest_state_hash_post,
                    user_viewpoint_pre=user_viewpoint_pre,
                    user_viewpoint_post=user_viewpoint_post,
                    video_viewpoint_score=float(v.viewpoint_score),
                )
            )

    return logs
