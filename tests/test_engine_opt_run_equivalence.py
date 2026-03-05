from __future__ import annotations

import random

import pytest

from fyp_sim.analysis import summarise_logs
from fyp_sim.models import User, UserPhenotype, Video
from fyp_sim.policy import decide_action, interest_score
from fyp_sim.simulation.engine import run_simulation as run_simulation_baseline
from fyp_sim.simulation.engine_opt import _decide_action_from_interest, run_simulation_opt


def _make_realistic_pool(seed: int, n: int = 400) -> list[Video]:
    rng = random.Random(seed)

    topics = ["comedy_memes", "sports", "finance_economics", "health_wellness", "politics"]
    tag_vocab = [
        "funny",
        "gym",
        "stocks",
        "election",
        "recipe",
        "football",
        "skincare",
        "travel",
        "ai",
        "music",
    ]

    pool: list[Video] = []
    for vid in range(1, n + 1):
        topic = topics[rng.randrange(len(topics))]

        # 0–3 tags (realistic mix)
        k = rng.randrange(0, 4)
        tags = tuple(rng.sample(tag_vocab, k=k)) if k else ()

        # vary duration and sentiment; include some below threshold
        duration_s = rng.randrange(5, 91)  # 5s..90s
        sentiment = rng.uniform(-0.3, 0.6)  # includes negatives for gating tests
        viewpoint = rng.uniform(0.0, 1.0)

        pool.append(
            Video(
                video_id=vid,
                topic_category=topic,
                viewpoint_score=float(viewpoint),
                sentiment_score=float(sentiment),
                duration_s=int(duration_s),
                tags=tags,
            )
        )

    # Add duplicated-feature videos to stress tie-breaking determinism
    for extra in range(n + 1, n + 21):
        pool.append(
            Video(
                video_id=extra,
                topic_category="sports",
                viewpoint_score=0.5,
                sentiment_score=0.2,
                duration_s=30,
                tags=("football", "travel"),
            )
        )

    return pool


def _make_user(phenotype: UserPhenotype) -> User:
    # Mixed topic + tag interests (realistic)
    iv = {
        "comedy_memes": 0.75,
        "sports": 0.55,
        "finance_economics": 0.35,
        "health_wellness": 0.60,
        "politics": 0.20,
        # tags
        "funny": 0.80,
        "gym": 0.70,
        "stocks": 0.45,
        "football": 0.60,
        "recipe": 0.30,
        "ai": 0.25,
        "travel": 0.40,
    }

    # Set a threshold so some videos get gated to AVOID
    return User(
        phenotype=phenotype,
        viewpoint_score=0.5,
        interest_vector=iv,
        sentiment_threshold=0.0,
    )


@pytest.mark.parametrize(
    "phenotype", [UserPhenotype.WATCHER, UserPhenotype.SAMPLER, UserPhenotype.AVOIDER]
)
def test_decide_action_matches_opt_wrapper_on_randomised_videos(phenotype: UserPhenotype) -> None:
    """
    - uses policy.interest_score (topic+tags)
    - compares policy.decide_action vs _decide_action_from_interest across many random videos
    """
    user = _make_user(phenotype)
    pool = _make_realistic_pool(seed=123, n=200)

    for v in pool:
        i = interest_score(user, v)
        baseline = decide_action(user, v)
        opt = _decide_action_from_interest(
            phenotype=user.phenotype,
            sentiment_threshold=user.sentiment_threshold,
            video_sentiment=v.sentiment_score,
            interest=i,
        )
        assert opt == baseline


@pytest.mark.parametrize(
    "phenotype", [UserPhenotype.WATCHER, UserPhenotype.SAMPLER, UserPhenotype.AVOIDER]
)
def test_run_simulation_opt_matches_baseline_realistic_pool(phenotype: UserPhenotype) -> None:
    """
    End-to-end equivalence:
    - same RNG seed
    - same pool
    - compare trajectory-critical fields per step
    - compare summarise_logs output
    """
    pool = _make_realistic_pool(seed=999, n=350)
    steps = 400
    top_k = 10
    rank_alpha = 0.5
    seed = 42

    user_base = _make_user(phenotype)
    user_opt = _make_user(phenotype)

    logs_base = run_simulation_baseline(
        user=user_base,
        video_pool=pool,
        steps=steps,
        rng=random.Random(seed),
        top_k=top_k,
        rank_alpha=rank_alpha,
        drift_alpha=0.0,
        enable_interest_updates=False,
        enable_viewpoint_drift=False,
        viewpoint_drift_rate=0.0,
    )

    logs_opt = run_simulation_opt(
        user=user_opt,
        video_pool=pool,
        steps=steps,
        rng=random.Random(seed),
        top_k=top_k,
        rank_alpha=rank_alpha,
        drift_alpha=0.0,
        enable_interest_updates=False,
        enable_viewpoint_drift=False,
        viewpoint_drift_rate=0.0,
    )

    assert len(logs_opt) == len(logs_base) == steps

    # Compare the fields that guarantee identical dynamics for this deterministic setup.
    base_trip = [(r.video_id, r.action, r.watch_time_s) for r in logs_base]
    opt_trip = [(r.video_id, r.action, r.watch_time_s) for r in logs_opt]

    if opt_trip != base_trip:
        # Helpful failure message: show first mismatch
        for idx, (a, b) in enumerate(zip(base_trip, opt_trip, strict=True)):
            if a != b:
                raise AssertionError(f"first mismatch at step {idx}: baseline={a} opt={b}")

    # Results-level equivalence: summary metrics should match exactly if logs match.
    s_base = summarise_logs(logs_base, lock_in_threshold=0.10, persistence_window=10)
    s_opt = summarise_logs(logs_opt, lock_in_threshold=0.10, persistence_window=10)
    assert s_opt == s_base
