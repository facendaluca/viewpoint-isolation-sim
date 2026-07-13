"""Risk-01 audit tool tests.

Covers the properties the audit must guarantee:
1. Observational transparency: auditing a run changes nothing about it.
2. Honest localisation: the collapse-stage classifier reports where non-Watch
   opportunity disappears, for saturated, slate-collapsed and healthy fixtures.
3. Predicted-versus-realised evidence: pool, slate and served counts, the
   confusion matrix and the refusal rate all reconcile, and the frozen E3
   corpus reproduces the established deterministic t=0 counts.
4. Config handling: heuristic configs like E3 are validated as batch configs,
   compare configs keep the compare contract.
"""

from __future__ import annotations

import random
from pathlib import Path

from fyp_sim.corpus import build_corpus
from fyp_sim.models import User, UserPhenotype, Video
from fyp_sim.simulation.engine import choose_video_weighted_top_k, run_simulation
from src.scripts.audit_action_opportunity import (
    OpportunityAuditor,
    classify_collapse_stage,
    detect_runner_kind,
    replay_heuristic_arm,
    summarise_seed,
)
from src.scripts.run_compare import load_config

E3_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "eval" / "E3_explore_low.json"


def _sampler_user() -> User:
    return User(
        phenotype=UserPhenotype.SAMPLER,
        viewpoint_score=0.4,
        interest_vector={"meme": 0.72, "comedy": 0.55},
        sentiment_threshold=-0.2,
    )


def _mixed_pool() -> list[Video]:
    """Pool with all three opportunity kinds for the sampler user above.

    6 meme-tagged videos (interest 0.72 -> Watch), 8 plain comedy videos
    (interest 0.55 -> Sample), 3 sentiment-gated videos (-> Avoid).
    """
    pool = [Video(i, "comedy", 0.5, 0.0, 20 + i, tags=("meme",)) for i in range(1, 7)]
    pool += [Video(i, "comedy", 0.5, 0.0, 30) for i in range(7, 15)]
    pool += [Video(i, "comedy", 0.5, -0.9, 30) for i in range(15, 18)]
    return pool


def _run(user, pool, *, chooser, seed, steps=25, top_k=3, **kwargs):
    rng = random.Random(seed)
    engagement_rng = random.Random(f"{seed}:engagement")
    logs = run_simulation(
        user=user,
        video_pool=pool,
        steps=steps,
        rng=rng,
        top_k=top_k,
        rank_alpha=0.3,
        drift_alpha=0.02,
        enable_viewpoint_drift=True,
        engagement_rng=engagement_rng,
        chooser=chooser,
        **kwargs,
    )
    return logs, rng, engagement_rng


def test_auditor_is_observationally_transparent():
    interest_kwargs = {"enable_interest_updates": True, "interest_tag_alpha": 0.05}

    auditor = OpportunityAuditor(seed=7)
    audited_user = _sampler_user()
    audited_logs, audited_rng, audited_eng = _run(
        audited_user, _mixed_pool(), chooser=auditor.chooser, seed=7, **interest_kwargs
    )

    plain_user = _sampler_user()
    plain_logs, plain_rng, plain_eng = _run(
        plain_user, _mixed_pool(), chooser=choose_video_weighted_top_k, seed=7, **interest_kwargs
    )

    # Same served videos, actions, watch times and metrics, step by step.
    assert audited_logs == plain_logs
    # Same RNG consumption on both streams.
    assert audited_rng.getstate() == plain_rng.getstate()
    assert audited_eng.getstate() == plain_eng.getstate()
    # Same end-of-run user state.
    assert audited_user.interest_vector == plain_user.interest_vector
    assert audited_user.viewpoint_score == plain_user.viewpoint_score
    # And the audit actually recorded every step.
    assert len(auditor.step_rows) == len(plain_logs)


def test_audit_localises_slate_collapse_when_pool_is_mixed():
    # With 6 Watch-eligible meme videos and top_k=5, ranking fills the whole
    # slate with Watch candidates even though most of the pool is not Watch.
    auditor = OpportunityAuditor(seed=0)
    user = _sampler_user()
    logs, _, _ = _run(user, _mixed_pool(), chooser=auditor.chooser, seed=0, steps=10, top_k=5)

    for row in auditor.step_rows:
        assert row["pool_sample"] > 0
        assert row["pool_avoid"] > 0
        assert row["pool_sentiment_gated"] == 3
        assert row["slate_watch"] == 5
        assert row["slate_sample"] == 0 and row["slate_avoid"] == 0
        # Interest-only ranking would fill the slate the same way, so the
        # circular engagement proxy is not what removes the opportunity here.
        assert row["interest_only_slate_watch"] == 5

    summary = summarise_seed(auditor.step_rows, logs)
    assert summary["collapse_stage"] == "top_k_slate_selection"
    assert summary["served_matches_logged"] is True
    assert summary["logged_action_counts"] == {"Watch": 10}


def test_audit_reports_policy_saturation_when_whole_pool_is_watch():
    user = User(
        phenotype=UserPhenotype.WATCHER,
        viewpoint_score=0.5,
        interest_vector={"sports": 0.9},
        sentiment_threshold=-0.2,
    )
    pool = [Video(i, "sports", 0.5, 0.0, 30) for i in range(1, 9)]
    auditor = OpportunityAuditor(seed=1)
    logs, _, _ = _run(user, pool, chooser=auditor.chooser, seed=1, steps=5, top_k=3)

    summary = summarise_seed(auditor.step_rows, logs)
    assert summary["collapse_stage"] == "policy_pool_saturation"
    assert summary["pool_t0"] == {"watch": 8, "sample": 0, "avoid": 0, "sentiment_gated": 0}


def test_audit_reports_no_collapse_when_served_actions_mix():
    # One Watch-eligible video and four Sample videos, all on the slate:
    # weighted selection keeps serving a mixture of actions.
    user = User(
        phenotype=UserPhenotype.SAMPLER,
        viewpoint_score=0.5,
        interest_vector={"sports": 0.75, "music": 0.3},
        sentiment_threshold=-0.2,
    )
    pool = [Video(1, "sports", 0.5, 0.0, 30)] + [
        Video(i, "music", 0.5, 0.0, 30) for i in range(2, 6)
    ]
    auditor = OpportunityAuditor(seed=0)
    logs, _, _ = _run(user, pool, chooser=auditor.chooser, seed=0, steps=15, top_k=5)

    summary = summarise_seed(auditor.step_rows, logs)
    assert summary["collapse_stage"] == "no_watch_collapse"
    assert set(summary["logged_action_counts"]) == {"Watch", "Sample"}


def test_audit_detects_first_feedback_driven_opportunity_change():
    # Watching the sports video bumps the shared 'gym' tag hard enough that the
    # second video crosses the sampler's watch bar, so the pool mix changes at
    # the very next step. The full-duration watch_time_fn makes it exact.
    user = User(
        phenotype=UserPhenotype.SAMPLER,
        viewpoint_score=0.5,
        interest_vector={"sports": 0.75, "gym": 0.18},
        sentiment_threshold=-0.2,
    )
    pool = [
        Video(1, "sports", 0.5, 0.0, 30, tags=("gym",)),
        Video(2, "misc", 0.5, 0.0, 30, tags=("gym",)),
        Video(3, "misc", 0.5, -0.9, 30),
    ]
    auditor = OpportunityAuditor(seed=3)
    logs, _, _ = _run(
        user,
        pool,
        chooser=auditor.chooser,
        seed=3,
        steps=4,
        top_k=1,
        watch_time_fn=lambda u, v, r: v.duration_s,
        enable_interest_updates=True,
        interest_topic_alpha=0.1,
        interest_tag_alpha=0.9,
        interest_decay=0.0,
    )

    summary = summarise_seed(auditor.step_rows, logs)
    assert auditor.step_rows[0]["pool_watch"] == 1
    assert auditor.step_rows[0]["pool_sample"] == 1
    assert auditor.step_rows[0]["pool_avoid"] == 1
    assert auditor.step_rows[0]["pool_sentiment_gated"] == 1
    assert auditor.step_rows[1]["pool_watch"] == 2
    assert summary["first_pool_mix_change_step"] == 1


def test_classifier_handles_empty_rows():
    assert classify_collapse_stage([]) == "no_steps"


def test_detect_runner_kind_reads_the_policy_block():
    assert detect_runner_kind({}) == "batch"
    assert detect_runner_kind({"policy": {"mode": "heuristic", "curiosity": 0.0}}) == "batch"
    assert detect_runner_kind({"policy": {"mode": "llm", "llm": {"model": "m"}}}) == "compare"
    # A dormant LLM block still marks a compare-style config.
    assert detect_runner_kind({"policy": {"mode": "heuristic", "llm": {"model": "m"}}}) == "compare"


def test_e3_direct_audit_reproduces_the_frozen_t0_counts():
    # The established risk-01 evidence, derived here from the frozen E3 corpus
    # (corpus.seed 260303) and the current policy, never from constants inside
    # the audit tool itself.
    cfg = load_config(E3_CONFIG_PATH)
    assert detect_runner_kind(cfg) == "batch"

    pool = build_corpus(cfg)
    auditor = OpportunityAuditor(seed=0)
    logs, _, _, _ = replay_heuristic_arm(cfg, pool, seed=0, steps=1, chooser=auditor.chooser)

    row = auditor.step_rows[0]
    predicted = (row["pool_predicted_watch"], row["pool_predicted_sample"], row["pool_predicted_avoid"])
    realised = (row["pool_watch"], row["pool_sample"], row["pool_avoid"])
    assert predicted == (461, 259, 280)
    assert realised == (530, 0, 470)
    assert row["pool_pred_watch_real_avoid"] == 121

    summary = summarise_seed(auditor.step_rows, logs)
    pool_t0 = summary["predicted_vs_realised"]["pool_t0"]
    assert pool_t0["total"] == 1000
    assert pool_t0["predicted"] == {"watch": 461, "sample": 259, "avoid": 280}
    assert pool_t0["realised"] == {"watch": 530, "sample": 0, "avoid": 470}
    assert pool_t0["refusals_predicted_watch_realised_avoid"] == 121
    assert pool_t0["refusal_rate"]["numerator"] == 121
    assert pool_t0["refusal_rate"]["denominator"] == 461
    assert pool_t0["refusal_rate"]["denominator_is"] == "pool_t0_predicted_watch"
    assert all(summary["reconciliation"].values())


def test_predicted_vs_realised_evidence_reconciles_for_a_hooked_watcher():
    # Watcher with an in-taste topic and a hooky off-topic tag: the platform
    # predicts Watch for both sentiment-safe videos, but the watcher refuses
    # the off-topic one, so every stage shows a predicted/realised split.
    user = User(
        phenotype=UserPhenotype.WATCHER,
        viewpoint_score=0.5,
        interest_vector={"sports": 0.9, "meme": 0.9},
        sentiment_threshold=-0.2,
    )
    pool = [
        Video(1, "sports", 0.5, 0.0, 30),
        Video(2, "misc", 0.5, 0.0, 30, tags=("meme",)),
        Video(3, "misc", 0.5, -0.9, 30),
    ]
    auditor = OpportunityAuditor(seed=5)
    logs, _, _ = _run(user, pool, chooser=auditor.chooser, seed=5, steps=8, top_k=2)

    row = auditor.step_rows[0]
    assert (row["pool_predicted_watch"], row["pool_predicted_sample"], row["pool_predicted_avoid"]) == (2, 0, 1)
    assert (row["pool_watch"], row["pool_sample"], row["pool_avoid"]) == (1, 0, 2)
    assert row["pool_pred_watch_real_avoid"] == 1
    assert row["pool_pred_real_mismatches"] == 1
    assert row["served_predicted_action"] == "Watch"

    summary = summarise_seed(auditor.step_rows, logs)
    pv = summary["predicted_vs_realised"]

    assert pv["pool_t0"]["confusion_predicted_to_realised"]["watch"] == {
        "watch": 1,
        "sample": 0,
        "avoid": 1,
    }
    assert pv["pool_t0"]["mismatches"] == 1

    # Both predicted-Watch videos fill the two slate slots, so every served
    # video is predicted Watch and refusals are exactly the realised Avoid serves.
    served = pv["served"]
    logged = summary["logged_action_counts"]
    assert served["steps"] == 8
    assert served["predicted"] == {"watch": 8, "sample": 0, "avoid": 0}
    assert served["realised"]["watch"] == logged.get("Watch", 0)
    assert served["realised"]["avoid"] == logged.get("Avoid", 0)
    assert served["refusals_predicted_watch_realised_avoid"] == served["realised"]["avoid"]
    assert served["refusal_rate"]["denominator"] == 8
    assert served["refusal_rate"]["denominator_is"] == "served_predicted_watch"

    slate = pv["slate"]
    assert slate["slots"] == 16
    assert slate["predicted"] == {"watch": 16, "sample": 0, "avoid": 0}
    assert slate["realised"] == {"watch": 8, "sample": 0, "avoid": 8}

    assert all(summary["reconciliation"].values())
