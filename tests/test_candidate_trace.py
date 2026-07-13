from __future__ import annotations

import csv
import random
from dataclasses import fields
from pathlib import Path

from fyp_sim.agents.deciders import LLMDecider
from fyp_sim.candidate_trace import (
    CandidateRecord,
    CandidateTraceCollector,
    matched_policy_diagnostics,
    summarise_candidate_rows,
    write_candidate_trace_csv,
)
from fyp_sim.models import User, UserAction, UserPhenotype, Video
from fyp_sim.policy import decide_action
from fyp_sim.simulation.engine import run_simulation


class ScriptedDecider:
    """Fixed action per video_id; counts every call like an LLM decider would."""

    def __init__(self, actions: dict[int, UserAction], default: UserAction = UserAction.SAMPLE):
        self.actions = actions
        self.default = default
        self.asked: list[int] = []
        self.last_meta = None

    def decide_next_action(self, user: User, video: Video) -> UserAction:  # noqa: ARG002
        self.asked.append(video.video_id)
        return self.actions.get(video.video_id, self.default)


def _user() -> User:
    return User(
        phenotype=UserPhenotype.WATCHER,
        viewpoint_score=0.5,
        interest_vector={"sports": 0.9, "news": 0.8, "music": 0.7, "travel": 0.1},
        sentiment_threshold=-1.0,
    )


def _pool() -> list[Video]:
    topics = ["sports", "news", "music", "travel", "travel", "travel"]
    return [
        Video(i, topic, 0.2 + 0.1 * i, 0.5 - 0.3 * (i % 3), 20 + 25 * i, tags=("meme",))
        for i, topic in enumerate(topics, start=1)
    ]


def _run(collector, *, seed: int = 7, steps: int = 6):
    decider = ScriptedDecider({1: UserAction.WATCH, 2: UserAction.AVOID})
    logs = run_simulation(
        user=_user(),
        video_pool=_pool(),
        steps=steps,
        rng=random.Random(seed),
        engagement_rng=random.Random(f"{seed}:engagement"),
        top_k=3,
        rank_alpha=0.5,
        drift_alpha=0.05,
        enable_viewpoint_drift=True,
        enable_interest_updates=True,
        decider=decider,
        llm_rerank=True,
        candidate_trace=collector,
    )
    return logs, decider


def test_trace_rows_equal_decider_calls_with_ranks_and_timesteps():
    collector = CandidateTraceCollector(seed=7)
    _, decider = _run(collector, steps=6)

    assert len(collector.rows) == len(decider.asked) == 3 * 6
    assert [r.t for r in collector.rows] == [t for t in range(6) for _ in range(3)]
    assert [r.slate_rank for r in collector.rows] == [1, 2, 3] * 6
    assert [r.video_id for r in collector.rows] == decider.asked


def test_trace_rows_use_exact_pre_update_context_and_mark_selection():
    collector = CandidateTraceCollector(seed=7)
    logs, _ = _run(collector, steps=6)

    for log in logs:
        step_rows = [r for r in collector.rows if r.t == log.t]
        assert len(step_rows) == 3
        for row in step_rows:
            # The state the decider was prompted with is the state the step log
            # captured before any interest update or drift.
            assert row.interest_state_hash_pre == log.interest_state_hash_pre
            assert row.user_viewpoint_pre == log.user_viewpoint_pre
        selected = [r for r in step_rows if r.selected]
        assert len(selected) == 1
        assert selected[0].video_id == log.video_id
        assert selected[0].llm_action == log.action

    # Interest updates happened, so pre-contexts must move over time; the trace
    # must reflect the evolving state, not the initial one.
    hashes = {r.interest_state_hash_pre for r in collector.rows}
    assert len(hashes) > 1


def test_trace_on_and_off_runs_are_identical_including_rng_streams():
    rng_off, rng_on = random.Random(7), random.Random(7)
    eng_off, eng_on = random.Random("7:e"), random.Random("7:e")
    kwargs = dict(
        video_pool=_pool(),
        steps=8,
        top_k=3,
        rank_alpha=0.5,
        drift_alpha=0.05,
        enable_viewpoint_drift=True,
        enable_interest_updates=True,
        llm_rerank=True,
    )

    logs_off = run_simulation(
        user=_user(),
        rng=rng_off,
        engagement_rng=eng_off,
        decider=ScriptedDecider({1: UserAction.WATCH, 2: UserAction.AVOID}),
        candidate_trace=None,
        **kwargs,
    )
    logs_on = run_simulation(
        user=_user(),
        rng=rng_on,
        engagement_rng=eng_on,
        decider=ScriptedDecider({1: UserAction.WATCH, 2: UserAction.AVOID}),
        candidate_trace=CandidateTraceCollector(seed=7),
        **kwargs,
    )

    assert logs_on == logs_off
    assert rng_on.getstate() == rng_off.getstate()
    assert eng_on.getstate() == eng_off.getstate()


def test_fallback_rows_resolve_to_the_heuristic_shadow_action():
    collector = CandidateTraceCollector(seed=0)
    user = _user()
    run_simulation(
        user=user,
        video_pool=_pool(),
        steps=3,
        rng=random.Random(0),
        top_k=3,
        rank_alpha=0.5,
        decider=LLMDecider(prompt_id="decision_v1", client=None),
        llm_rerank=True,
        candidate_trace=collector,
    )

    assert len(collector.rows) == 9
    for row in collector.rows:
        assert row.llm_valid is False
        assert row.llm_fallback_reason == "no_client"
        assert row.llm_action_raw == ""
        # A fallback answers with the heuristic policy on the same context, so
        # it can never disagree with the shadow action.
        assert row.llm_action == row.heuristic_action

    summary = summarise_candidate_rows(collector.rows, expected_rows=9)
    assert summary["rows_equal_llm_calls"] is True
    assert summary["disagreement_rows"] == 0
    assert summary["fallback_rows"] == 9
    assert summary["fallback_disagreement_rows"] == 0
    assert summary["fallback_reasons"] == {"no_client": 9}


def test_shadow_action_matches_policy_on_the_recorded_context():
    collector = CandidateTraceCollector(seed=7)
    _run(collector, steps=4)
    user = _user()  # states at t=0 are identical to a fresh user
    pool = {v.video_id: v for v in _pool()}
    for row in [r for r in collector.rows if r.t == 0]:
        assert row.heuristic_action == decide_action(user, pool[row.video_id]).value


def _record(**overrides) -> CandidateRecord:
    base = dict(
        seed=0,
        t=0,
        slate_rank=1,
        video_id=1,
        selected=False,
        interest_state_hash_pre="h0",
        user_viewpoint_pre=0.4,
        topic="comedy_memes",
        tags="meme",
        video_viewpoint=0.5,
        video_sentiment=0.0,
        duration_s=30,
        interest=0.3,
        heuristic_action="Sample",
        llm_action="Sample",
        llm_action_raw="Sample",
        llm_valid=True,
        llm_fallback_reason="",
        llm_confidence=0.8,
        heuristic_score=0.5,
        llm_engagement=0.2,
        rerank_score=0.5,
    )
    base.update(overrides)
    return CandidateRecord(**base)


def test_summarise_candidate_rows_math():
    rows = [
        _record(slate_rank=1, selected=True),
        _record(
            slate_rank=2,
            video_id=2,
            interest=0.75,
            video_sentiment=-1.0,
            duration_s=120,
            heuristic_action="Sample",
            llm_action="Watch",
            llm_action_raw="Watch",
        ),
        _record(slate_rank=3, video_id=3, interest=0.1, duration_s=45),
    ]
    summary = summarise_candidate_rows(rows, expected_rows=3)

    assert summary["observed_rows"] == summary["matched_context_rows"] == 3
    assert summary["rows_equal_llm_calls"] is True
    assert summary["disagreement_rows"] == 1
    assert summary["disagreement_rate"] == 1 / 3
    confusion = summary["confusion_heuristic_rows_llm_cols"]
    assert confusion["Sample"]["Watch"] == 1
    assert confusion["Sample"]["Sample"] == 2
    assert summary["confusion_total"] == 3
    assert summary["confusion_total_equals_rows"] is True
    assert summary["selected"] == {"rows": 1, "disagreement_rows": 0, "disagreement_rate": 0.0}
    assert summary["unselected"]["rows"] == 2
    assert summary["unselected"]["disagreement_rows"] == 1
    assert summary["by_slate_rank"]["2"]["disagreement_rows"] == 1
    assert summary["by_interest_band"]["interest_ge_0.70"]["disagreement_rows"] == 1
    assert summary["by_interest_band"]["interest_lt_0.20"]["rows"] == 1
    assert summary["by_sentiment_band"]["sentiment_negative"]["rows"] == 1
    assert summary["by_duration_band"]["duration_gt_90s"]["disagreement_rows"] == 1

    mismatch = summarise_candidate_rows(rows, expected_rows=4)
    assert mismatch["rows_equal_llm_calls"] is False


def test_matched_policy_diagnostics_gates_and_seed_split():
    rows_by_seed = {
        0: [_record(), _record(slate_rank=2, video_id=2)],
        1: [_record(seed=1, llm_action="Watch", llm_action_raw="Watch")],
    }
    payload = matched_policy_diagnostics(rows_by_seed, {0: 2, 1: 1})

    assert payload["comparison_type"] == "matched_candidate_context"
    assert payload["gates"] == {
        "rows_equal_llm_calls": True,
        "confusion_totals_equal_rows": True,
    }
    assert [entry["seed"] for entry in payload["per_seed"]] == [0, 1]
    assert payload["aggregate"]["expected_rows"] == 3
    assert payload["aggregate"]["observed_rows"] == 3
    assert payload["aggregate"]["disagreement_rows"] == 1

    broken = matched_policy_diagnostics(rows_by_seed, {0: 2, 1: 5})
    assert broken["gates"]["rows_equal_llm_calls"] is False


def test_candidate_trace_csv_roundtrip(tmp_path: Path):
    rows = [_record(selected=True), _record(slate_rank=2, video_id=2, llm_confidence=None)]
    out = tmp_path / "trace.csv"
    write_candidate_trace_csv(out, rows)

    with out.open(encoding="utf-8") as fh:
        parsed = list(csv.reader(fh))
    header, data = parsed[0], parsed[1:]
    assert header == [f.name for f in fields(CandidateRecord)]
    assert len(data) == 2
    assert data[0][header.index("selected")] == "1"
    assert data[1][header.index("llm_confidence")] == ""
