"""
Candidate-level matched-context trace for LLM rerank runs.

During slate reranking the decider is asked about every candidate against one
frozen pre-update user state. For each of those already-paid LLM calls the
heuristic action can be computed locally on the exact same (user, video)
context, which gives a paired policy comparison with zero extra LLM calls.

The collector is strictly passive: recording never mutates user, video, RNG,
or decider state, so trace-on and trace-off runs produce identical
simulations. The timestep-zipped arm-to-arm comparison in run_compare stays
separate; that one is an unpaired architecture divergence, not a matched
policy disagreement.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

_ACTIONS = ("Watch", "Sample", "Avoid")

# Predeclared bands, fixed before looking at matched outcomes. Interest edges
# mirror the frozen policy thresholds (0.2 watcher floor, 0.5 bracket,
# 0.7 sampler watch bar); sentiment and duration bands follow the corpus
# design (sentiment mass sits at -1/0/+1, durations are short-form seconds).
_INTEREST_BAND_EDGES = ((0.2, "interest_lt_0.20"), (0.5, "interest_0.20_0.50"), (0.7, "interest_0.50_0.70"))
_INTEREST_BAND_TOP = "interest_ge_0.70"


@dataclass(slots=True)
class CandidateRecord:
    """One decider call about one slate candidate, plus its heuristic shadow.

    Every field describes the same frozen pre-update context: the user state
    the LLM was prompted with is the state the heuristic shadow action was
    computed on.
    """

    seed: int
    t: int
    slate_rank: int  # 1-based position in the heuristic first-pass ranking
    video_id: int
    selected: bool
    interest_state_hash_pre: str
    user_viewpoint_pre: float
    topic: str
    tags: str  # "|"-joined
    video_viewpoint: float
    video_sentiment: float
    duration_s: int
    interest: float
    heuristic_action: str
    llm_action: str  # action used by the rerank (fallback already resolved)
    llm_action_raw: str  # raw model action, "" when the call fell back
    llm_valid: bool
    llm_fallback_reason: str
    llm_confidence: float | None
    heuristic_score: float  # first-pass score with the heuristic engagement proxy
    llm_engagement: float  # engagement proxy of the decider's action
    rerank_score: float  # final score used for weighted selection


@dataclass(slots=True)
class CandidateTraceCollector:
    """Optional sink the engine fills with one CandidateRecord per decider call."""

    seed: int
    rows: list[CandidateRecord] = field(default_factory=list)
    steps_recorded: int = 0

    def next_t(self, trace_t: int | None) -> int:
        return self.steps_recorded if trace_t is None else int(trace_t)

    def record_step(self, records: list[CandidateRecord]) -> None:
        self.rows.extend(records)
        self.steps_recorded += 1


_CSV_COLUMNS = [f.name for f in fields(CandidateRecord)]


def write_candidate_trace_csv(path: Path, rows: list[CandidateRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(_CSV_COLUMNS)
        for row in rows:
            out = []
            for name in _CSV_COLUMNS:
                value = getattr(row, name)
                if isinstance(value, bool):
                    value = int(value)
                elif value is None:
                    value = ""
                out.append(value)
            writer.writerow(out)


def interest_band(interest: float) -> str:
    for edge, label in _INTEREST_BAND_EDGES:
        if interest < edge:
            return label
    return _INTEREST_BAND_TOP


def sentiment_band(sentiment: float) -> str:
    if sentiment < 0.0:
        return "sentiment_negative"
    if sentiment > 0.0:
        return "sentiment_positive"
    return "sentiment_neutral"


def duration_band(duration_s: int) -> str:
    if duration_s <= 30:
        return "duration_le_30s"
    if duration_s <= 90:
        return "duration_31_90s"
    return "duration_gt_90s"


def _rate(part: int, whole: int) -> float:
    return part / whole if whole else 0.0


def _bucket_stats(rows: list[CandidateRecord]) -> dict[str, int | float]:
    disagreements = sum(r.llm_action != r.heuristic_action for r in rows)
    return {
        "rows": len(rows),
        "disagreement_rows": disagreements,
        "disagreement_rate": _rate(disagreements, len(rows)),
    }


def _grouped_stats(
    rows: list[CandidateRecord], key_fn
) -> dict[str, dict[str, int | float]]:
    groups: dict[str, list[CandidateRecord]] = {}
    for row in rows:
        groups.setdefault(str(key_fn(row)), []).append(row)
    return {key: _bucket_stats(groups[key]) for key in sorted(groups)}


def summarise_candidate_rows(
    rows: list[CandidateRecord], *, expected_rows: int | None = None
) -> dict[str, Any]:
    """Aggregate matched-context diagnostics over a set of candidate rows.

    Every row is a matched context by construction (the heuristic shadow was
    computed on the very state the LLM was prompted with), so
    matched_context_rows equals observed_rows.
    """
    observed = len(rows)
    disagreement_rows = [r for r in rows if r.llm_action != r.heuristic_action]

    confusion = {h: dict.fromkeys(_ACTIONS, 0) for h in _ACTIONS}
    for row in rows:
        confusion[row.heuristic_action][row.llm_action] += 1
    confusion_total = sum(sum(col.values()) for col in confusion.values())

    fallback_rows = [r for r in rows if not r.llm_valid]
    fallback_reasons: dict[str, int] = {}
    for row in fallback_rows:
        reason = row.llm_fallback_reason or "unknown"
        fallback_reasons[reason] = fallback_reasons.get(reason, 0) + 1
    valid_rows = [r for r in rows if r.llm_valid]
    valid_disagreements = sum(r.llm_action != r.heuristic_action for r in valid_rows)

    selected_rows = [r for r in rows if r.selected]
    unselected_rows = [r for r in rows if not r.selected]

    heuristic_counts = dict.fromkeys(_ACTIONS, 0)
    llm_counts = dict.fromkeys(_ACTIONS, 0)
    for row in rows:
        heuristic_counts[row.heuristic_action] += 1
        llm_counts[row.llm_action] += 1

    return {
        "expected_rows": expected_rows,
        "observed_rows": observed,
        "rows_equal_llm_calls": (
            expected_rows == observed if expected_rows is not None else None
        ),
        "matched_context_rows": observed,
        "disagreement_rows": len(disagreement_rows),
        "disagreement_rate": _rate(len(disagreement_rows), observed),
        "valid_rows": len(valid_rows),
        "valid_disagreement_rows": valid_disagreements,
        "valid_disagreement_rate": _rate(valid_disagreements, len(valid_rows)),
        "fallback_rows": len(fallback_rows),
        # Fallback rows resolve to the heuristic action, so any disagreement
        # here would indicate a context mismatch inside the trace itself.
        "fallback_disagreement_rows": sum(
            r.llm_action != r.heuristic_action for r in fallback_rows
        ),
        "fallback_reasons": fallback_reasons,
        "confusion_heuristic_rows_llm_cols": confusion,
        "confusion_total": confusion_total,
        "confusion_total_equals_rows": confusion_total == observed,
        "heuristic_action_counts": heuristic_counts,
        "llm_action_counts": llm_counts,
        "selected": _bucket_stats(selected_rows),
        "unselected": _bucket_stats(unselected_rows),
        "by_slate_rank": _grouped_stats(rows, lambda r: r.slate_rank),
        "by_interest_band": _grouped_stats(rows, lambda r: interest_band(r.interest)),
        "by_sentiment_band": _grouped_stats(
            rows, lambda r: sentiment_band(r.video_sentiment)
        ),
        "by_duration_band": _grouped_stats(
            rows, lambda r: duration_band(r.duration_s)
        ),
    }


def matched_policy_diagnostics(
    rows_by_seed: dict[int, list[CandidateRecord]],
    llm_calls_by_seed: dict[int, int],
) -> dict[str, Any]:
    """Build the per-seed and aggregate matched-context diagnostics payload."""
    per_seed = []
    all_rows: list[CandidateRecord] = []
    for seed in sorted(rows_by_seed):
        rows = rows_by_seed[seed]
        all_rows.extend(rows)
        per_seed.append(
            {
                "seed": seed,
                **summarise_candidate_rows(
                    rows, expected_rows=llm_calls_by_seed.get(seed)
                ),
            }
        )

    expected_total = (
        sum(llm_calls_by_seed[s] for s in rows_by_seed if s in llm_calls_by_seed)
        if llm_calls_by_seed
        else None
    )
    aggregate = summarise_candidate_rows(all_rows, expected_rows=expected_total)

    return {
        "comparison_type": "matched_candidate_context",
        "note": (
            "Each row pairs the LLM action with the heuristic action computed on the "
            "exact pre-update (user, video) context of that rerank call; the trace adds "
            "zero LLM calls. This is the paired policy comparison; the timestep-zipped "
            "arm comparison in comparison_diagnostics.json is unpaired architecture "
            "divergence."
        ),
        "gates": {
            "rows_equal_llm_calls": bool(aggregate["rows_equal_llm_calls"])
            and all(entry["rows_equal_llm_calls"] for entry in per_seed),
            "confusion_totals_equal_rows": bool(aggregate["confusion_total_equals_rows"])
            and all(entry["confusion_total_equals_rows"] for entry in per_seed),
        },
        "per_seed": per_seed,
        "aggregate": aggregate,
    }
