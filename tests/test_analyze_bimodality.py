from __future__ import annotations

import csv
import json
from pathlib import Path

from src.scripts.analyze_bimodality import (
    analyse_run,
    identical_context_flips,
    regime_separation,
    seed_branch_diagnostics,
)


def _row(
    t: int,
    video_id: int,
    action: str,
    *,
    llm_action: str | None = None,
    hash_pre: str = "h0",
    hash_post: str | None = None,
    vp: str = "0.4000",
):
    return {
        "t": t,
        "video_id": video_id,
        "action": action,
        "llm_action": action if llm_action is None else llm_action,
        "llm_valid": "True",
        "llm_fallback_reason": "",
        "interest_state_hash_pre": hash_pre,
        "interest_state_hash_post": hash_pre if hash_post is None else hash_post,
        "user_viewpoint_pre": vp,
        "user_viewpoint_post": vp,
    }


def _pure_seed_rows() -> list[dict]:
    # Samples everywhere except one Watch at t=2, which updates the state.
    # 20 of 21 actions are Sample, above the 0.95 near-pure threshold.
    rows = [_row(0, 10, "Sample"), _row(1, 11, "Sample")]
    rows.append(_row(2, 12, "Watch", hash_post="h1"))
    rows += [_row(t, 10, "Sample", hash_pre="h1") for t in range(3, 21)]
    return rows


def _mixed_seed_rows() -> list[dict]:
    rows = [_row(0, 10, "Watch", hash_post="h1")]
    rows += [
        _row(t, 20 + t, "Watch", hash_pre=f"h{t}", hash_post=f"h{t + 1}") for t in range(1, 6)
    ]
    rows += [_row(6, 30, "Sample", hash_pre="h6"), _row(7, 31, "Avoid", hash_pre="h6")]
    return rows


def test_seed_branch_diagnostics_finds_first_events_and_regime():
    diag = seed_branch_diagnostics(_pure_seed_rows(), block_size=4)

    assert diag["steps"] == 21
    assert diag["action_counts"] == {"Watch": 1, "Sample": 20, "Avoid": 0}
    assert diag["regime"] == "near_pure_sample"
    assert diag["first_actions"]["Watch"] == {"t": 2, "video_id": 12, "action": "Watch"}
    assert diag["first_actions"]["Sample"]["t"] == 0
    assert diag["first_actions"]["Avoid"] is None
    assert diag["first_interest_update"]["t"] == 2
    assert diag["first_state_divergence_from_initial"]["t"] == 3
    assert diag["watch_events"] == [{"t": 2, "video_id": 12}]
    assert len(diag["block_action_rates"]) == 6
    assert diag["block_action_rates"][0]["watch_rate"] == 0.25
    assert diag["block_action_rates"][1]["sample_rate"] == 1.0
    assert diag["unique_videos"] == 3

    mixed = seed_branch_diagnostics(_mixed_seed_rows(), block_size=4)
    assert mixed["regime"] == "mixed"
    assert mixed["watch_rate"] == 0.75


def test_identical_context_flips_within_and_across_seeds():
    seed_a = [
        _row(0, 5, "Sample"),
        # Same video, same hash, same viewpoint later in the run, but the LLM
        # answered differently: a within-seed flip.
        _row(5, 5, "Watch"),
    ]
    seed_b = [_row(0, 5, "Watch"), _row(1, 6, "Sample", hash_pre="hx")]
    flips = identical_context_flips({0: seed_a, 1: seed_b})

    assert flips["within_seed"]["count"] == 1
    assert flips["within_seed"]["entries"][0]["seed"] == 0
    assert flips["cross_seed"]["count"] == 1
    occurrences = flips["cross_seed"]["entries"][0]["occurrences_by_llm_action"]
    assert sorted(occurrences) == ["Sample", "Watch"]
    # The t=0 view catches the exact-context cross-seed disagreement.
    assert flips["t0_cross_seed"]["count"] == 1
    t0_actions = flips["t0_cross_seed"]["entries"][0]["occurrences_by_llm_action"]
    assert t0_actions["Sample"] == [[0, 0]]
    assert t0_actions["Watch"] == [[1, 0]]


def test_regime_separation_earliest_stable_t():
    pure = [_row(t, 10, "Sample") for t in range(6)]
    mixed = [_row(0, 10, "Sample")] + [_row(t, 11, "Watch") for t in range(1, 6)]
    result = regime_separation(
        {0: pure, 1: mixed},
        {0: "near_pure_sample", 1: "mixed"},
        checkpoints=(0, 1, 5),
    )

    assert result["near_pure_sample_seeds"] == [0]
    assert result["mixed_seeds"] == [1]
    # Cumulative watches: pure stays 0, mixed reaches 1 at t=1 and keeps growing.
    assert result["earliest_stable_separation_t"] == 1
    assert result["cumulative_watch_at_checkpoints"]["1"] == {"0": 0, "1": 1, "5": 5}


def test_analyse_run_end_to_end(tmp_path: Path):
    run_dir = tmp_path / "run"
    logs_dir = run_dir / "logs" / "llm"
    logs_dir.mkdir(parents=True)
    header = [
        "t",
        "video_id",
        "action",
        "llm_action",
        "llm_valid",
        "llm_fallback_reason",
        "interest_state_hash_pre",
        "interest_state_hash_post",
        "user_viewpoint_pre",
        "user_viewpoint_post",
    ]
    for seed, rows in ((0, _pure_seed_rows()), (1, _mixed_seed_rows())):
        with (logs_dir / f"run_seed_{seed}.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)

    summary_fields = ["agent", "seed", "watch_rate", "sample_rate", "mean_vii", "lock_in_rate", "unique_videos_seen"]
    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerow(dict(zip(summary_fields, ["llm", 0, 0.125, 0.875, 0.07, 1.0, 3], strict=True)))
        writer.writerow(dict(zip(summary_fields, ["llm", 1, 0.75, 0.125, 0.15, 0.2, 8], strict=True)))
        writer.writerow(
            dict(zip(summary_fields, ["heuristic", 0, 1.0, 0.0, 0.12, 0.0, 9], strict=True))
        )

    result = analyse_run(run_dir)

    assert set(result["per_seed"]) == {"0", "1"}
    assert result["per_seed"]["0"]["regime"] == "near_pure_sample"
    assert result["per_seed"]["1"]["regime"] == "mixed"
    summary = result["regime_summary"]
    assert summary["near_pure_sample"]["seeds"] == [0]
    assert summary["mixed"]["watch_rate"]["mean"] == 0.75
    assert json.dumps(result)  # payload must be JSON-serialisable
