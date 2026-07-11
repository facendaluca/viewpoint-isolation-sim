"""
Branch-point diagnostics for bimodal LLM compare runs. Offline only: reads the
per-step CSV logs of an existing compare run and makes zero LLM calls.

For each LLM-arm seed it reports where the trajectory first committed (first
Watch/Sample/Avoid, first interest update, first divergence from the initial
interest state), how the action mix evolves over fixed time blocks, and which
regime the seed ends in. Across seeds it looks for identical pre-action
contexts that received different LLM actions (same interest-state hash, same
logged viewpoint, same video), which separates inference instability from
seeded feedback divergence.

Usage:
    python -m src.scripts.analyze_bimodality --run-dir outputs/real/experiment_compare/<...>
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

# A seed counts as near-pure Sample when at least this fraction of its selected
# actions are Sample. Declared up front; the frozen run's regimes sit at
# >= 0.9867 and <= 0.4333, far from this edge.
PURE_SAMPLE_THRESHOLD = 0.95
BLOCK_SIZE = 30
CUMULATIVE_WATCH_CHECKPOINTS = (0, 1, 2, 3, 5, 10, 15, 20, 30, 50, 75, 100, 149)

_ACTIONS = ("Watch", "Sample", "Avoid")


def load_llm_seed_logs(run_dir: Path) -> dict[int, list[dict[str, Any]]]:
    logs_dir = run_dir / "logs" / "llm"
    rows_by_seed: dict[int, list[dict[str, Any]]] = {}
    for path in sorted(logs_dir.glob("run_seed_*.csv")):
        seed = int(path.stem.rsplit("_", 1)[1])
        with path.open(encoding="utf-8") as fh:
            rows = []
            for raw in csv.DictReader(fh):
                rows.append(
                    {
                        "t": int(raw["t"]),
                        "video_id": int(raw["video_id"]),
                        "action": raw["action"],
                        "llm_action": raw.get("llm_action", ""),
                        "llm_valid": raw.get("llm_valid", ""),
                        "llm_fallback_reason": raw.get("llm_fallback_reason", ""),
                        "interest_state_hash_pre": raw.get("interest_state_hash_pre", ""),
                        "interest_state_hash_post": raw.get("interest_state_hash_post", ""),
                        # Kept as the logged 4-decimal string so grouping states
                        # the precision it actually has.
                        "user_viewpoint_pre": raw.get("user_viewpoint_pre", ""),
                        "user_viewpoint_post": raw.get("user_viewpoint_post", ""),
                    }
                )
        rows_by_seed[seed] = rows
    if not rows_by_seed:
        raise FileNotFoundError(f"No LLM seed logs found under {logs_dir}")
    return rows_by_seed


def seed_branch_diagnostics(
    rows: list[dict[str, Any]],
    *,
    block_size: int = BLOCK_SIZE,
    pure_sample_threshold: float = PURE_SAMPLE_THRESHOLD,
) -> dict[str, Any]:
    """Per-seed branch-point summary of one LLM-arm trajectory."""
    steps = len(rows)
    counts = dict.fromkeys(_ACTIONS, 0)
    for row in rows:
        counts[row["action"]] += 1

    def first_event(predicate) -> dict[str, Any] | None:
        for row in rows:
            if predicate(row):
                return {"t": row["t"], "video_id": row["video_id"], "action": row["action"]}
        return None

    first_actions = {
        action: first_event(lambda r, a=action: r["action"] == a) for action in _ACTIONS
    }
    first_interest_update = first_event(
        lambda r: r["interest_state_hash_pre"] != r["interest_state_hash_post"]
    )
    initial_hash = rows[0]["interest_state_hash_pre"] if rows else ""
    first_state_divergence = first_event(
        lambda r: r["interest_state_hash_pre"] != initial_hash
    )

    blocks = []
    for start in range(0, steps, block_size):
        chunk = rows[start : start + block_size]
        block_counts = dict.fromkeys(_ACTIONS, 0)
        for row in chunk:
            block_counts[row["action"]] += 1
        blocks.append(
            {
                "t_start": start,
                "t_end": start + len(chunk) - 1,
                **{
                    f"{action.lower()}_rate": block_counts[action] / len(chunk)
                    for action in _ACTIONS
                },
            }
        )

    sample_rate = counts["Sample"] / steps if steps else 0.0
    last_block = rows[-block_size:] if steps else []

    return {
        "steps": steps,
        "action_counts": counts,
        "watch_rate": counts["Watch"] / steps if steps else 0.0,
        "sample_rate": sample_rate,
        "avoid_rate": counts["Avoid"] / steps if steps else 0.0,
        "regime": "near_pure_sample" if sample_rate >= pure_sample_threshold else "mixed",
        "first_actions": first_actions,
        "first_interest_update": first_interest_update,
        "first_state_divergence_from_initial": first_state_divergence,
        "watch_events": [
            {"t": r["t"], "video_id": r["video_id"]} for r in rows if r["action"] == "Watch"
        ],
        "block_action_rates": blocks,
        "unique_videos": len({r["video_id"] for r in rows}),
        "unique_videos_last_block": len({r["video_id"] for r in last_block}),
        "llm_fallback_steps": sum(1 for r in rows if r["llm_fallback_reason"]),
        "final_user_viewpoint_post": rows[-1]["user_viewpoint_post"] if rows else "",
    }


def _context_key(row: dict[str, Any]) -> tuple[int, str, str]:
    return (row["video_id"], row["interest_state_hash_pre"], row["user_viewpoint_pre"])


def identical_context_flips(
    rows_by_seed: dict[int, list[dict[str, Any]]], *, max_entries: int = 100
) -> dict[str, Any]:
    """Find identical pre-action contexts that received different LLM actions.

    Context identity means same video, same interest-state hash, and same logged
    pre-step viewpoint. The decision prompt is a pure function of (user, video),
    so two rows with the same key rendered the same prompt, up to the 4-decimal
    precision of the logged viewpoint. At t=0 identity is exact by construction:
    every seed starts from the same freshly built user.
    """

    def collect(rows_iter) -> dict[tuple[int, str, str], dict[str, list[Any]]]:
        groups: dict[tuple[int, str, str], dict[str, list[Any]]] = {}
        for seed, row in rows_iter:
            actions = groups.setdefault(_context_key(row), {})
            actions.setdefault(row["llm_action"], []).append([seed, row["t"]])
        return groups

    def flips(groups) -> list[dict[str, Any]]:
        out = []
        for (video_id, state_hash, viewpoint), actions in groups.items():
            if len(actions) > 1:
                out.append(
                    {
                        "video_id": video_id,
                        "interest_state_hash_pre": state_hash,
                        "user_viewpoint_pre": viewpoint,
                        "occurrences_by_llm_action": actions,
                    }
                )
        out.sort(key=lambda e: (e["video_id"], e["user_viewpoint_pre"]))
        return out

    within_seed = []
    for seed in sorted(rows_by_seed):
        groups = collect((seed, row) for row in rows_by_seed[seed])
        for entry in flips(groups):
            within_seed.append({"seed": seed, **entry})

    all_rows = ((seed, row) for seed in sorted(rows_by_seed) for row in rows_by_seed[seed])
    cross_seed = flips(collect(all_rows))

    t0_rows = (
        (seed, row)
        for seed in sorted(rows_by_seed)
        for row in rows_by_seed[seed]
        if row["t"] == 0
    )
    t0_cross_seed = flips(collect(t0_rows))

    def cap(entries: list) -> dict[str, Any]:
        return {
            "count": len(entries),
            "entries": entries[:max_entries],
            "truncated": len(entries) > max_entries,
        }

    return {
        "note": (
            "Only the selected candidate's LLM action is visible in step logs, so "
            "these flips are a lower bound; the candidate-level trace removes this "
            "blind spot for future runs. Viewpoint equality is at the logged "
            "4-decimal precision except at t=0, where contexts are exact."
        ),
        "within_seed": cap(within_seed),
        "cross_seed": cap(cross_seed),
        "t0_cross_seed": cap(t0_cross_seed),
    }


def regime_separation(
    rows_by_seed: dict[int, list[dict[str, Any]]],
    regimes: dict[int, str],
    *,
    checkpoints: tuple[int, ...] = CUMULATIVE_WATCH_CHECKPOINTS,
) -> dict[str, Any]:
    """Earliest timestep from which the two regimes stay separated.

    Separation is judged on cumulative selected-Watch counts: the earliest t such
    that every mixed seed strictly exceeds every near-pure-Sample seed at t and at
    every later step.
    """
    cumulative: dict[int, list[int]] = {}
    for seed, rows in rows_by_seed.items():
        counter = 0
        series = []
        for row in rows:
            counter += row["action"] == "Watch"
            series.append(counter)
        cumulative[seed] = series

    pure = [s for s, r in regimes.items() if r == "near_pure_sample"]
    mixed = [s for s, r in regimes.items() if r == "mixed"]
    steps = min((len(v) for v in cumulative.values()), default=0)

    earliest = None
    if pure and mixed and steps:
        separated = [
            max(cumulative[s][t] for s in pure) < min(cumulative[s][t] for s in mixed)
            for t in range(steps)
        ]
        for t in range(steps):
            if all(separated[t:]):
                earliest = t
                break

    return {
        "near_pure_sample_seeds": sorted(pure),
        "mixed_seeds": sorted(mixed),
        "earliest_stable_separation_t": earliest,
        "cumulative_watch_at_checkpoints": {
            str(seed): {str(t): cumulative[seed][t] for t in checkpoints if t < len(cumulative[seed])}
            for seed in sorted(cumulative)
        },
    }


def regime_summary_from_summary_csv(run_dir: Path, regimes: dict[int, str]) -> dict[str, Any]:
    """Stratify the run's headline LLM-arm metrics by regime instead of pooling."""
    summary_path = run_dir / "summary.csv"
    if not summary_path.exists():
        return {"available": False}

    with summary_path.open(encoding="utf-8") as fh:
        llm_rows = [r for r in csv.DictReader(fh) if r.get("agent") == "llm"]

    metrics = ("watch_rate", "sample_rate", "mean_vii", "lock_in_rate", "unique_videos_seen")
    out: dict[str, Any] = {"available": True}
    for regime in sorted(set(regimes.values())):
        seeds = [s for s, r in regimes.items() if r == regime]
        rows = [r for r in llm_rows if int(r["seed"]) in seeds]
        stats = {}
        for metric in metrics:
            values = [float(r[metric]) for r in rows]
            stats[metric] = {
                "mean": sum(values) / len(values) if values else None,
                "min": min(values) if values else None,
                "max": max(values) if values else None,
            }
        out[regime] = {"seeds": sorted(seeds), "n": len(rows), **stats}
    return out


def analyse_run(run_dir: Path) -> dict[str, Any]:
    rows_by_seed = load_llm_seed_logs(run_dir)
    per_seed = {
        seed: seed_branch_diagnostics(rows) for seed, rows in sorted(rows_by_seed.items())
    }
    regimes = {seed: diag["regime"] for seed, diag in per_seed.items()}
    return {
        "run_dir": str(run_dir),
        "arm": "llm",
        "pure_sample_threshold": PURE_SAMPLE_THRESHOLD,
        "block_size": BLOCK_SIZE,
        "per_seed": {str(seed): per_seed[seed] for seed in sorted(per_seed)},
        "regime_summary": regime_summary_from_summary_csv(run_dir, regimes),
        "regime_separation": regime_separation(rows_by_seed, regimes),
        "identical_context_flips": identical_context_flips(rows_by_seed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path (default: outputs/analysis/risk02_bimodality/<run_id>/branch_diagnostics.json)",
    )
    args = parser.parse_args()

    result = analyse_run(args.run_dir)

    out_path = args.out
    if out_path is None:
        out_path = (
            Path("outputs/analysis/risk02_bimodality")
            / args.run_dir.name
            / "branch_diagnostics.json"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)

    print(f"Wrote branch diagnostics to: {out_path}")
    for seed_key in sorted(result["per_seed"], key=int):
        diag = result["per_seed"][seed_key]
        first_watch = diag["first_actions"]["Watch"]
        print(
            f"seed {seed_key}: regime={diag['regime']} watch_rate={diag['watch_rate']:.4f} "
            f"first_watch_t={first_watch['t'] if first_watch else None} "
            f"first_update_t={diag['first_interest_update']['t'] if diag['first_interest_update'] else None} "
            f"unique_videos={diag['unique_videos']}"
        )
    flips = result["identical_context_flips"]
    print(
        f"identical-context flips: within-seed={flips['within_seed']['count']} "
        f"cross-seed={flips['cross_seed']['count']} t0={flips['t0_cross_seed']['count']}"
    )
    print(
        "earliest stable regime separation t="
        f"{result['regime_separation']['earliest_stable_separation_t']}"
    )


if __name__ == "__main__":
    main()
