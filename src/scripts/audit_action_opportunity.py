"""
Risk-01 diagnostic: locate where non-Watch action opportunity disappears in a
heuristic run, and record the predicted-versus-realised action evidence.

The script replays the heuristic arm of a config (same corpus builder, user
builder, seeds and RNG streams as src/scripts/run_compare.py) with a purely
observational wrapper around the exposure chooser. It accepts both compare
configs (policy.llm present, e.g. configs/experiment_compare.json) and plain
heuristic configs (e.g. configs/eval/E3_explore_low.json); the config is
validated under the matching runner contract, and only the heuristic arm is
ever replayed — no LLM calls. The wrapper never draws randomness and never
mutates user or video state, so an audited run and a plain run are
behaviourally identical. The script proves that per seed, and for compare
configs it can also byte-compare the replayed step logs against the heuristic
CSVs of a frozen run.

Three stages are measured at every step, on the pre-serve user state:

    pool   -> every video in the corpus
    slate  -> the top_k candidates the ranker retains (same ordering rule as
              choose_video_weighted_top_k, recomputed without touching the RNG)
    served -> the video the chooser actually returns

Each stage records both the platform's predicted action (policy.predicted_action,
what ranking believes) and the realised action (policy.decide_action, what the
user does), plus their confusion matrix and the predicted-Watch/realised-Avoid
refusal count with an explicit denominator. In the CSV/JSON fields, plain
action names (pool_watch, slate_watch, action, served_action) are realised
actions; predicted fields are always named as such.

A collapse stage is then classified with the packet's decision rules:
pool Watch-only means the policy itself saturates; slate Watch-only with a mixed
pool means top-k selection removes the opportunity; served Watch-only with a
mixed slate means weighted selection removes it; otherwise there is no collapse.

Everything is written as CSV/JSON into an isolated diagnostics directory.
Nothing under the frozen run directory is modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from fyp_sim.agents import HeuristicDecider
from fyp_sim.artefacts import runtime_provenance
from fyp_sim.config_validation import RunnerKind, validate_experiment_config
from fyp_sim.corpus import build_corpus
from fyp_sim.models import User, UserAction, Video
from fyp_sim.policy import decide_action, interest_score, predicted_action
from fyp_sim.simulation.engine import (
    choose_video_weighted_top_k,
    run_simulation,
    video_score,
)
from src.scripts.run_compare import build_user, load_config, write_step_logs_csv

STEP_FIELDS = [
    "seed",
    "t",
    "pool_watch",
    "pool_sample",
    "pool_avoid",
    "pool_sentiment_gated",
    "pool_interest_min",
    "pool_interest_p25",
    "pool_interest_p50",
    "pool_interest_p75",
    "pool_interest_max",
    "pool_interest_mean",
    "slate_watch",
    "slate_sample",
    "slate_avoid",
    "slate_interest_min",
    "slate_interest_max",
    "slate_interest_mean",
    "interest_only_slate_watch",
    "interest_only_slate_sample",
    "interest_only_slate_avoid",
    "served_video_id",
    "served_action",
    "served_interest",
    "unique_served_so_far",
    "top_served_share_so_far",
    "pool_mix_changed_from_prev",
    # Predicted-versus-realised extension (appended to keep column order stable).
    "pool_predicted_watch",
    "pool_predicted_sample",
    "pool_predicted_avoid",
    "pool_pred_watch_real_watch",
    "pool_pred_watch_real_sample",
    "pool_pred_watch_real_avoid",
    "pool_pred_sample_real_watch",
    "pool_pred_sample_real_sample",
    "pool_pred_sample_real_avoid",
    "pool_pred_avoid_real_watch",
    "pool_pred_avoid_real_sample",
    "pool_pred_avoid_real_avoid",
    "pool_pred_real_mismatches",
    "slate_predicted_watch",
    "slate_predicted_sample",
    "slate_predicted_avoid",
    "served_predicted_action",
]

SLATE_FIELDS = [
    "seed",
    "t",
    "rank",
    "video_id",
    "score",
    "interest",
    "action",
    "sentiment_gated",
    "duration_s",
    "predicted_action",
]

PV_FIELDS = [
    "seed",
    "scope",
    "total",
    "predicted_watch",
    "predicted_sample",
    "predicted_avoid",
    "realised_watch",
    "realised_sample",
    "realised_avoid",
    "pred_watch_real_watch",
    "pred_watch_real_sample",
    "pred_watch_real_avoid",
    "pred_sample_real_watch",
    "pred_sample_real_sample",
    "pred_sample_real_avoid",
    "pred_avoid_real_watch",
    "pred_avoid_real_sample",
    "pred_avoid_real_avoid",
    "mismatches",
    "refusals_predicted_watch_realised_avoid",
    "refusal_denominator",
    "refusal_rate",
]

# Lower-case stage keys used across rows and summary blocks, in a fixed order,
# mapped to the UserAction values ("Watch"/"Sample"/"Avoid") they stand for.
_ACTION_KEYS = ("watch", "sample", "avoid")
_ACTION_VALUE = {
    "watch": UserAction.WATCH.value,
    "sample": UserAction.SAMPLE.value,
    "avoid": UserAction.AVOID.value,
}


def detect_runner_kind(cfg: dict[str, Any]) -> RunnerKind:
    """Pick the validation contract a config belongs to.

    Configs with an LLM policy block are compare-style configs (run_compare is
    their runner, and it is the only place the heuristic and LLM arms pair up).
    Everything else is a plain heuristic batch config, like the E-series evals.
    The audit itself always replays the heuristic arm only.
    """
    policy = cfg.get("policy") or {}
    mode = str(policy.get("mode", "heuristic")).strip().lower()
    if mode == "llm" or policy.get("llm") is not None:
        return "compare"
    return "batch"


def _dist_stats(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    n = len(ordered)

    def q(p: float) -> float:
        return ordered[min(n - 1, max(0, round(p * (n - 1))))]

    return {
        "min": ordered[0],
        "p25": q(0.25),
        "p50": q(0.50),
        "p75": q(0.75),
        "max": ordered[-1],
        "mean": sum(ordered) / n,
    }


def _action_counts(actions: list[UserAction]) -> dict[str, int]:
    counts = Counter(a.value for a in actions)
    return {
        "watch": counts.get(UserAction.WATCH.value, 0),
        "sample": counts.get(UserAction.SAMPLE.value, 0),
        "avoid": counts.get(UserAction.AVOID.value, 0),
    }


def _ranked_slate(
    user: User, pool: list[Video], *, top_k: int, rank_alpha: float, max_duration: int
) -> list[tuple[float, Video]]:
    # Recompute the exact ranking choose_video_weighted_top_k uses: score
    # descending, then video_id ascending. Pure reads only, no RNG.
    scored = [
        (video_score(user, v, rank_alpha=rank_alpha, max_duration=max_duration), v) for v in pool
    ]
    scored.sort(key=lambda x: (-x[0], x[1].video_id))
    return scored[: min(top_k, len(scored))]


class OpportunityAuditor:
    """Observation-only wrapper around choose_video_weighted_top_k.

    The wrapper computes pool/slate/served opportunity measures with pure reads
    before delegating the actual choice, unchanged, to the real chooser. It
    performs no RNG draws and no state mutation, so enabling it cannot alter
    the simulation.
    """

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.step_rows: list[dict[str, Any]] = []
        self.slate_rows: list[dict[str, Any]] = []
        self._t = 0
        self._served_counts: Counter[int] = Counter()
        self._prev_pool_mix: tuple[int, int, int] | None = None

    def chooser(
        self, user: User, pool: list[Video], rng, *, top_k: int, rank_alpha: float
    ) -> Video:
        row = self._observe_pre_serve(user, pool, top_k=top_k, rank_alpha=rank_alpha)

        video = choose_video_weighted_top_k(user, pool, rng, top_k=top_k, rank_alpha=rank_alpha)

        # The decider runs on this same pre-update state, so the realised
        # action here should match the action the engine logs for the step.
        # The predicted action is what the ranker believed about the same video.
        row["served_video_id"] = video.video_id
        row["served_action"] = decide_action(user, video).value
        row["served_predicted_action"] = predicted_action(user, video).value
        row["served_interest"] = round(interest_score(user, video), 6)
        self._served_counts[video.video_id] += 1
        row["unique_served_so_far"] = len(self._served_counts)
        row["top_served_share_so_far"] = round(max(self._served_counts.values()) / (self._t + 1), 6)
        self.step_rows.append(row)
        self._t += 1
        return video

    def _observe_pre_serve(
        self, user: User, pool: list[Video], *, top_k: int, rank_alpha: float
    ) -> dict[str, Any]:
        max_d = max(v.duration_s for v in pool) if pool else 0

        pool_actions = [decide_action(user, v) for v in pool]
        pool_predicted = [predicted_action(user, v) for v in pool]
        pool_interests = [interest_score(user, v) for v in pool]
        pool_counts = _action_counts(pool_actions)
        pool_predicted_counts = _action_counts(pool_predicted)
        pool_stats = _dist_stats(pool_interests)
        sentiment_gated = sum(1 for v in pool if v.sentiment_score < user.sentiment_threshold)

        # Predicted-versus-realised confusion over the whole pool, still pure reads.
        pool_confusion = Counter(
            (p.value, r.value) for p, r in zip(pool_predicted, pool_actions, strict=True)
        )
        pool_mismatches = sum(n for (p, r), n in pool_confusion.items() if p != r)

        slate = _ranked_slate(user, pool, top_k=top_k, rank_alpha=rank_alpha, max_duration=max_d)
        slate_actions = [decide_action(user, v) for _, v in slate]
        slate_predicted = [predicted_action(user, v) for _, v in slate]
        slate_interests = [interest_score(user, v) for _, v in slate]
        slate_counts = _action_counts(slate_actions)
        slate_predicted_counts = _action_counts(slate_predicted)

        # Counterfactual slate under interest-only ranking (rank_alpha=0).
        # This separates "the engagement proxy pushed Watch items up" from
        # "the interest profile alone already fills the slate".
        interest_only = _ranked_slate(user, pool, top_k=top_k, rank_alpha=0.0, max_duration=max_d)
        interest_only_counts = _action_counts([decide_action(user, v) for _, v in interest_only])

        for rank, (score, v) in enumerate(slate):
            self.slate_rows.append(
                {
                    "seed": self.seed,
                    "t": self._t,
                    "rank": rank,
                    "video_id": v.video_id,
                    "score": round(score, 6),
                    "interest": round(interest_score(user, v), 6),
                    "action": decide_action(user, v).value,
                    "sentiment_gated": int(v.sentiment_score < user.sentiment_threshold),
                    "duration_s": v.duration_s,
                    "predicted_action": predicted_action(user, v).value,
                }
            )

        pool_mix = (pool_counts["watch"], pool_counts["sample"], pool_counts["avoid"])
        changed = int(self._prev_pool_mix is not None and pool_mix != self._prev_pool_mix)
        self._prev_pool_mix = pool_mix

        row = {
            "seed": self.seed,
            "t": self._t,
            "pool_watch": pool_counts["watch"],
            "pool_sample": pool_counts["sample"],
            "pool_avoid": pool_counts["avoid"],
            "pool_sentiment_gated": sentiment_gated,
            "pool_interest_min": round(pool_stats["min"], 6),
            "pool_interest_p25": round(pool_stats["p25"], 6),
            "pool_interest_p50": round(pool_stats["p50"], 6),
            "pool_interest_p75": round(pool_stats["p75"], 6),
            "pool_interest_max": round(pool_stats["max"], 6),
            "pool_interest_mean": round(pool_stats["mean"], 6),
            "slate_watch": slate_counts["watch"],
            "slate_sample": slate_counts["sample"],
            "slate_avoid": slate_counts["avoid"],
            "slate_interest_min": round(min(slate_interests), 6),
            "slate_interest_max": round(max(slate_interests), 6),
            "slate_interest_mean": round(sum(slate_interests) / len(slate_interests), 6),
            "interest_only_slate_watch": interest_only_counts["watch"],
            "interest_only_slate_sample": interest_only_counts["sample"],
            "interest_only_slate_avoid": interest_only_counts["avoid"],
            "pool_mix_changed_from_prev": changed,
            "pool_predicted_watch": pool_predicted_counts["watch"],
            "pool_predicted_sample": pool_predicted_counts["sample"],
            "pool_predicted_avoid": pool_predicted_counts["avoid"],
            "pool_pred_real_mismatches": pool_mismatches,
            "slate_predicted_watch": slate_predicted_counts["watch"],
            "slate_predicted_sample": slate_predicted_counts["sample"],
            "slate_predicted_avoid": slate_predicted_counts["avoid"],
        }
        for p in _ACTION_KEYS:
            for r in _ACTION_KEYS:
                cell = pool_confusion.get((_ACTION_VALUE[p], _ACTION_VALUE[r]), 0)
                row[f"pool_pred_{p}_real_{r}"] = cell
        return row


def classify_collapse_stage(step_rows: list[dict[str, Any]]) -> str:
    """Apply the packet's decision rules to the measured stages."""
    if not step_rows:
        return "no_steps"
    pool_watch_only = all(r["pool_sample"] == 0 and r["pool_avoid"] == 0 for r in step_rows)
    slate_watch_only = all(r["slate_sample"] == 0 and r["slate_avoid"] == 0 for r in step_rows)
    served_watch_only = all(r["served_action"] == UserAction.WATCH.value for r in step_rows)
    if pool_watch_only:
        return "policy_pool_saturation"
    if slate_watch_only:
        return "top_k_slate_selection"
    if served_watch_only:
        return "weighted_selection_within_slate"
    return "no_watch_collapse"


def _refusal_rate(refusals: int, denominator: int, denominator_is: str) -> dict[str, Any]:
    """Refusal rate with its denominator spelled out, never a bare percentage."""
    return {
        "numerator": refusals,
        "denominator": denominator,
        "denominator_is": denominator_is,
        "rate": (refusals / denominator) if denominator else None,
    }


def _pv_block(
    predicted: dict[str, int],
    realised: dict[str, int],
    confusion: dict[str, dict[str, int]],
    *,
    total: int,
    denominator_is: str,
) -> dict[str, Any]:
    """One predicted-versus-realised evidence block (pool, slate or served)."""
    diagonal = sum(confusion[a][a] for a in _ACTION_KEYS)
    refusals = confusion["watch"]["avoid"]
    return {
        "total": total,
        "predicted": predicted,
        "realised": realised,
        "confusion_predicted_to_realised": confusion,
        "mismatches": total - diagonal,
        "refusals_predicted_watch_realised_avoid": refusals,
        "refusal_rate": _refusal_rate(refusals, predicted["watch"], denominator_is),
    }


def _served_block_from_confusion(
    confusion: dict[str, dict[str, int]], steps: int
) -> dict[str, Any]:
    predicted = {p: sum(confusion[p].values()) for p in _ACTION_KEYS}
    realised = {r: sum(confusion[p][r] for p in _ACTION_KEYS) for r in _ACTION_KEYS}
    block = _pv_block(
        predicted, realised, confusion, total=steps, denominator_is="served_predicted_watch"
    )
    return {"steps": steps, **block}


def _sum_confusions(
    confusions: list[dict[str, dict[str, int]]],
) -> dict[str, dict[str, int]]:
    return {
        p: {r: sum(c[p][r] for c in confusions) for r in _ACTION_KEYS} for p in _ACTION_KEYS
    }


def aggregate_predicted_vs_realised(pv_by_seed: list[dict[str, Any]]) -> dict[str, Any]:
    """Cross-seed aggregate of the per-seed evidence blocks.

    The t=0 pool block is deterministic (same corpus, same initial user), so it
    is reported once and flagged if any seed ever disagrees; served and slate
    blocks are summed across seeds.
    """
    pool_blocks = [pv["pool_t0"] for pv in pv_by_seed]
    pool_identical = all(block == pool_blocks[0] for block in pool_blocks)
    served_confusion = _sum_confusions(
        [pv["served"]["confusion_predicted_to_realised"] for pv in pv_by_seed]
    )
    return {
        "pool_t0_identical_across_seeds": pool_identical,
        "pool_t0": pool_blocks[0] if pool_identical else None,
        "served": _served_block_from_confusion(
            served_confusion, sum(pv["served"]["steps"] for pv in pv_by_seed)
        ),
        "slate": {
            "slots": sum(pv["slate"]["slots"] for pv in pv_by_seed),
            "predicted": {
                k: sum(pv["slate"]["predicted"][k] for pv in pv_by_seed) for k in _ACTION_KEYS
            },
            "realised": {
                k: sum(pv["slate"]["realised"][k] for pv in pv_by_seed) for k in _ACTION_KEYS
            },
        },
    }


def _predicted_vs_realised_blocks(step_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Assemble the pool_t0 / served / slate evidence blocks from step rows."""
    t0 = step_rows[0]
    pool_total = t0["pool_watch"] + t0["pool_sample"] + t0["pool_avoid"]
    pool_t0 = _pv_block(
        {k: t0[f"pool_predicted_{k}"] for k in _ACTION_KEYS},
        {k: t0[f"pool_{k}"] for k in _ACTION_KEYS},
        {p: {r: t0[f"pool_pred_{p}_real_{r}"] for r in _ACTION_KEYS} for p in _ACTION_KEYS},
        total=pool_total,
        denominator_is="pool_t0_predicted_watch",
    )

    served_pairs = Counter(
        (r["served_predicted_action"], r["served_action"]) for r in step_rows
    )
    served_confusion = {
        p: {r: served_pairs.get((_ACTION_VALUE[p], _ACTION_VALUE[r]), 0) for r in _ACTION_KEYS}
        for p in _ACTION_KEYS
    }
    served = _served_block_from_confusion(served_confusion, len(step_rows))

    slate = {
        "slots": sum(r["slate_watch"] + r["slate_sample"] + r["slate_avoid"] for r in step_rows),
        "predicted": {k: sum(r[f"slate_predicted_{k}"] for r in step_rows) for k in _ACTION_KEYS},
        "realised": {k: sum(r[f"slate_{k}"] for r in step_rows) for k in _ACTION_KEYS},
    }

    return {"pool_t0": pool_t0, "served": served, "slate": slate}


def _reconciliation_checks(
    step_rows: list[dict[str, Any]],
    pv: dict[str, Any],
    logged_counts: Counter,
) -> dict[str, bool]:
    """Prove every predicted/realised total against the row counts it came from."""
    pool_total = pv["pool_t0"]["total"]
    pool_ok = all(
        sum(r[f"pool_predicted_{k}"] for k in _ACTION_KEYS) == pool_total
        and sum(r[f"pool_{k}"] for k in _ACTION_KEYS) == pool_total
        and sum(r[f"pool_pred_{p}_real_{q}"] for p in _ACTION_KEYS for q in _ACTION_KEYS)
        == pool_total
        for r in step_rows
    )
    slate_ok = all(
        sum(r[f"slate_predicted_{k}"] for k in _ACTION_KEYS)
        == sum(r[f"slate_{k}"] for k in _ACTION_KEYS)
        for r in step_rows
    ) and pv["slate"]["slots"] == sum(pv["slate"]["realised"].values())
    steps = len(step_rows)
    served = pv["served"]
    served_ok = (
        served["steps"] == steps
        and sum(served["predicted"].values()) == steps
        and sum(served["realised"].values()) == steps
    )
    served_matches_logs = served["realised"] == {
        k: logged_counts.get(_ACTION_VALUE[k], 0) for k in _ACTION_KEYS
    }
    return {
        "pool_totals_match_pool_size": pool_ok,
        "slate_totals_match_slots": slate_ok,
        "served_totals_match_steps": served_ok,
        "served_realised_matches_engine_logs": served_matches_logs,
    }


def summarise_seed(step_rows: list[dict[str, Any]], logs: list[Any]) -> dict[str, Any]:
    steps = len(step_rows)
    pool_total = (
        step_rows[0]["pool_watch"] + step_rows[0]["pool_sample"] + step_rows[0]["pool_avoid"]
    )
    served_counts = Counter(r["served_action"] for r in step_rows)
    logged_counts = Counter(r.action for r in logs)
    first_change = next((r["t"] for r in step_rows if r["pool_mix_changed_from_prev"]), None)
    pv = _predicted_vs_realised_blocks(step_rows)
    return {
        "seed": step_rows[0]["seed"],
        "steps": steps,
        "pool_size": pool_total,
        "pool_t0": {
            "watch": step_rows[0]["pool_watch"],
            "sample": step_rows[0]["pool_sample"],
            "avoid": step_rows[0]["pool_avoid"],
            "sentiment_gated": step_rows[0]["pool_sentiment_gated"],
        },
        "pool_mean_share": {
            "watch": sum(r["pool_watch"] for r in step_rows) / (steps * pool_total),
            "sample": sum(r["pool_sample"] for r in step_rows) / (steps * pool_total),
            "avoid": sum(r["pool_avoid"] for r in step_rows) / (steps * pool_total),
        },
        "slate_watch_only_step_share": sum(
            1 for r in step_rows if r["slate_sample"] == 0 and r["slate_avoid"] == 0
        )
        / steps,
        "interest_only_slate_watch_only_step_share": sum(
            1
            for r in step_rows
            if r["interest_only_slate_sample"] == 0 and r["interest_only_slate_avoid"] == 0
        )
        / steps,
        "served_action_counts": dict(served_counts),
        "logged_action_counts": dict(logged_counts),
        "served_matches_logged": [r["served_action"] for r in step_rows]
        == [r.action for r in logs],
        "unique_served": step_rows[-1]["unique_served_so_far"],
        "top_served_share": step_rows[-1]["top_served_share_so_far"],
        "first_pool_mix_change_step": first_change,
        "collapse_stage": classify_collapse_stage(step_rows),
        "predicted_vs_realised": pv,
        "reconciliation": _reconciliation_checks(step_rows, pv, logged_counts),
    }


def _interest_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    # Mirrors the resolution in run_compare.main so the replay is exact.
    return {
        "enable_interest_updates": bool(cfg.get("enable_interest_updates", False)),
        "interest_topic_alpha": float(cfg.get("interest_topic_alpha", 0.10)),
        "interest_tag_alpha": float(cfg.get("interest_tag_alpha", 0.05)),
        "interest_decay": float(cfg.get("interest_decay", 0.02)),
        "interest_normalise": bool(cfg.get("interest_normalise", False)),
        "interest_prune_below": float(cfg.get("interest_prune_below", 0.001)),
    }


def replay_heuristic_arm(
    cfg: dict[str, Any],
    pool: list[Video],
    seed: int,
    steps: int,
    *,
    chooser,
) -> tuple[list[Any], User, Any, Any]:
    """Replay one heuristic-arm seed exactly as run_compare.main does."""
    rng = random.Random(seed)
    engagement_rng = (
        random.Random(f"{seed}:engagement")
        if bool(cfg.get("separate_rng_streams", False))
        else None
    )
    user = build_user(cfg)
    logs = run_simulation(
        user=user,
        video_pool=pool,
        steps=steps,
        rng=rng,
        top_k=int(cfg["top_k"]),
        rank_alpha=float(cfg["rank_alpha"]),
        drift_alpha=float(cfg.get("drift_alpha", cfg.get("viewpoint_drift_rate", 0.0))),
        enable_viewpoint_drift=bool(cfg.get("enable_viewpoint_drift", False)),
        decider=HeuristicDecider(),
        engagement_rng=engagement_rng,
        llm_rerank=False,
        chooser=chooser,
        **_interest_kwargs(cfg),
    )
    return logs, user, rng, engagement_rng


def verify_observer_effect(
    cfg: dict[str, Any], pool: list[Video], seed: int, steps: int
) -> tuple[bool, list[Any], OpportunityAuditor]:
    """Run the same seed with and without the auditor and require identity."""
    auditor = OpportunityAuditor(seed)
    audited_logs, audited_user, audited_rng, audited_eng = replay_heuristic_arm(
        cfg, pool, seed, steps, chooser=auditor.chooser
    )
    plain_logs, plain_user, plain_rng, plain_eng = replay_heuristic_arm(
        cfg, pool, seed, steps, chooser=choose_video_weighted_top_k
    )
    identical = (
        audited_logs == plain_logs
        and audited_rng.getstate() == plain_rng.getstate()
        and (audited_eng is None) == (plain_eng is None)
        and (audited_eng is None or audited_eng.getstate() == plain_eng.getstate())
        and audited_user.interest_vector == plain_user.interest_vector
        and audited_user.viewpoint_score == plain_user.viewpoint_score
    )
    return identical, audited_logs, auditor


def frozen_prefix_matches(frozen_csv: Path, replay_csv: Path, steps: int) -> bool:
    frozen_lines = frozen_csv.read_text(encoding="utf-8").splitlines()
    replay_lines = replay_csv.read_text(encoding="utf-8").splitlines()
    return (
        len(replay_lines) == steps + 1
        and len(frozen_lines) >= steps + 1
        and frozen_lines[: steps + 1] == replay_lines
    )


def _config_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _pv_csv_row(seed_label: str, scope: str, block: dict[str, Any]) -> dict[str, Any]:
    """Flatten one evidence block into a predicted_vs_realised.csv row.

    Slate blocks carry counts only, so their confusion and refusal columns stay
    blank; the per-candidate detail lives in action_opportunity_slate.csv.
    """
    row: dict[str, Any] = {field: "" for field in PV_FIELDS}
    row["seed"] = seed_label
    row["scope"] = scope
    row["total"] = block["total"] if "total" in block else block["slots"]
    for k in _ACTION_KEYS:
        row[f"predicted_{k}"] = block["predicted"][k]
        row[f"realised_{k}"] = block["realised"][k]
    confusion = block.get("confusion_predicted_to_realised")
    if confusion is not None:
        for p in _ACTION_KEYS:
            for r in _ACTION_KEYS:
                row[f"pred_{p}_real_{r}"] = confusion[p][r]
        rate = block["refusal_rate"]
        row["mismatches"] = block["mismatches"]
        row["refusals_predicted_watch_realised_avoid"] = block[
            "refusals_predicted_watch_realised_avoid"
        ]
        row["refusal_denominator"] = rate["denominator"]
        row["refusal_rate"] = "" if rate["rate"] is None else round(rate["rate"], 6)
    return row


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Observational action-opportunity audit of a heuristic run: the "
            "heuristic arm of a compare config, or a heuristic-mode config "
            "such as the E-series evals."
        )
    )
    p.add_argument("--config", type=Path, default=Path("configs/experiment_compare.json"))
    p.add_argument("--steps", type=int, default=75)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument(
        "--out", type=Path, default=Path("outputs/diagnostics/risk01_action_opportunity")
    )
    p.add_argument(
        "--frozen-run",
        type=Path,
        default=None,
        help="Optional compare run dir; replayed logs are byte-compared against its first N heuristic rows.",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    runner_kind = detect_runner_kind(cfg)
    audit = validate_experiment_config(cfg, runner=runner_kind, cfg_path=args.config)
    for warning in audit.warnings:
        print(f"[config warning] {warning}")

    if args.frozen_run is not None and runner_kind != "compare":
        raise SystemExit(
            "--frozen-run byte-comparison expects a compare run layout "
            "(logs/heuristic/run_seed_<seed>.csv). This config has no LLM "
            "block, so there is no compare run to match; re-run without "
            "--frozen-run."
        )

    steps = int(args.steps)
    seeds = [int(s) for s in args.seeds]
    pool = build_corpus(cfg)

    out_dir = args.out / f"{args.config.stem}_steps{steps}_seeds{'-'.join(map(str, seeds))}"
    out_dir.mkdir(parents=True, exist_ok=True)

    step_rows: list[dict[str, Any]] = []
    slate_rows: list[dict[str, Any]] = []
    per_seed: dict[str, Any] = {}
    verification: dict[str, Any] = {
        "observer_effect_identical": {},
        "frozen_prefix_match": {},
        "totals_reconcile": {},
    }
    all_ok = True

    for seed in seeds:
        identical, logs, auditor = verify_observer_effect(cfg, pool, seed, steps)
        verification["observer_effect_identical"][str(seed)] = identical
        all_ok = all_ok and identical

        replay_csv = out_dir / "replay_logs" / f"run_seed_{seed}.csv"
        write_step_logs_csv(replay_csv, logs)

        if args.frozen_run is not None:
            frozen_csv = args.frozen_run / "logs" / "heuristic" / f"run_seed_{seed}.csv"
            match = frozen_prefix_matches(frozen_csv, replay_csv, steps)
            verification["frozen_prefix_match"][str(seed)] = match
            all_ok = all_ok and match

        seed_summary = summarise_seed(auditor.step_rows, logs)
        reconciled = all(seed_summary["reconciliation"].values())
        verification["totals_reconcile"][str(seed)] = reconciled
        all_ok = all_ok and seed_summary["served_matches_logged"] and reconciled
        per_seed[str(seed)] = seed_summary
        step_rows.extend(auditor.step_rows)
        slate_rows.extend(auditor.slate_rows)

    aggregate_pv = aggregate_predicted_vs_realised(
        [per_seed[str(seed)]["predicted_vs_realised"] for seed in seeds]
    )
    all_ok = all_ok and aggregate_pv["pool_t0_identical_across_seeds"]

    stages = sorted({s["collapse_stage"] for s in per_seed.values()})
    summary = {
        "config": str(args.config),
        "config_sha256": _config_sha256(args.config),
        "runner_kind": runner_kind,
        "provenance": runtime_provenance(),
        "steps": steps,
        "seeds": seeds,
        "engine_params": {
            "top_k": int(cfg["top_k"]),
            "rank_alpha": float(cfg["rank_alpha"]),
            "enable_interest_updates": bool(cfg.get("enable_interest_updates", False)),
            "enable_viewpoint_drift": bool(cfg.get("enable_viewpoint_drift", False)),
            "separate_rng_streams": bool(cfg.get("separate_rng_streams", False)),
        },
        "config_warnings": list(audit.warnings),
        "per_seed": per_seed,
        "aggregate_collapse_stage": stages[0] if len(stages) == 1 else "mixed",
        "predicted_vs_realised_aggregate": aggregate_pv,
        "verification": verification,
    }

    _write_csv(out_dir / "action_opportunity_steps.csv", STEP_FIELDS, step_rows)
    _write_csv(out_dir / "action_opportunity_slate.csv", SLATE_FIELDS, slate_rows)

    pv_rows: list[dict[str, Any]] = []
    for seed in seeds:
        pv = per_seed[str(seed)]["predicted_vs_realised"]
        for scope in ("pool_t0", "served", "slate"):
            pv_rows.append(_pv_csv_row(str(seed), scope, pv[scope]))
    if aggregate_pv["pool_t0"] is not None:
        pv_rows.append(_pv_csv_row("all", "pool_t0", aggregate_pv["pool_t0"]))
    pv_rows.append(_pv_csv_row("all", "served", aggregate_pv["served"]))
    pv_rows.append(_pv_csv_row("all", "slate", aggregate_pv["slate"]))
    _write_csv(out_dir / "predicted_vs_realised.csv", PV_FIELDS, pv_rows)

    summary_path = out_dir / "action_opportunity_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    for seed, s in per_seed.items():
        pv = s["predicted_vs_realised"]
        pool_t0 = pv["pool_t0"]
        served = pv["served"]
        print(
            f"[seed {seed}] pool_t0 W/S/A={s['pool_t0']['watch']}/{s['pool_t0']['sample']}/"
            f"{s['pool_t0']['avoid']} slate_watch_only={s['slate_watch_only_step_share']:.2f} "
            f"served={s['logged_action_counts']} unique_served={s['unique_served']} "
            f"first_pool_mix_change={s['first_pool_mix_change_step']} stage={s['collapse_stage']}"
        )
        print(
            f"[seed {seed}] predicted_t0 W/S/A={pool_t0['predicted']['watch']}/"
            f"{pool_t0['predicted']['sample']}/{pool_t0['predicted']['avoid']} "
            f"pred_watch->real_avoid_t0="
            f"{pool_t0['refusals_predicted_watch_realised_avoid']}/"
            f"{pool_t0['refusal_rate']['denominator']} "
            f"served_refusals={served['refusals_predicted_watch_realised_avoid']}/"
            f"{served['refusal_rate']['denominator']}"
        )
    print(f"[verdict] collapse_stage={summary['aggregate_collapse_stage']}")
    agg_served = aggregate_pv["served"]
    print(
        f"[predicted-vs-realised] served refusals across seeds: "
        f"{agg_served['refusals_predicted_watch_realised_avoid']}/"
        f"{agg_served['refusal_rate']['denominator']} "
        f"(denominator = served predicted-Watch)"
    )
    print(f"[verification] {json.dumps(verification)}")
    print(f"Wrote audit artefacts to: {out_dir}")

    if not all_ok:
        raise SystemExit("Audit verification failed; see verification block above.")


if __name__ == "__main__":
    main()
