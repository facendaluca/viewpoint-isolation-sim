from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from fyp_sim.agents.deciders import HeuristicDecider
from fyp_sim.artefacts import _fail_fast_old_alpha
from fyp_sim.corpus import build_corpus
from fyp_sim.models import User, UserPhenotype
from fyp_sim.simulation.engine import run_simulation


def phenotype_from_str(s: str) -> UserPhenotype:
    s = s.strip().lower()
    if s == "watcher":
        return UserPhenotype.WATCHER
    if s == "sampler":
        return UserPhenotype.SAMPLER
    if s == "avoider":
        return UserPhenotype.AVOIDER
    raise ValueError(f"Unknown phenotype: {s!r} (expected watcher/sampler/avoider)")


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def build_user(cfg: dict[str, Any]) -> User:
    u = cfg["user"]
    return User(
        phenotype=phenotype_from_str(u["phenotype"]),
        viewpoint_score=float(u["viewpoint_score"]),
        interest_vector={str(k): float(v) for k, v in u["interest_vector"].items()},
        sentiment_threshold=float(u["sentiment_threshold"]),
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Smoke-check viewpoint drift on a single seed.")
    p.add_argument("config", nargs="?", type=Path, default=Path("configs/experiment_baseline.json"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--steps", type=int, default=None, help="Override config steps for this smoke run."
    )
    p.add_argument("--enable-drift", action="store_true")
    p.add_argument("--drift-rate", type=float, default=0.2)
    p.add_argument("--rows", type=int, default=12)
    args = p.parse_args()

    cfg = load_config(args.config)
    _fail_fast_old_alpha(cfg, args.config)
    steps = int(cfg["steps"]) if args.steps is None else int(args.steps)

    user = build_user(cfg)
    pool = build_corpus(cfg)
    rng = random.Random(args.seed)

    logs = run_simulation(
        user=user,
        video_pool=pool,
        steps=steps,
        rng=rng,
        top_k=int(cfg["top_k"]),
        rank_alpha=float(cfg["rank_alpha"]),
        drift_alpha=float(cfg.get("drift_alpha", cfg.get("viewpoint_drift_rate", 0.0))),
        decider=HeuristicDecider(),
        enable_interest_updates=bool(cfg.get("enable_interest_updates", False)),
        interest_topic_alpha=float(cfg.get("interest_topic_alpha", 0.10)),
        interest_tag_alpha=float(cfg.get("interest_tag_alpha", 0.05)),
        interest_decay=float(cfg.get("interest_decay", 0.02)),
        interest_normalise=bool(cfg.get("interest_normalise", False)),
        interest_prune_below=float(cfg.get("interest_prune_below", 0.001)),
        enable_viewpoint_drift=args.enable_drift,
        viewpoint_drift_rate=args.drift_rate,
    )

    print("t  vid  act    vii_t   vp_pre  vp_post  v_vp")
    print("-- ---- ------ ------ ------- ------- -----")
    for row in logs[: args.rows]:
        print(
            f"{row.t:2d} {row.video_id:4d} {row.action:6s} "
            f"{row.vii_t:6.3f} {row.user_viewpoint_pre:7.3f} "
            f"{row.user_viewpoint_post:7.3f} {row.video_viewpoint_score:5.3f} "
        )

    if args.enable_drift and args.drift_rate > 0.0:
        checked = 0
        violations = 0
        for row in logs:
            if row.action.lower() not in ("watch", "sample"):
                continue
            checked += 1
            d_pre = abs(row.video_viewpoint_score - row.user_viewpoint_pre)
            d_post = abs(row.video_viewpoint_score - row.user_viewpoint_post)
            if d_post > d_pre:
                violations += 1

        print(
            f"\nSanity: monotonic pull holds for {checked - violations}/{checked} "
            f"WATCH/SAMPLE steps (violations={violations})."
        )


if __name__ == "__main__":
    main()
