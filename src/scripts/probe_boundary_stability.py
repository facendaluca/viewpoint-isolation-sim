"""
Boundary action-stability probe: send the *same* decision prompt to the live
endpoint several times and count how often the action changes.

The target contexts are step-0 compare contexts (freshly built user + a named
corpus video), where the frozen run already shows different actions across
seeds for an identical prompt. This probe measures that instability directly
with a small, hard-capped number of live calls. It never mutates any state and
writes its evidence to a standalone JSON file.

Since the request-seed change every repeat re-sends the identical logical
request with the SAME derived sampling seed (probe stream, call_role="probe"),
so an action flip now measures residual server/runtime nondeterminism under a
pinned seed rather than raw repeat noise. Historical probe outputs written
before request seeding measured the unpinned behaviour and are not comparable.

Usage:
    python -m src.scripts.probe_boundary_stability \
        --config configs/experiment_compare.json --video-ids 871 431 --repeats 12
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path
from urllib.request import urlopen

from fyp_sim.corpus import build_corpus
from fyp_sim.llm.prompting import render_decision_prompt
from fyp_sim.policy import interest_score
from fyp_sim.simulation.engine import _interest_state_hash
from src.scripts.run_compare import build_llm_decider, build_user, load_config, write_json

MAX_LIVE_CALLS = 30


def endpoint_available(base_url: str, timeout_s: float = 3.0) -> bool:
    try:
        with urlopen(base_url.rstrip("/") + "/models", timeout=timeout_s):
            return True
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment_compare.json"))
    parser.add_argument("--video-ids", type=int, nargs="+", required=True)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/analysis/risk02_bimodality/boundary_probe.json"),
    )
    args = parser.parse_args()

    planned_calls = len(args.video_ids) * args.repeats
    if planned_calls > MAX_LIVE_CALLS:
        raise SystemExit(
            f"Refusing to run: {planned_calls} planned calls exceed the "
            f"{MAX_LIVE_CALLS}-call probe cap."
        )

    cfg = load_config(args.config)
    config_sha = hashlib.sha256(args.config.read_bytes()).hexdigest()
    llm_cfg = (cfg.get("policy") or {}).get("llm") or {}
    base_url = str(llm_cfg.get("base_url", ""))
    if not endpoint_available(base_url):
        raise SystemExit(f"Endpoint unavailable at {base_url}; made 0 LLM calls.")

    user = build_user(cfg)
    pool = {v.video_id: v for v in build_corpus(cfg)}
    prompt_id = str(llm_cfg.get("prompt_id", "decision_v1"))
    decider = build_llm_decider(cfg)

    results = {}
    calls_made = 0
    for video_id in args.video_ids:
        video = pool[video_id]
        prompt = render_decision_prompt(prompt_id, user=user, video=video)
        actions: list[str] = []
        details = []
        for _ in range(args.repeats):
            # Same identity every repeat on purpose: the repeats re-send one
            # logical request, so they share one derived sampling seed.
            decider.set_request_context(
                experiment_seed=0,
                agent_id="user",
                stream="probe",
                step=0,
                call_role="probe",
                draw_index=video_id,
            )
            action = decider.decide_next_action(user, video)
            meta = decider.last_meta
            calls_made += 1
            actions.append(action.value)
            details.append(
                {
                    "action": action.value,
                    "llm_action_raw": meta.llm_action,
                    "valid": meta.valid,
                    "fallback_reason": meta.fallback_reason,
                    "confidence": meta.llm_confidence,
                    "total_tokens": meta.total_tokens,
                    "request_seed": meta.request_seed,
                    "request_seed_sent": meta.request_seed_sent,
                    "response_sha256": meta.response_sha256,
                }
            )
        counts: dict[str, int] = {}
        for value in actions:
            counts[value] = counts.get(value, 0) + 1
        results[str(video_id)] = {
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "interest_state_hash": _interest_state_hash(user),
            "user_viewpoint": float(user.viewpoint_score),
            "video_topic": video.topic_category,
            "video_tags": list(video.tags),
            "interest_score": float(interest_score(user, video)),
            "action_counts": counts,
            "distinct_actions": sorted(counts),
            "unstable": len(counts) > 1,
            "calls": details,
        }

    payload = {
        "config": str(args.config),
        "config_sha256": config_sha,
        "model": str(llm_cfg.get("model", "")),
        "prompt_id": prompt_id,
        "temperature": float(llm_cfg.get("temperature", 0.0)),
        "repeats_per_context": args.repeats,
        "llm_call_count": calls_made,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "contexts": results,
    }
    write_json(args.out, payload)
    print(f"Wrote boundary probe to: {args.out} (live LLM calls: {calls_made})")
    for video_id, entry in results.items():
        print(f"video {video_id}: {entry['action_counts']} unstable={entry['unstable']}")


if __name__ == "__main__":
    main()
