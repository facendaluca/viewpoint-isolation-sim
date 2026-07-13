"""
Fixed-request LM Studio reproducibility check for the request-seed change.

Sends ONE fixed decision request (same prompt, same derived sampling seed)
several times and records SHA-256 hashes of the raw response and the parsed
result. Run it twice — once normally, then again with --phase post_reload
after unloading and reloading the same model in LM Studio — and it compares
the two phases. That bounds the reproducibility claim empirically: identical
hashes demonstrate the seed and sampling configuration pin the output for
this model file, runtime version and hardware; any mismatch is measured
server/runtime nondeterminism, not simulation noise.

This is a small fixture only (hard-capped live calls), never the production
experiment.

Usage:
    # Phase 1: baseline within one model load
    python -m src.scripts.llm_repro_check --config configs/exploratory/experiment_compare_watcher_taste.json

    # Phase 2: after unloading/reloading the model in LM Studio
    python -m src.scripts.llm_repro_check --config configs/exploratory/experiment_compare_watcher_taste.json \
        --phase post_reload
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from fyp_sim.llm.prompting import render_decision_prompt
from fyp_sim.llm.request_seed import REQUEST_SEED_SCHEMA_VERSION, derive_request_seed
from src.scripts.probe_boundary_stability import endpoint_available
from src.scripts.run_compare import build_llm_decider, build_user, load_config, write_json

MAX_LIVE_CALLS = 12
DEFAULT_OUT_DIR = Path("outputs/diagnostics/llm_repro_check")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment_compare.json"))
    parser.add_argument("--video-id", type=int, default=None, help="Fixture video (default: first corpus video).")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--phase",
        choices=("within_load", "post_reload"),
        default="within_load",
        help="within_load writes the baseline; post_reload compares against it.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    if args.repeats > MAX_LIVE_CALLS:
        raise SystemExit(f"Refusing to run: {args.repeats} repeats exceed the {MAX_LIVE_CALLS}-call cap.")

    cfg = load_config(args.config)
    llm_cfg = (cfg.get("policy") or {}).get("llm") or {}
    base_url = str(llm_cfg.get("base_url", ""))
    if not endpoint_available(base_url):
        raise SystemExit(f"Endpoint unavailable at {base_url}; made 0 LLM calls.")

    # Fixture: deterministic user/video/prompt from the config, plus one fixed
    # derived seed. The identity is constant on purpose — every call in every
    # phase re-sends the same logical request.
    from fyp_sim.corpus import build_corpus  # local import keeps startup cheap

    user = build_user(cfg)
    pool = build_corpus(cfg)
    video = pool[0] if args.video_id is None else {v.video_id: v for v in pool}[args.video_id]
    prompt_id = str(llm_cfg.get("prompt_id", "decision_v1"))
    prompt = render_decision_prompt(prompt_id, user=user, video=video)
    request_seed = derive_request_seed(
        experiment_seed=0,
        agent_id="user",
        step=0,
        call_role="probe",
        draw_index=int(video.video_id),
        stream="repro_check",
    )

    decider = build_llm_decider(cfg)
    client = decider.client

    calls = []
    for index in range(args.repeats):
        raw = client.complete(prompt, timeout_s=float(llm_cfg.get("timeout_s", 15.0)), request_seed=request_seed)
        payload = client.last_request_payload or {}
        meta = {
            "index": index,
            "raw_sha256": _sha256(raw),
            "seed_in_request_body": payload.get("seed"),
            "response_model": client.last_response_model,
        }
        try:
            from fyp_sim.agents.deciders import _extract_first_json_object
            from fyp_sim.llm.decision_contract import parse_decision_json

            decision = parse_decision_json(_extract_first_json_object(raw))
            meta["parsed_action"] = decision.action.value
            meta["parsed_confidence"] = decision.confidence
            meta["parsed_sha256"] = _sha256(
                json.dumps(
                    {"action": decision.action.value, "confidence": decision.confidence},
                    sort_keys=True,
                )
            )
            meta["parsed_valid"] = True
        except Exception as error:
            meta["parsed_valid"] = False
            meta["parse_error"] = type(error).__name__
        calls.append(meta)

    raw_hashes = sorted({c["raw_sha256"] for c in calls})
    parsed_hashes = sorted({c.get("parsed_sha256", "") for c in calls})
    seeds_in_body = sorted({c["seed_in_request_body"] for c in calls})
    result = {
        "phase": args.phase,
        "config": str(args.config),
        "prompt_id": prompt_id,
        "prompt_sha256": _sha256(prompt),
        "video_id": int(video.video_id),
        "model_requested": str(llm_cfg.get("model", "")),
        "model_reported_by_server": sorted({str(c.get("response_model")) for c in calls}),
        "request_seed": request_seed,
        "request_seed_schema_version": REQUEST_SEED_SCHEMA_VERSION,
        "effective_sampling": client.effective_sampling(),
        "repeats": args.repeats,
        "seed_present_in_every_request_body": seeds_in_body == [request_seed],
        "within_phase_raw_identical": len(raw_hashes) == 1,
        "within_phase_parsed_identical": len(parsed_hashes) == 1,
        "distinct_raw_hashes": raw_hashes,
        "distinct_parsed_hashes": parsed_hashes,
        "calls": calls,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = args.out_dir / "within_load.json"
    out_path = args.out_dir / f"{args.phase}.json"

    if args.phase == "post_reload":
        if not baseline_path.exists():
            raise SystemExit("No within_load.json baseline found; run --phase within_load first.")
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        result["matches_within_load_raw"] = (
            baseline.get("distinct_raw_hashes") == result["distinct_raw_hashes"]
        )
        result["matches_within_load_parsed"] = (
            baseline.get("distinct_parsed_hashes") == result["distinct_parsed_hashes"]
        )

    write_json(out_path, result)
    print(f"Wrote {args.phase} reproducibility evidence to: {out_path}")
    for key in (
        "seed_present_in_every_request_body",
        "within_phase_raw_identical",
        "within_phase_parsed_identical",
        "matches_within_load_raw",
        "matches_within_load_parsed",
    ):
        if key in result:
            print(f"  {key}: {result[key]}")


if __name__ == "__main__":
    main()
