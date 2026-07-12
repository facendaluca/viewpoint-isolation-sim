from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fyp_sim.models import Video
from fyp_sim.policy import POLICY_CONTRACT


@dataclass(frozen=True)
class RunArtefacts:
    """Immutable artefact metadata."""

    run_id: str
    date_ymd: str
    root_dir: Path
    seeds_dir: Path
    plots_dir: Path
    manifest_path: Path
    summary_path: Path


def _cfg_hash(cfg: dict[str, Any]) -> str:
    """Stable hash of the resolved config dict (independent of key order)."""
    payload = json.dumps(cfg, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _git_provenance(anchor: Path) -> tuple[str | None, bool | None]:
    """Best-effort (commit, dirty) of the git worktree containing `anchor`.

    Returns (None, None) when git is missing, times out, or the path is not
    inside a repository, so callers never fail on provenance collection.
    """
    try:
        head = subprocess.run(
            ["git", "-C", str(anchor), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if head.returncode != 0 or not head.stdout.strip():
            return None, None
        status = subprocess.run(
            ["git", "-C", str(anchor), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
        return head.stdout.strip(), dirty
    except (OSError, subprocess.SubprocessError):
        return None, None


def runtime_provenance() -> dict[str, Any]:
    """Concise record of which source code produced an artefact.

    cfg_hash cannot separate runs whose configs are identical but whose policy
    semantics differ (the risk-01 watcher change), so manifests also record the
    source commit, whether the worktree had uncommitted changes, and the policy
    contract version. Output naming must never depend on these fields.
    """
    commit, dirty = _git_provenance(Path(__file__).resolve().parent)
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "policy_contract": POLICY_CONTRACT,
    }


def _corpus_payload(videos: list[Video]) -> list[dict[str, Any]]:
    """Canonical JSON payload for a corpus.

    Deterministic (sorted, stable key ordering) so a sha256 hash can prove "same corpus" across runs/configs.

    IMPORTANT: This does not mutate the in-memory order used by the simulation.
    """
    ordered = sorted(
        videos,
        key=lambda v: (
            int(v.video_id),
            str(v.topic_category),
            float(v.viewpoint_score),
            float(v.sentiment_score),
            int(v.duration_s),
            tuple(v.tags),
        ),
    )

    out: list[dict[str, Any]] = []
    for v in ordered:
        out.append(
            {
                "video_id": int(v.video_id),
                "topic_category": str(v.topic_category),
                "viewpoint_score": float(v.viewpoint_score),
                "sentiment_score": float(v.sentiment_score),
                "duration_s": int(v.duration_s),
                "tags": list(v.tags),
            }
        )
    return out


def _sha256_canonical_json(obj: Any) -> str:
    """Stable sha256 over a canonical JSON representation."""
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def _fail_fast_old_alpha(cfg: dict[str, Any], cfg_path: Path | None) -> None:
    if "alpha" in cfg:
        where = f" in {cfg_path}" if cfg_path is not None else ""
        raise ValueError(
            "Config key 'alpha' has been split to avoid ambiguity. "
            f"Update your config{where} to use:\n"
            "    - rank_alpha (ranking / exposure scoring)\n"
            "    - drift_alpha (viewpoint drift strength; optional unless drift enabled)\n"
            "If you previously used 'alpha' for ranking, rename it to 'rank_alpha'."
        )


def create_run_artefacts(
    cfg: dict[str, Any],
    cfg_path: Path | None,
    mode: str,
    seeds: list[int],
    outputs_root: Path,
    *,
    corpus: list[Video] | None = None,
    runner: str | None = None,
    write_config_snapshot: bool = True,
) -> RunArtefacts:
    """Create a run directory + manifest for an experiment execution.

    Notes:
        - Simulation determinism is unchanged; run_id is unique per execution.
        - The date is a parent folder; run_id stays short.
    """
    _fail_fast_old_alpha(cfg, cfg_path)
    ts = _utc_now()
    date_ymd = ts.strftime("%Y%m%d")
    time_hms = ts.strftime("%H%M%S")
    cfg_hash = _cfg_hash(cfg)
    hash8 = cfg_hash[:8]

    base_run_id = f"{time_hms}Z_{mode}_{hash8}"
    date_dir = outputs_root / date_ymd
    date_dir.mkdir(parents=True, exist_ok=True)
    collision_index = 0
    while True:
        run_id = base_run_id if collision_index == 0 else f"{base_run_id}_{collision_index:02d}"
        root_dir = date_dir / run_id
        try:
            root_dir.mkdir(exist_ok=False)
            break
        except FileExistsError:
            collision_index += 1

    seeds_dir = root_dir / "seeds"
    plots_dir = root_dir / "plots"
    manifest_path = root_dir / "manifest.json"
    summary_path = root_dir / "summary.csv"

    seeds_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    key_params: dict[str, Any] = {
        "steps": cfg.get("steps"),
        "top_k": cfg.get("top_k"),
        "rank_alpha": cfg.get("rank_alpha"),
        "drift_alpha": cfg.get("drift_alpha"),
        "enable_interest_updates": cfg.get("enable_interest_updates"),
        "interest_topic_alpha": cfg.get("interest_topic_alpha"),
        "interest_tag_alpha": cfg.get("interest_tag_alpha"),
        "interest_decay": cfg.get("interest_decay"),
        "interest_normalise": cfg.get("interest_normalise"),
        "interest_prune_below": cfg.get("interest_prune_below"),
        "enable_viewpoint_drift": cfg.get("enable_viewpoint_drift"),
        "separate_rng_streams": cfg.get("separate_rng_streams"),
        "lock_in_threshold": cfg.get("lock_in_threshold"),
        "persistence_window": cfg.get("persistence_window"),
        "top_k_grid": cfg.get("top_k_grid"),
        "rank_alpha_grid": cfg.get("rank_alpha_grid"),
    }

    policy_cfg = cfg.get("policy", {}) or {}
    llm_cfg = policy_cfg.get("llm", {}) or {}
    key_params["policy_mode"] = policy_cfg.get("mode", "heuristic")
    key_params["llm_model"] = llm_cfg.get("model")
    key_params["llm_prompt_id"] = llm_cfg.get("prompt_id")
    key_params["llm_rerank_slate"] = llm_cfg.get("rerank_slate")

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "date_ymd": date_ymd,
        "mode": mode,
        "runner": runner,
        "cfg_path": str(cfg_path) if cfg_path is not None else None,
        "cfg_hash": cfg_hash,
        "seeds": seeds,
        "key_params": key_params,
        "collision_index": collision_index,
        "provenance": runtime_provenance(),
    }

    if corpus is not None:
        corpus_rel = "corpus.json"
        corpus_path = root_dir / corpus_rel

        payload = _corpus_payload(corpus)
        corpus_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        corpus_cfg = cfg.get("corpus", {}) or {}
        manifest["corpus"] = {
            "path": corpus_rel,
            "hash": _sha256_canonical_json(payload),
            "n_videos": len(corpus),
            "source": corpus_cfg.get("source", "file"),
            "seed": corpus_cfg.get("seed"),
        }

    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if write_config_snapshot:
        (root_dir / "config_resolved.json").write_text(
            json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    return RunArtefacts(
        run_id=run_id,
        date_ymd=date_ymd,
        root_dir=root_dir,
        seeds_dir=seeds_dir,
        plots_dir=plots_dir,
        manifest_path=manifest_path,
        summary_path=summary_path,
    )
