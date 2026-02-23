from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


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


def create_run_artefacts(
    cfg: dict[str, Any],
    cfg_path: Path | None,
    mode: str,
    seeds: list[int],
    outputs_root: Path,
    *,
    write_config_snapshot: bool = True,
) -> RunArtefacts:
    """Create a run directory + manifest for an experiment execution.

    Notes:
        - Simulation determinism is unchanged; run_id is unique per execution.
        - The date is a parent folder; run_id stays short.
    """
    ts = _utc_now()
    date_ymd = ts.strftime("%Y%m%d")
    time_hms = ts.strftime("%H%M%S")
    cfg_hash = _cfg_hash(cfg)
    hash8 = cfg_hash[:8]

    run_id = f"{time_hms}Z_{mode}_{hash8}"

    root_dir = outputs_root / date_ymd / run_id
    seeds_dir = root_dir / "seeds"
    plots_dir = root_dir / "plots"
    manifest_path = root_dir / "manifest.json"
    summary_path = root_dir / "summary.csv"

    seeds_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    key_params: dict[str, Any] = {
        "steps": cfg.get("steps"),
        "top_k": cfg.get("top_k"),
        "alpha": cfg.get("alpha"),
        "enable_interest_updates": cfg.get("enable_interest_updates"),
        "interest_topic_alpha": cfg.get("interest_topic_alpha"),
        "interest_tag_alpha": cfg.get("interest_tag_alpha"),
        "interest_decay": cfg.get("interest_decay"),
        "interest_normalise": cfg.get("interest_normalise"),
        "interest_prune_below": cfg.get("interest_prune_below"),
    }

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "date_ymd": date_ymd,
        "mode": mode,
        "cfg_path": str(cfg_path) if cfg_path is not None else None,
        "cfg_hash": cfg_hash,
        "seeds": seeds,
        "key_params": key_params,
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
