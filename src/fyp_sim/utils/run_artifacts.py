"""
Utility functions for managing run artifacts, manifests, and structured logging.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


def slugify(text: str) -> str:
    """Normalise text to a safe slug path component."""
    s = text.lower().strip()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_-]+", "_", s)
    return s.strip("_")


def canonical_json_bytes(obj: Any) -> bytes:
    """Serialize object to canonical, consistent JSON bytes."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def stable_config_hash(config: dict[str, Any]) -> str:
    """Compute a stable SHA256 hash of a configuration dictionary."""
    return hashlib.sha256(canonical_json_bytes(config)).hexdigest()


def make_run_id(
    exp_name: str, config_slug: str, corpus_mode: str, base_seed: int, n_runs: int, hash8: str
) -> str:
    """Generate a deterministic, self-describing run ID."""
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    return (
        f"{timestamp}__{exp_name}__{config_slug}__{corpus_mode}__"
        f"seed{base_seed}__n{n_runs}__{hash8}"
    )


def write_json(path: Path, obj: Any) -> None:
    """Atomic write of JSON to path."""
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    tmp.replace(path)


def write_manifest(output_root: Path, manifest: dict[str, Any]) -> None:
    """Write or update the manifest.json file."""
    write_json(output_root / "manifest.json", manifest)
