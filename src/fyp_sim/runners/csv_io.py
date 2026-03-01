from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from fyp_sim.simulation.engine import StepLog


def write_run_log_csv(
    path: Path, logs: Iterable[StepLog], *, include_viewpoint: bool = False
) -> None:
    """Write per-step logs for one seed to CSV (schema matches existing scripts)."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as f:
        w = csv.writer(f)

        header = [
            "t",
            "video_id",
            "action",
            "watch_time_s",
            "interest",
            "topic_interest",
            "vii_t",
            "vii_cum",
        ]
        if include_viewpoint:
            header += ["user_viewpoint_pre", "user_viewpoint_post", "video_viewpoint_score"]
        w.writerow(header)

        for row in logs:
            base = [
                row.t,
                row.video_id,
                row.action,
                row.watch_time_s,
                f"{row.interest:.4f}",
                f"{row.topic_interest:.4f}",
                f"{row.vii_t:.4f}",
                f"{row.vii_cum:.4f}",
            ]
            if include_viewpoint:
                base += [
                    f"{row.user_viewpoint_pre:.4f}",
                    f"{row.user_viewpoint_post:.4f}",
                    f"{row.video_viewpoint_score:.4f}",
                ]
            w.writerow(base)


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write summary.csv from per-seed row dicts."""
    if not rows:
        raise ValueError("rows must not be empty")

    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
