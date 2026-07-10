from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from fyp_sim.simulation.engine import StepLog


def write_run_log_csv(
    path: Path,
    logs: Iterable[StepLog],
    *,
    include_viewpoint: bool = False,
    include_agent_id: bool = False,
    include_llm_meta: bool = False,
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

        if include_agent_id:
            header = ["agent_id", *header]

        if include_viewpoint:
            header += ["user_viewpoint_pre", "user_viewpoint_post", "video_viewpoint_score"]

        if include_llm_meta:
            header += [
                "policy_mode",
                "llm_prompt_id",
                "llm_valid",
                "llm_fallback_reason",
                "llm_action",
                "llm_confidence",
            ]
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

            if include_agent_id:
                base = [row.agent_id, *base]

            if include_viewpoint:
                base += [
                    f"{row.user_viewpoint_pre:.4f}",
                    f"{row.user_viewpoint_post:.4f}",
                    f"{row.video_viewpoint_score:.4f}",
                ]

            if include_llm_meta:
                # llm_* columns stay blank for heuristic rows (matches run_compare CSVs)
                is_llm = row.policy_mode == "llm"
                base += [
                    row.policy_mode,
                    row.llm_prompt_id if is_llm else "",
                    row.llm_valid if is_llm else "",
                    row.llm_fallback_reason if is_llm else "",
                    row.llm_action if is_llm else "",
                    row.llm_confidence if is_llm and row.llm_confidence is not None else "",
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
