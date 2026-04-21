from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from fyp_sim.examiner_dashboard.backend import run_heuristic, run_heuristic_real

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    PLACEHOLDER = "placeholder"
    REAL = "real"


class ProgressSink(Protocol):
    """UI-facing progress sink (Streamlit implementation lives elsewhere)."""

    def on_seed_start(self, *, i: int, n: int, seed: int) -> None: ...
    def on_done(self) -> None: ...


@dataclass(frozen=True)
class RunResult:
    run_dir: Path
    mode: ExecutionMode


def execute_run(
    *,
    resolved_cfg: dict[str, Any],
    cfg_path: str | None,
    mode: ExecutionMode,
    progress: ProgressSink | None = None,
) -> RunResult:
    """
    Execute a run via the dashboard backend.

    UI passes a ProgressSink; the runner calls it per seed (real mode only).
    """
    logger.info("Simulation run starting: cfg=%s mode=%s", cfg_path, mode.value)

    if mode is ExecutionMode.PLACEHOLDER:
        run_dir = run_heuristic(resolved_cfg, cfg_path)
        return RunResult(run_dir=run_dir, mode=mode)

    # Real simulation mode
    def cb(i: int, n: int, seed: int) -> None:
        if progress is not None:
            progress.on_seed_start(i=i, n=n, seed=seed)

    run_dir = run_heuristic_real(resolved_cfg, cfg_path=cfg_path, progress_cb=cb)
    if progress is not None:
        progress.on_done()
    logger.info("Simulation run complete: run_dir=%s", run_dir)
    return RunResult(run_dir=run_dir, mode=mode)
