from __future__ import annotations

import statistics
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchmarkParams:
    """Common benchmark parameters."""

    steps: int
    n_videos: int
    seed: int
    top_k: int
    rank_alpha: float
    repeats: int
    threshold_s: float
    outputs_root: Path

    def validate(self) -> None:
        if self.steps <= 0:
            raise ValueError("--steps must be > 0")
        if self.n_videos <= 0:
            raise ValueError("--n-videos must be > 0")
        if self.seed < 0:
            raise ValueError("--seed must be >= 0")
        if self.top_k <= 0:
            raise ValueError("--top-k must be > 0")
        if self.repeats <= 0:
            raise ValueError("--repeats must be > 0")
        if self.threshold_s <= 0.0:
            raise ValueError("--threshold-s must be > 0.0")
        if not (0.0 < self.rank_alpha < 1.0):
            raise ValueError("--rank-alpha must be between 0.0 and 1.0")


@dataclass(frozen=True)
class TimingStats:
    timings_s: list[float]
    mean_s: float
    best_s: float
    worst_s: float
    passed: bool

    @staticmethod
    def from_timings(timings_s: list[float], *, threshold_s: float) -> TimingStats:
        if not timings_s:
            raise ValueError("No timings recorded")
        mean_s = float(statistics.fmean(timings_s))
        best_s = float(min(timings_s))
        worst_s = float(max(timings_s))
        passed = bool(mean_s < threshold_s)
        return TimingStats(
            timings_s=timings_s, mean_s=mean_s, best_s=best_s, worst_s=worst_s, passed=passed
        )


class ConsoleCapture:
    """Prints to console while also capturing the exact output for stdout.txt."""

    def __init__(self) -> None:
        self._lines: list[str] = []

    def emit(self, line: str = "") -> None:
        print(line)
        self._lines.append(line)

    def to_text(self) -> str:
        return "\n".join(self._lines) + "\n"


def format_os_text(specs: Mapping[str, Any]) -> str:
    """Returns a human-friendly OS label from a machine specs dict."""
    return (
        str(specs.get("os_pretty") or "")
        or str(specs.get("platform") or "")
        or f"{specs.get('system', '')} {specs.get('release', '')}".strip()
        or "unknown"
    )
