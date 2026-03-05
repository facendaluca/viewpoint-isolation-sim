from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class PhaseTimings:
    """Aggregate time spent per phase across a run

    - totals are in seconds
    - counts are number of entries (steps for per-step phases)
    """

    totals_s: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    def add(self, phase: str, dt_s: float) -> None:
        self.totals_s[phase] = self.totals_s.get(phase, 0.0) + float(dt_s)
        self.counts[phase] = self.counts.get(phase, 0) + 1

    def total_s(self) -> float:
        return float(sum(self.totals_s.values()))


@dataclass(slots=True)
class PhaseTimer:
    """Concrete tracer used by benchmarks to time engine phase."""

    timings: PhaseTimings = field(default_factory=PhaseTimings)

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.timings.add(name, time.perf_counter() - t0)


def build_perf_breakdown(
    *,
    timings: PhaseTimings,
    steps: int,
    n_videos: int,
    top_k: int,
    rank_alpha: float,
    extra_phases_s: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serialisable perf breakdown dict for benchmark artefacts."""
    phases_s: dict[str, float] = dict(timings.totals_s)
    if extra_phases_s:
        for k, v in extra_phases_s.items():
            phases_s[str(k)] = float(v)

    total_s = float(sum(phases_s.values()))
    if total_s <= 0.0:
        total_s = 1e-12

    out: dict[str, Any] = {
        "steps": int(steps),
        "n_videos": int(n_videos),
        "top_k": int(top_k),
        "rank_alpha": float(rank_alpha),
        "total_s": total_s,
        "phases": {},
    }

    extra_keys = set((extra_phases_s or {}).keys())

    for name, dt_s in sorted(phases_s.items()):
        count = int(timings.counts.get(name, 0))
        entry: dict[str, Any] = {
            "total_s": round(float(dt_s), 6),
            "pct_total": round(float(dt_s) / total_s * 100.0, 2),
            "count": count,
        }
        if name in extra_keys:
            entry["per_run_ms"] = round(float(dt_s) * 1000.0, 6)
        else:
            entry["per_step_ms"] = round(float(dt_s) / float(steps) * 1000.0, 6)

        out["phases"][name] = entry

    return out
