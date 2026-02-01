from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class LockInMetrics:
    """Summary of operational lock-in behaviour over a run."""

    lock_in_events: int
    time_to_first_lock_in: int  # -1 if never reaches lock-in
    max_consecutive_lock_in_steps: int
    total_lock_in_steps: int  # steps counted only after the persistence threshold is met
    lock_in_rate: float  # total_lock_in_steps / n_steps


def compute_lock_in_metrics(
    vii_ts: Iterable[float],
    *,
    lock_in_threshold: float,
    persistence_window: int,
) -> LockInMetrics:
    """Compute lock-in metrics from per-step VII values.

    Lock-in episode: a consecutive run where VII_t <= lock_in_threshold for at least
    persistence_window steps.

    - lock_in_events: number of such runs
    - time_to_first_lock_in: first timestep where the first run reaches the window (end index)
    - total_lock_in_steps: sum of run lengths for runs that meet the window
    """
    if persistence_window <= 0:
        raise ValueError("persistence_window must be > 0")
    if not (0.0 <= lock_in_threshold <= 1.0):
        raise ValueError("lock_in_threshold must be within [0.0, 1.0]")

    values = list(vii_ts)
    n = len(values)
    if n == 0:
        return LockInMetrics(
            lock_in_events=0,
            time_to_first_lock_in=-1,
            max_consecutive_lock_in_steps=0,
            total_lock_in_steps=0,
            lock_in_rate=0.0,
        )

    locked = [v <= lock_in_threshold for v in values]

    lock_in_events = 0
    time_to_first = -1
    max_consecutive = 0
    total_lock_in_steps = 0

    run_start = 0
    i = 0
    while i < n:
        if not locked[i]:
            i += 1
            continue

        # start of a locked run
        run_start = i
        while i < n and locked[i]:
            i += 1
        run_end = i  # exclusive
        run_len = run_end - run_start

        max_consecutive = max(max_consecutive, run_len)

        if run_len >= persistence_window:
            lock_in_events += 1
            total_lock_in_steps += run_len
            if time_to_first == -1:
                time_to_first = run_start + persistence_window - 1

    lock_in_rate = total_lock_in_steps / n

    return LockInMetrics(
        lock_in_events=lock_in_events,
        time_to_first_lock_in=time_to_first,
        max_consecutive_lock_in_steps=max_consecutive,
        total_lock_in_steps=total_lock_in_steps,
        lock_in_rate=lock_in_rate,
    )


def summarise_logs(
    logs: list[Any],
    *,
    lock_in_threshold: float,
    persistence_window: int,
) -> dict[str, float | int]:
    """Summarise step logs into run-level metrics for table/plots."""
    n = len(logs)
    if n == 0:
        raise ValueError("No logs to summarise.")

    vii_mean = sum(r.vii_t for r in logs) / n
    final_vii_cum = float(logs[-1].vii_cum)

    actions = [str(r.action).lower() for r in logs]
    watch_rate = actions.count("watch") / n
    sample_rate = actions.count("sample") / n
    avoid_rate = actions.count("avoid") / n

    unique_videos_seen = len({r.video_id for r in logs})
    mean_watch_time = sum(r.watch_time_s for r in logs) / n

    vii_series = [float(r.vii_t) for r in logs]
    li = compute_lock_in_metrics(
        vii_series,
        lock_in_threshold=lock_in_threshold,
        persistence_window=persistence_window,
    )

    return {
        "mean_vii": float(vii_mean),
        "final_vii_cum": float(final_vii_cum),
        "watch_rate": float(watch_rate),
        "sample_rate": float(sample_rate),
        "avoid_rate": float(avoid_rate),
        "unique_videos_seen": int(unique_videos_seen),
        "mean_watch_time": float(mean_watch_time),
        "lock_in_events": int(li.lock_in_events),
        "time_to_first_lock_in": int(li.time_to_first_lock_in),
        "max_consecutive_lock_in_steps": int(li.max_consecutive_lock_in_steps),
        "total_lock_in_steps": int(li.total_lock_in_steps),
        "lock_in_rate": float(li.lock_in_rate),
    }
