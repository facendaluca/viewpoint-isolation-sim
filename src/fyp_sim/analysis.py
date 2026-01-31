from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


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
