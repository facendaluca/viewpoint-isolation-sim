from __future__ import annotations

from fyp_sim.analysis import compute_lock_in_metrics


def test_no_lock_in_when_never_below_threshold_long_enough():
    vii = [0.4, 0.3, 0.2, 0.3, 0.4]  # only one 0.2
    m = compute_lock_in_metrics(vii, lock_in_threshold=0.2, persistence_window=2)
    assert m.lock_in_events == 0
    assert m.time_to_first_lock_in == -1
    assert m.total_lock_in_steps == 0
    assert m.lock_in_rate == 0.0


def test_lock_in_event_detected_and_time_to_first_is_correct():
    # threshold=0.2, window=3 => first time we have 3 consecutive <= 0.2 is at index 3
    vii = [0.3, 0.2, 0.2, 0.2, 0.25]
    m = compute_lock_in_metrics(vii, lock_in_threshold=0.2, persistence_window=3)
    assert m.lock_in_events == 1
    assert m.time_to_first_lock_in == 3
    assert m.total_lock_in_steps == 3  # window counted, then breaks at 0.25


def test_lock_in_counts_steps_after_episode_start_and_continues_until_break():
    vii = [0.2, 0.2, 0.2, 0.2, 0.2]  # always locked
    m = compute_lock_in_metrics(vii, lock_in_threshold=0.2, persistence_window=3)
    assert m.lock_in_events == 1
    assert m.time_to_first_lock_in == 2
    assert m.total_lock_in_steps == 5  # full run is a lock-in episode
    assert m.max_consecutive_lock_in_steps == 5


def test_multiple_lock_in_episodes():
    vii = [0.2, 0.2, 0.2, 0.5, 0.2, 0.2, 0.2]
    m = compute_lock_in_metrics(vii, lock_in_threshold=0.2, persistence_window=3)
    assert m.lock_in_events == 2
    assert m.time_to_first_lock_in == 2
    assert m.total_lock_in_steps == 6
