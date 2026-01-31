"""
Metrics for viewpoint exposure.

- VII_t: per-step viewpoint distance between user stance and consumed content stance.
- VII_cum: running mean of VII_t over time (online update for efficiency and streaming logs).
"""

from __future__ import annotations


def viewpoint_distance(user_viewpoint: float, video_viewpoint: float) -> float:
    """Per-step viewpoint distance (VII_t)

    Defined as absolute difference between user and video stance scores.
    If stance scores are in [0, 1], then VII_t is in [0, 1].
    """
    return abs(user_viewpoint - video_viewpoint)


def running_mean(previous_mean: float, step_index: int, new_value: float) -> float:
    """Online (streaming) running mean.

    Args:
        previous_mean: mean after processing steps [0..step_index-1]
        step_index: 0-based index of the new observation being incorporated.
        new_value: the new observation at this step.

    Returns:
        Updated mean after incorporating new_value.

    Notes:
        - Floating-point arithmetic can introduce tiny rounding error (e.g., 0.4999999999).
        - Tests should use approximate comparisons when needed.
    """
    n = step_index + 1
    return previous_mean + (new_value - previous_mean) / n
