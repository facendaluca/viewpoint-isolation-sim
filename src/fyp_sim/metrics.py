from __future__ import annotations


def viewpoint_distance(user_viewpoint: float, video_viewpoint: float) -> float:
    """Absolute stance distance. Matches VII_t definition in report."""
    return abs(user_viewpoint - video_viewpoint)


def running_mean(previous_mean: float, step_index: int, new_value: float) -> float:
    """
    Online mean update.
    step_index is 0-based (0,1,2,...) Returns mean after incorporating new_value.
    """
    n = step_index + 1
    return previous_mean + (new_value - previous_mean) / n
