from __future__ import annotations

from fyp_sim.models import UserAction


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def action_weight(action: UserAction | str) -> float:
    """
    Map an engagement action to a drift weight (Chapter 3 design weights).
        - Watch -> 1.0
        - Sample -> 0.2
        - Avoid -> 0.0

    Accepts either UserAction or a string value ("Watch"/"watch, etc.).
    """
    if isinstance(action, UserAction):
        key = action.value.strip().lower()
    else:
        key = str(action).strip().lower()

    if key == "watch":
        return 1.0
    elif key == "sample":
        return 0.2
    elif key == "avoid":
        return 0.0

    raise ValueError(f"Unknown action for weight mapping: {action!r}")


def apply_viewpoint_drift(
    user_viewpoint: float,
    video_viewpoint: float,
    *,
    drift_rate: float | None = None,
    rate: float | None = None,
    action: UserAction | str,
) -> float:
    """
    Update user viewpoint based toward consumed content viewpoint.

    Design (linear interpolation / 'pull'):
        new = old + k * (target - old)
        k = clamp01(drift_rate * weight(action))

    With 0 <= k <= 1, new is always between old and target (no overshoot),
    and distance to target shrinks by factor (1-k) for that step.
    """
    if drift_rate is None:
        drift_rate = 0.0 if rate is None else rate
    elif rate is not None:
        raise TypeError("Pass only one of 'drift_rate' or 'rate'.")

    if drift_rate < 0.0:
        raise ValueError("drift_rate must be >= 0.0")
    w = action_weight(action)
    k = _clamp01(drift_rate * w)

    old = float(user_viewpoint)
    target = float(video_viewpoint)
    new = old + k * (target - old)

    # Keep stance values normalised even if callers provide out-of-range inputs
    return _clamp01(new)
