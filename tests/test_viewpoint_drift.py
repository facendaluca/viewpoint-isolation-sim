import pytest

from fyp_sim.models import UserAction
from fyp_sim.simulation.viewpoint_drift import action_weight, apply_viewpoint_drift


def test_action_weight_mapping():
    assert action_weight(UserAction.WATCH) == pytest.approx(1.0)
    assert action_weight(UserAction.SAMPLE) == pytest.approx(0.2)
    assert action_weight(UserAction.AVOID) == pytest.approx(0.0)
    assert action_weight("Watch") == pytest.approx(1.0)
    assert action_weight("Sample") == pytest.approx(0.2)
    assert action_weight(" Avoid ") == pytest.approx(0.0)

    with pytest.raises(ValueError):
        action_weight("Unknown")


def test_drift_rate_zero_is_noop():
    old = 0.25
    target = 0.90
    new = apply_viewpoint_drift(old, target, drift_rate=0.0, action=UserAction.WATCH)
    assert new == pytest.approx(old)


def test_avoid_weight_is_noop_even_when_positive_rate():
    old = 0.25
    target = 0.90
    new = apply_viewpoint_drift(old, target, drift_rate=0.9, action=UserAction.AVOID)
    assert new == pytest.approx(old)


@pytest.mark.parametrize(
    "old,target",
    [(0.1, 0.9), (0.9, 0.1), (0.33, 0.66), (0.66, 0.33)],
)
def test_single_step_moves_toward_target_and_reduces_distance(old, target):
    d0 = abs(target - old)
    new = apply_viewpoint_drift(old, target, drift_rate=0.5, action=UserAction.WATCH)
    d1 = abs(target - new)

    # New should lie between old and target (convex combination) and be closer to target.
    assert min(old, target) <= new <= max(old, target)
    assert d1 <= d0
    assert d1 < d0  # k>0 so distance should strictly shrink unless already equal


def test_repeated_step_monotonically_approach_constant_target():
    old = 0.0
    target = 1.0
    drift_rate = 0.2  # WATCH => k=0.2

    prev = old
    prev_dist = abs(target - prev)

    for _ in range(20):
        nxt = apply_viewpoint_drift(prev, target, drift_rate=drift_rate, action=UserAction.WATCH)
        dist = abs(target - nxt)

        # Should never move away from target for a constant target
        assert dist <= prev_dist + 1e-12
        assert min(prev, target) <= nxt <= max(prev, target)

        prev, prev_dist = nxt, dist
