import pytest

from fyp_sim.metrics import viewpoint_distance


def test_viewpoint_distance_is_bounded_and_hits_endpoints() -> None:
    assert viewpoint_distance(0.0, 0.0) == pytest.approx(0.0, abs=1e-12)
    assert viewpoint_distance(1.0, 1.0) == pytest.approx(0.0, abs=1e-12)
    assert viewpoint_distance(0.0, 1.0) == pytest.approx(1.0, abs=1e-12)
    assert viewpoint_distance(1.0, 0.0) == pytest.approx(1.0, abs=1e-12)

    # representative interior points
    for a, b in [(0.2, 0.8), (0.33, 0.34), (0.75, 0.1)]:
        d = viewpoint_distance(a, b)
        assert 0.0 <= d <= 1.0
