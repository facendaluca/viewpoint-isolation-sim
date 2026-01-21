import pytest

from fyp_sim.metrics import running_mean, viewpoint_distance


def test_viewpoint_distance():
    assert viewpoint_distance(0.2, 0.7) == pytest.approx(0.5)
    assert viewpoint_distance(0.7, 0.2) == pytest.approx(0.5)


def test_running_mean_online_update():
    mean = 0.0
    values = [1.0, 2.0, 3.0, 4.0]
    for i, v in enumerate(values):
        mean = running_mean(mean, i, v)
    assert mean == 2.5  # Mean of [1,2,3,4] is 2.5
