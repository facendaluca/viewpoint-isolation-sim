from __future__ import annotations

import math

import pytest

from fyp_sim.plotting.compare import nonnegative_yerr


def test_lower_whisker_is_cut_off_at_zero_when_std_exceeds_mean():
    # The dissertation-readiness review case: watch-time mean ~14 s, std ~28 s.
    lower, upper = nonnegative_yerr(14.0, 28.0)
    assert lower == [14.0]
    assert upper == [28.0]


def test_symmetric_whiskers_when_std_fits_above_zero():
    lower, upper = nonnegative_yerr(0.125, 0.01)
    assert lower == [0.01]
    assert upper == [0.01]


def test_nan_std_draws_no_whisker():
    lower, upper = nonnegative_yerr(0.5, math.nan)
    assert lower == [0.0]
    assert upper == [0.0]


@pytest.mark.parametrize(("mean", "std"), [(0.0, 1.0), (0.0, 0.0)])
def test_zero_mean_never_goes_negative(mean: float, std: float):
    lower, _ = nonnegative_yerr(mean, std)
    assert lower[0] == 0.0
