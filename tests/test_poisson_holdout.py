"""Analytic PINN validation samples must be spatially held out."""

from fractions import Fraction

import pytest

from physica.learning import _poisson_holdout_rationals


@pytest.mark.parametrize("collocation", [2, 3, 4, 9, 39, 199, 399, 1000])
def test_holdout_points_are_distinct_from_training_collocation(collocation):
    train = {Fraction(i, collocation + 1) for i in range(1, collocation + 1)}
    holdout = _poisson_holdout_rationals(collocation)
    assert len(holdout) == 199
    assert len(set(holdout)) == 199
    assert all(Fraction(0) < point < Fraction(1) for point in holdout)
    assert train.isdisjoint(holdout)


def test_original_holdout_grid_was_identical_at_199_collocation():
    train = {Fraction(i, 200) for i in range(1, 200)}
    historical_holdout = {Fraction(i, 200) for i in range(1, 200)}
    assert train == historical_holdout
