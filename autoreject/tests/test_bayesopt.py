import numpy as np
from nose.tools import assert_true
from autoreject.bayesopt import bayes_opt, expected_improvement


def test_bayesopt():
    """Test for bayesian optimization."""

    x_star = 1.1

    def func(x):
        return (x - x_star) ** 2

    grid = np.linspace(.0, 2.0, 30)
    initial_x = grid[::5]
    best_x, _ = bayes_opt(func, initial_x, expected_improvement,
                          max_iter=10, debug=False)
    assert_true(best_x - x_star < 0.01)
