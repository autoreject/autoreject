import numpy as np
from autoreject.bayesopt import bayes_opt, expected_improvement


def test_bayesopt():
    """Test for bayesian optimization."""
    x_star = 1.1

    def func(x):
        return (x - x_star) ** 2

    initial_x = np.linspace(.0, 2.0, 5)
    all_x = np.arange(0., 2.1, 0.1)
    best_x, _ = bayes_opt(func, initial_x, all_x, expected_improvement,
                          max_iter=10, debug=False)
    assert best_x - x_star < 0.01
