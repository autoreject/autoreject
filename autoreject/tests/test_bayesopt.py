import numpy as np
from numpy.testing import assert_allclose
import pytest
from sklearn.gaussian_process import GaussianProcessRegressor

from autoreject.bayesopt import _GaussianProcess, bayes_opt, expected_improvement


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


@pytest.mark.parametrize('scale', (1e-5, 1.))  # thresholds are ~1e-5 for MEG
def test_gaussian_process(scale):
    """Test that our Gaussian process matches scikit-learn's."""
    rng = np.random.RandomState(0)
    X = np.sort(rng.rand(30)) * scale
    X = np.concatenate([X, X[:1]])  # bayes_opt can sample a point again
    y = rng.randn(len(X))
    x = np.linspace(0, 1.2, 100) * scale
    want = GaussianProcessRegressor(random_state=0, optimizer=None).fit(
        X[:, None], y).predict(x[:, None], return_std=True)
    gp = _GaussianProcess().fit(X[:, None], y)
    got = gp.predict(x[:, None], return_std=True)
    for g, w in zip(got, want):
        assert_allclose(g, w, rtol=1e-7, atol=1e-12)
    # and so does the expected improvement, which is what bayes_opt uses
    gp_sk = GaussianProcessRegressor(random_state=0, optimizer=None).fit(
        X[:, None], y)
    assert_allclose(expected_improvement(gp, y.min(), x),
                    expected_improvement(gp_sk, y.min(), x),
                    rtol=1e-7, atol=1e-12)
