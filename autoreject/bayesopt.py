"""Bayesian optimization according to:

Brochu, Cora, and de Freitas' tutorial at
http://haikufactory.com/files/bayopt.pdf

Adopted from http://atpassos.me/post/44900091837/bayesian-optimization
"""

# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Alexandre Passos <alexandre.tp@gmail.com>
#          Mainak Jas <mainak.jas@telecom-paristech.fr>

import numpy as np
from scipy.linalg import cho_solve, cholesky, solve_triangular
import scipy.stats as st


class _GaussianProcess:
    """Gaussian process regression for 1D inputs.

    This computes the same as ``GaussianProcessRegressor(optimizer=None)`` of
    scikit-learn, with its default (fixed) kernel ``1.0 * RBF(1.0)`` and
    ``alpha=1e-10``. scikit-learn's is not used because its input and
    parameter validation dominate the computation time for the small data
    used here, and bayes_opt fits and predicts many times (10 times per
    channel and fold).
    """

    alpha = 1e-10

    @staticmethod
    def _kernel(a, b):
        return np.exp(-0.5 * (a - b.T) ** 2)

    def fit(self, X, y):
        self.X_train_ = X
        K = self._kernel(X, X)
        K[np.diag_indices_from(K)] += self.alpha
        self.L_ = cholesky(K, lower=True, check_finite=False)
        self.alpha_ = cho_solve((self.L_, True), y, check_finite=False)
        return self

    def predict(self, X, return_std=False):
        K_trans = self._kernel(X, self.X_train_)
        y_mean = K_trans @ self.alpha_
        if not return_std:
            return y_mean
        V = solve_triangular(self.L_, K_trans.T, lower=True, check_finite=False)
        # the kernel is 1 on the diagonal, and rounding can make y_var < 0
        y_var = np.clip(1. - np.einsum('ij,ij->j', V, V), 0., None)
        return y_mean, np.sqrt(y_var)


def expected_improvement(gp, best_y, x):
    """The expected improvement acquisition function.

    The equation is explained in Eq (3) of the tutorial."""
    y, y_std = gp.predict(x[:, None], return_std=True)
    Z = (y - best_y) / (y_std + 1e-12)
    return (y - best_y) * st.norm.cdf(Z) + y_std * st.norm.pdf(Z)


def bayes_opt(f, initial_x, all_x, acquisition, max_iter=100, debug=False):
    """The actual bayesian optimization function.

    f is the very expensive function we want to minimize.

    initial_x is a matrix of at least two data points (preferrably
    more, randomly sampled).

    acquisition is the acquisiton function we want to use to find
    query points."""

    X, y = list(), list()
    for x in initial_x:
        if not np.isinf(f(x)):
            y.append(f(x))
            X.append(x)

    best_x = X[np.argmin(y)]
    best_f = y[np.argmin(y)]
    gp = _GaussianProcess()

    if debug:  # pragma: no cover
        print("iter", -1, "best_x", best_x, best_f)

    for i in range(max_iter):
        gp.fit(np.array(X)[:, None], np.array(y))
        new_x = all_x[acquisition(gp, best_f, all_x).argmin()]
        new_f = f(new_x)
        if not np.isinf(new_f):
            X.append(new_x)
            y.append(new_f)
            if new_f < best_f:
                best_f = new_f
                best_x = new_x

        if debug:  # pragma: no cover
            print("iter", i, "best_x", best_x, best_f)

    if debug:  # pragma: no cover
        import matplotlib.pyplot as plt
        scale = 1e6
        sort_idx = np.argsort(X)
        plt.plot(np.array(X)[sort_idx] * scale,
                 np.array(y)[sort_idx] * scale, 'bo-')
        plt.axvline(best_x * scale, linestyle='--')
        plt.show()

    return best_x, best_f
