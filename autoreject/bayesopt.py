"""Bayesian optimization according to:

Brochu, Cora, and de Freitas' tutorial at
http://haikufactory.com/files/bayopt.pdf

Adopted from http://atpassos.me/post/44900091837/bayesian-optimization
"""

# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Alexandre Passos <alexandre.tp@gmail.com>
#          Mainak Jas <mainak.jas@telecom-paristech.fr>

import warnings
from sklearn import gaussian_process
import numpy as np

import scipy.stats as st


def expected_improvement(gp, best_y, x):
    """The expected improvement acquisition function.

    The equation is explained in Eq (3) of the tutorial."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, y_std = gp.predict(x[:, None], return_std=True)
    Z = (y - best_y) / (y_std + 1e-12)
    return (y - best_y) * st.norm.cdf(Z) + y_std * st.norm.pdf(Z)


def bayes_opt(f, initial_x, all_x, acquisition, max_iter=100, debug=False,
              random_state=None):
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
    gp = gaussian_process.GaussianProcessRegressor(random_state=random_state)

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
