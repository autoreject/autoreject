"""Bayesian optimization according to Brochu, Cora, and de Freitas' tutorial
http://haikufactory.com/files/bayopt.pdf
"""

# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Mainak Jas <mainak.jas@telecom-paristech.fr>

from sklearn import gaussian_process
import numpy as np

import scipy.optimize
import scipy.stats as st


def expected_improvement(gp, best_y):
    """The expected improvement acquisition function.

    The equation is explained in Eq (3) of the tutorial"""
    def ev(x):
        y, y_std = gp.predict(x * np.ones((1, 1)), return_std=True)
        Z = (y[0] - best_y) / (y_std[0] + 1e-12)
        return (y[0] - best_y) * st.norm.cdf(Z) + y_std[0] * st.norm.pdf(Z)
    return ev


def bayes_opt(f, initial_x, acquisition, max_iter=100, tol=1e-12, debug=False):
    """The actual bayesian optimization function.

    f is the very expensive function we want to minimize.

    initial_x is a matrix of at least two data points (preferrably
    more, randomly sampled).

    acquisition is the acquisiton function we want to use to find
    query points."""

    X = list(initial_x)
    y = [f(x) for x in initial_x]
    best_x = initial_x[np.argmin(y)]
    best_f = y[np.argmin(y)]
    gp = gaussian_process.GaussianProcessRegressor()

    if debug:
        print("iter", -1, "best_x", best_x, best_f)

    for i in range(max_iter):
        gp.fit(np.array(X)[:, None], np.array(y))
        new_x = scipy.optimize.fmin_l_bfgs_b(acquisition(gp, best_f),
                                             x0=best_x, approx_grad=True)[0]
        new_f = f(new_x)
        X.append(new_x[0])
        y.append(new_f)

        if new_f < best_f:
            best_f = new_f
            best_x = new_x[0]

        if debug:
            print("iter", i, "best_x", best_x, best_f)
    return best_x, best_f
