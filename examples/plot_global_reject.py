"""
===================================
Plotting the cross-validation curve
===================================

This example demonstrates how to use :mod:`autoreject` to
plot the cross-validation curve that is used to estimate the
global rejection thresholds.
"""

# Author: Mainak Jas <mainak.jas@telecom-paristech.fr>
# License: BSD (3-clause)

###############################################################################
# Let us import the data using MNE-Python and epoch it.

import mne
from mne import io
from mne.datasets import sample

event_id = {'Visual/Left': 3}
tmin, tmax = -0.2, 0.5

data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + ('/MEG/sample/sample_audvis_filt-0-40_raw-'
                           'eve.fif')

raw = io.read_raw_fif(raw_fname, preload=True)
events = mne.read_events(event_fname)

include = []
picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False,
                       eog=False, include=include, exclude='bads')
epochs = mne.Epochs(raw, events, event_id, tmin, tmax,
                    picks=picks, baseline=(None, 0),
                    reject=None, verbose=False, detrend=1)

###############################################################################
# Let us define a range of candidate thresholds which we would like to try.
# In this particular case, we try from :math:`40{\mu}V` to :math:`200{\mu}V`

import numpy as np  # noqa
param_range = np.linspace(40e-6, 200e-6, 30)

###############################################################################
# Next, we can use :class:`autoreject.GlobalAutoReject` to find global
# (i.e., for all channels) peak-to-peak thresholds. It is a class which
# follows a :mod:`scikit-learn`-like API. To compute the Root Mean Squared
# (RMSE) values at the candidate thresholds, we will use the function
# :func:`autoreject.validation_curve`.

from autoreject import GlobalAutoReject, validation_curve  # noqa

_, test_scores = validation_curve(
    GlobalAutoReject(), epochs, y=None,
    param_name="thresh", param_range=param_range, cv=5)

test_scores = -test_scores.mean(axis=1)
best_thresh = param_range[np.argmin(test_scores)]

###############################################################################
# Now let us plot the RMSE values against the candidate thresholds.

import matplotlib.pyplot as plt  # noqa
from autoreject import set_matplotlib_defaults  # noqa
set_matplotlib_defaults(plt)

human_thresh = 80e-6
unit = r'$\mu$V'
scaling = 1e6

plt.figure(figsize=(8, 5))
plt.tick_params(axis='x', which='both', bottom='off', top='off')
plt.tick_params(axis='y', which='both', left='off', right='off')

colors = ['#E24A33', '#348ABD', '#988ED5']

plt.plot(scaling * param_range, scaling * test_scores,
         'o-', markerfacecolor='w',
         color=colors[0], markeredgewidth=2, linewidth=2,
         markeredgecolor=colors[0], markersize=8, label='CV scores')
plt.ylabel('RMSE (%s)' % unit)
plt.xlabel('Threshold (%s)' % unit)
plt.xlim((scaling * param_range[0] - 10, scaling * param_range[-1] + 10))
plt.axvline(scaling * best_thresh, label='auto global', color=colors[2],
            linewidth=2, linestyle='--')
plt.axvline(scaling * human_thresh, label='manual', color=colors[1],
            linewidth=2, linestyle=':')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
