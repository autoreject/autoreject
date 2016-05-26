"""
===============================
Find global rejection threshold
===============================

This example demonstrates how to use ``autoreject`` to
find global rejection thresholds.
"""

# Author: Mainak Jas <mainak.jas@telecom-paristech.fr>
# License: BSD (3-clause)

import numpy as np

import mne
import matplotlib.pyplot as plt
from autoreject import (GlobalAutoReject, validation_curve,
                        set_matplotlib_defaults)

from mne.datasets import sample
from mne import io

print(__doc__)

set_matplotlib_defaults(plt)

data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
raw = io.Raw(raw_fname, preload=True)

event_fname = data_path + ('/MEG/sample/sample_audvis_filt-0-40_raw-'
                           'eve.fif')
event_id = {'Visual/Left': 3}
tmin, tmax = -0.2, 0.5
events = mne.read_events(event_fname)


include = []
picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False,
                       eog=False, include=include, exclude='bads')
epochs = mne.Epochs(raw, events, event_id, tmin, tmax,
                    picks=picks, baseline=(None, 0),
                    reject=None, verbose=False, detrend=True)

param_range = np.linspace(400e-7, 200e-6, 30)

human_thresh = 80e-6
unit = r'$\mu$V'
scaling = 1e6

_, test_scores = validation_curve(
    GlobalAutoReject(), epochs, y=None,
    param_name="thresh", param_range=param_range, cv=5, n_jobs=1)

test_scores = -test_scores.mean(axis=1)
best_thresh = param_range[np.argmin(test_scores)]

plt.figure(figsize=(8, 5))
plt.tick_params(axis='x', which='both', bottom='off', top='off')
plt.tick_params(axis='y', which='both', left='off', right='off')

colors = plt.rcParams['axes.color_cycle']

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

ax = plt.gca()
plt.tight_layout()
plt.show()
