"""
===========================
Automatically repair epochs
===========================

This example demonstrates how to use ``autoreject`` to automatically
repair epochs.
"""

# Author: Mainak Jas <mainak.jas@telecom-paristech.fr>
# License: BSD (3-clause)

import numpy as np

import mne
from mne import io
from mne import Epochs
from mne.utils import check_random_state
from mne.datasets import sample

from autoreject import (LocalAutoReject, compute_threshes, grid_search,
                        set_matplotlib_defaults)

import matplotlib.pyplot as plt

print(__doc__)

check_random_state(42)

scaling = 1e6
n_folds = 3
n_interpolates = np.array([1, 4, 32])
consensus_percs = np.linspace(0, 1.0, 11)

data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + ('/MEG/sample/sample_audvis_filt-0-40_raw-'
                           'eve.fif')
event_id = {'Auditory/Left': 1}
tmin, tmax = -0.2, 0.5

raw = io.Raw(raw_fname, preload=True)
projs, _ = mne.preprocessing.compute_proj_ecg(raw, n_eeg=1, average=True,
                                              verbose=False)
raw.add_proj(projs).apply_proj()

events = mne.read_events(event_fname)

# pick EEG and MEG channels
raw.info['bads'] = []
picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=False, eog=False,
                       include=[], exclude=[])

n_events = events.shape[0]
epochs = Epochs(raw, events, event_id, tmin, tmax,
                picks=picks, baseline=(None, 0), reject=None,
                verbose=False, detrend=0, preload=True)
epochs.drop_bad_epochs()

prefix = 'MEG data'
err_cons = grid_search(epochs, n_interpolates, consensus_percs,
                       prefix=prefix, n_folds=n_folds)

# try the best consensus perc to get clean evoked now
best_idx, best_jdx = np.unravel_index(err_cons.mean(axis=-1).argmin(),
                                      err_cons.shape[:2])
consensus_perc = consensus_percs[best_idx]
n_interpolate = n_interpolates[best_jdx]
auto_reject = LocalAutoReject(compute_threshes, consensus_perc,
                              n_interpolate=n_interpolate)
epochs_clean = auto_reject.fit_transform(epochs)

evoked = epochs.average()
evoked_clean = epochs_clean.average()

evoked.info['bads'] = ['MEG 2443']
evoked_clean.info['bads'] = ['MEG 2443']

set_matplotlib_defaults(plt)

fig, axes = plt.subplots(2, 1, figsize=(6, 6))

for ax in axes:
    ax.tick_params(axis='x', which='both', bottom='off', top='off')
    ax.tick_params(axis='y', which='both', left='off', right='off')

ylim = dict(grad=(-170, 200))
evoked1 = evoked.copy().pick_types(meg='grad', exclude=[])
evoked1.plot(exclude=[], axes=axes[0], ylim=ylim, show=False)
axes[0].set_title('Before')
evoked2 = evoked_clean.copy().pick_types(meg='grad', exclude=[])
evoked2.plot(exclude=[], axes=axes[1], ylim=ylim)
axes[1].set_title('After')
plt.tight_layout()
