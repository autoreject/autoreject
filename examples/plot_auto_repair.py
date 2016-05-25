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

from autoreject import ConsensusAutoReject, compute_threshes, grid_search

import matplotlib.pyplot as plt
import matplotlib

print(__doc__)

check_random_state(42)

scaling = 1e6
n_folds = 3
n_interpolates = np.array([1, 4, 32])
consensus_percs = np.linspace(0, 1.0, 11)

data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
raw = io.Raw(raw_fname, preload=True)

projs, _ = mne.preprocessing.compute_proj_ecg(raw, n_eeg=1, average=True,
                                              verbose=False)
raw.add_proj(projs).apply_proj()

event_fname = data_path + ('/MEG/sample/sample_audvis_filt-0-40_raw-'
                           'eve.fif')
event_id = {'Auditory/Left': 1}
tmin, tmax = -0.2, 0.5
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
auto_reject = ConsensusAutoReject(compute_threshes, consensus_perc,
                                  n_interpolate=n_interpolate)
epochs_transformed = auto_reject.fit_transform(epochs)

evoked = epochs.average()
evoked_transformed = epochs_transformed.average()

evoked.info['bads'] = ['MEG 2443']
evoked_transformed.info['bads'] = ['MEG 2443']

matplotlib.style.use('ggplot')
fontsize = 17
params = {'axes.labelsize': fontsize + 2,
          'text.fontsize': fontsize,
          'legend.fontsize': fontsize,
          'xtick.labelsize': fontsize,
          'ytick.labelsize': fontsize}
plt.rcParams.update(params)

fig, axes = plt.subplots(2, 1, figsize=(6, 6))

for ax in axes:
    ax.tick_params(axis='x', which='both', bottom='off', top='off')
    ax.tick_params(axis='y', which='both', left='off', right='off')

ylim = dict(grad=(-170, 200))
evoked1 = evoked.copy().pick_types(meg='grad', exclude=[])
evoked1.plot(exclude=[], axes=axes[0], ylim=ylim)
axes[0].set_title('Before', fontsize=fontsize)
evoked2 = evoked_transformed.copy().pick_types(meg='grad', exclude=[])
evoked2.plot(exclude=[], axes=axes[1], ylim=ylim)
axes[1].set_title('After', fontsize=fontsize)
plt.tight_layout()
