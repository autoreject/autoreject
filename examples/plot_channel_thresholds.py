"""
=============================
Plot channel-level thresholds
=============================

This example demonstrates how to use ``autoreject`` to find
channel-wise thresholds.
"""

# Author: Mainak Jas <mainak.jas@telecom-paristech.fr>
# License: BSD (3-clause)

import numpy as np

import mne
from mne import io
from mne.datasets import sample

from autoreject import compute_threshes, set_matplotlib_defaults
import matplotlib.pyplot as plt

set_matplotlib_defaults(plt)

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

include = []
picks = mne.pick_types(raw.info, meg='grad', eeg=False, stim=False,
                       eog=False, include=include, exclude='bads')

epochs = mne.Epochs(raw, events, event_id, tmin, tmax,
                    picks=picks, baseline=(None, 0),
                    reject=None, verbose=False, detrend=True)
epochs.load_data()

epochs.pick_types(meg='grad', eeg=False, stim=False, eog=False,
                  include=include, exclude='bads')

thresh_range = dict(grad=(4e-13, 900e-13))
threshes = np.array(compute_threshes(epochs, thresh_range)['meg'])

unit = r'fT/cm'
scaling = 1e13

plt.figure(figsize=(6, 5))
plt.tick_params(axis='x', which='both', bottom='off', top='off')
plt.tick_params(axis='y', which='both', left='off', right='off')

plt.hist(scaling * threshes, 30, color='g', alpha=0.4)
plt.xlabel('Threshold (%s)' % unit)
plt.ylabel('Number of sensors')
plt.xlim((100, 950))
plt.tight_layout()
plt.show()
