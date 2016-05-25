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

from sklearn.grid_search import RandomizedSearchCV
from sklearn.cross_validation import KFold
from scipy.stats.distributions import uniform

import mne
from mne import io
from mne.datasets import sample

from autoreject import ChannelAutoReject
from autoreject.utils import clean_by_interp

import matplotlib
import matplotlib.pyplot as plt

matplotlib.style.use('ggplot')

fontsize = 17
params = {'axes.labelsize': fontsize + 2,
          'text.fontsize': fontsize,
          'legend.fontsize': fontsize,
          'xtick.labelsize': fontsize,
          'ytick.labelsize': fontsize,
          'axes.titlesize': fontsize + 2}
plt.rcParams.update(params)

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

epochs_gt = clean_by_interp(epochs)

picks = mne.pick_types(epochs.info, meg='grad', eeg=False, stim=False,
                       eog=False, include=include, exclude='bads')

X = epochs.get_data()
X_gt = epochs_gt.get_data()
X = np.concatenate((X, X_gt), axis=0)
np.random.seed(42)
cv = KFold(X.shape[0], 10, random_state=42)

low, high = 4e-13, 900e-13
best_threshes = np.zeros((len(picks), ))
for idx, pick in enumerate(picks):
    est = ChannelAutoReject()
    param_dist = dict(thresh=uniform(low, high))
    rs = RandomizedSearchCV(est,
                            param_distributions=param_dist,
                            n_iter=20, cv=cv)
    rs.fit(X[:, pick])
    best_thresh = rs.best_estimator_.thresh
    best_threshes[idx] = best_thresh

unit = r'fT/cm'
scaling = 1e13

plt.figure(figsize=(6, 5))
plt.tick_params(axis='x', which='both', bottom='off', top='off')
plt.tick_params(axis='y', which='both', left='off', right='off')

counts, bins, _ = plt.hist(scaling * best_threshes, 30, color='g', alpha=0.4)
plt.xlabel('Threshold (%s)' % unit)
plt.ylabel('Number of sensors')
plt.xlim((100, 950))
plt.tight_layout()
plt.show()
