"""
===========================
Automatically repair epochs
===========================

This example demonstrates how to use :mod:`autoreject` to automatically
repair epochs.
"""

# Author: Mainak Jas <mainak.jas@telecom-paristech.fr>
# License: BSD (3-clause)

###############################################################################
# Let us first define the parameters. `n_interpolates` are the :math:`\rho`
# values that we would like :mod:`autoreject` to try and `consensus_percs`
# are the :math:`\kappa` values that :mod:`autoreject` will try.
#
# Epochs with more than :math:`\kappa * N` sensors (:math:`N` total sensors)
# bad are dropped. For the rest of the epochs, the worst :math:`\rho`
# bad sensors (as determined by channel-level thresholds) are interpolated.
# The exact values of these parameters are not preselected but learned from
# the data. If the number of bad sensors for a particular trial is less than
# :math:`\rho`, all the bad sensors are interpolated.

###############################################################################
import numpy as np

n_interpolates = np.array([1, 4, 32])
consensus_percs = np.linspace(0, 1.0, 11)

###############################################################################
# For the purposes of this example, we shall use the MNE sample dataset.
# Therefore, let us make some MNE related imports.

###############################################################################

import mne
from mne import io
from mne import Epochs
from mne.utils import check_random_state
from mne.datasets import sample

###############################################################################
# Now, we can import the class required for rejecting and repairing bad
# epochs. :func:`autoreject.compute_thresholds` is a callable which must be
# provided to the :class:`autoreject.LocalAutoRejectCV` class for computing
# the channel-level thresholds.

###############################################################################

from autoreject import (LocalAutoRejectCV, compute_thresholds,
                        set_matplotlib_defaults)

###############################################################################
# Let us now read in the raw `fif` file for MNE sample dataset.

###############################################################################

check_random_state(42)

data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
raw = io.read_raw_fif(raw_fname, preload=True)

###############################################################################
# We can then read in the events

###############################################################################
event_fname = data_path + ('/MEG/sample/sample_audvis_filt-0-40_raw-'
                           'eve.fif')
event_id = {'Auditory/Left': 1}
tmin, tmax = -0.2, 0.5

events = mne.read_events(event_fname)

###############################################################################
# And pick MEG channels for repairing. Currently, :mod:`autoreject` can repair
# only one channel type at a time.

###############################################################################
raw.info['bads'] = []
picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=False, eog=False,
                       include=[], exclude=[])

###############################################################################
# Now, we can create epochs. The ``reject`` params will be set to ``None``
# because we do not want epochs to be dropped when instantiating
# :class:`mne.Epochs`.

###############################################################################
raw.info['projs'] = list()  # remove proj, don't proj while interpolating
epochs = Epochs(raw, events, event_id, tmin, tmax,
                picks=picks, baseline=(None, 0), reject=None,
                verbose=False, detrend=0, preload=True)

###############################################################################
# First, we set up the function to compute the sensor-level thresholds.

###############################################################################
from functools import partial
thresh_func = partial(compute_thresholds, method='random_search',
                      random_state=42)

###############################################################################
# :class:`autoreject.LocalAutoRejectCV` internally does cross-validation to
# determine the optimal values :math:`\rho^{*}` and :math:`\kappa^{*}`

###############################################################################

ar = LocalAutoRejectCV(n_interpolates, consensus_percs,
                       thresh_func=thresh_func)
epochs_clean = ar.fit_transform(epochs)

evoked = epochs.average()
evoked_clean = epochs_clean.average()

###############################################################################
# Now, we will manually mark the bad channels just for plotting.

###############################################################################

evoked.info['bads'] = ['MEG 2443']
evoked_clean.info['bads'] = ['MEG 2443']

###############################################################################
# Let us plot the results.

###############################################################################
import matplotlib.pyplot as plt
set_matplotlib_defaults(plt)

fig, axes = plt.subplots(2, 1, figsize=(6, 6))

for ax in axes:
    ax.tick_params(axis='x', which='both', bottom='off', top='off')
    ax.tick_params(axis='y', which='both', left='off', right='off')

ylim = dict(grad=(-170, 200))
evoked.pick_types(meg='grad', exclude=[])
evoked.plot(exclude=[], axes=axes[0], ylim=ylim, show=False)
axes[0].set_title('Before autoreject')
evoked_clean.pick_types(meg='grad', exclude=[])
evoked_clean.plot(exclude=[], axes=axes[1], ylim=ylim)
axes[1].set_title('After autoreject')
plt.tight_layout()

###############################################################################
# To top things up, we can also visualize the bad sensors for each trial using
# :func:`seaborn.heatmap`.

###############################################################################
import seaborn as sns
set_matplotlib_defaults(plt)

plt.figure(figsize=(18, 6))
ax = sns.heatmap(ar.bad_segments, xticklabels=10, yticklabels=20, square=True,
                 cbar=False, cmap='Reds')
ax.set_xlabel('Sensors')
ax.set_ylabel('Trials')

plt.setp(ax.get_yticklabels(), rotation=0)
plt.setp(ax.get_xticklabels(), rotation=90)
plt.tight_layout(rect=[None, None, None, 1.1])
plt.show()
