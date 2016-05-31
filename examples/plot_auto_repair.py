"""
===========================
Automatically repair epochs
===========================

This example demonstrates how to use ``autoreject`` to automatically
repair epochs.
"""

# Author: Mainak Jas <mainak.jas@telecom-paristech.fr>
# License: BSD (3-clause)

###############################################################################
# Let us first define the parameters. `n_interpolates` are the :math:`\rho`
# values that we would like ``autoreject`` to try and `consensus_percs`
# are the :math:`\kappa` values that ``autoreject`` will try.
#
# Epochs with more than :math:`\kappa * N` sensors (:math:`N` total sensors)
# bad are dropped. For the rest of the epochs, the worst :math:`\rho` sensors
# are interpolated. The exact values of these parameters are not preselected
# but learned from the data.

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
# epochs. :func:`autoreject.compute_threshes` is a callable which must be
# provided to the :class:`autoreject.LocalAutoRejectCV` class for computing
# the channel-level thresholds.

###############################################################################

from autoreject import (LocalAutoRejectCV, compute_threshes,
                        set_matplotlib_defaults)

###############################################################################
# Let us now read in the raw `fif` file for MNE sample dataset.

###############################################################################

check_random_state(42)

data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
raw = io.Raw(raw_fname, preload=True)

###############################################################################
# We will remove the ECG artifacts from our signal using SSP projectors.

###############################################################################

projs, _ = mne.preprocessing.compute_proj_ecg(raw, n_eeg=1, average=True,
                                              verbose=False)
raw.add_proj(projs).apply_proj()

###############################################################################
# We can then read in the events

###############################################################################
event_fname = data_path + ('/MEG/sample/sample_audvis_filt-0-40_raw-'
                           'eve.fif')
event_id = {'Auditory/Left': 1}
tmin, tmax = -0.2, 0.5

events = mne.read_events(event_fname)

###############################################################################
# And pick MEG channels for repairing. Currently, ``autoreject`` can repair
# only one channel type at a time.

###############################################################################
raw.info['bads'] = []
picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=False, eog=False,
                       include=[], exclude=[])

###############################################################################
# Now, we can create epochs. The reject params will be set to ``None`` because
# we do not want MNE-Python to drop any epochs.

###############################################################################

epochs = Epochs(raw, events, event_id, tmin, tmax,
                picks=picks, baseline=(None, 0), reject=None,
                verbose=False, detrend=0, preload=True)

###############################################################################
# :class:`autoreject.LocalAutoRejectCV` internally does cross-validation to
# determine the optimal values :math:`\rho^{*}` and :math:`\kappa^{*}`

###############################################################################

ar = LocalAutoRejectCV(n_interpolates, consensus_percs, compute_threshes)
epochs_clean = ar.fit_transform(epochs)

evoked = epochs.average()
evoked_clean = epochs_clean.average()

###############################################################################
# Now, we will manually mark the bad channels just for plotting.

###############################################################################

evoked.info['bads'] = ['MEG 2443']
evoked_clean.info['bads'] = ['MEG 2443']

###############################################################################
# Finally, let us plot the results.

###############################################################################
import matplotlib.pyplot as plt
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

###############################################################################
