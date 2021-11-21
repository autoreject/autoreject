"""
===========================
Automatically repair epochs
===========================

This example demonstrates how to use :mod:`autoreject` to automatically
repair epochs.
"""

# Author: Mainak Jas <mainak.jas@telecom-paristech.fr>
#         Denis A. Engemann <denis.engemann@gmail.com>
# License: BSD-3-Clause

# %%
# Let us first define the parameters. `n_interpolates` are the :math:`\rho`
# values that we would like :mod:`autoreject` to try and `consensus_percs`
# are the :math:`\kappa` values that :mod:`autoreject` will try (see the
# `autoreject paper <https://doi.org/10.1016/j.neuroimage.2017.06.030>`_) for
# more information on these parameters).
#
# Epochs with more than :math:`\kappa * N` sensors (:math:`N` total sensors)
# bad are dropped. For the rest of the epochs, the worst :math:`\rho`
# bad sensors (as determined by channel-level thresholds) are interpolated.
# The exact values of these parameters are not preselected but learned from
# the data. If the number of bad sensors for a particular trial is less than
# :math:`\rho`, all the bad sensors are interpolated.

# %%
import numpy as np

n_interpolates = np.array([1, 4, 32])
consensus_percs = np.linspace(0, 1.0, 11)

# %%
# For the purposes of this example, we shall use the MNE sample dataset.
# Therefore, let us make some MNE related imports.

import mne  # noqa
from mne.utils import check_random_state  # noqa
from mne.datasets import sample  # noqa

# %%
# Now, we can import the class required for rejecting and repairing bad
# epochs. :func:`autoreject.compute_thresholds` is a callable which must be
# provided to the :class:`autoreject.AutoReject` class for computing
# the channel-level thresholds.

from autoreject import (AutoReject, set_matplotlib_defaults)  # noqa

# %%
# Let us now read in the raw `fif` file for MNE sample dataset.

check_random_state(42)

data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
raw = mne.io.read_raw_fif(raw_fname, preload=True)

# %%
# We can then read in the events

event_fname = data_path + ('/MEG/sample/sample_audvis_filt-0-40_raw-'
                           'eve.fif')
event_id = {'Auditory/Left': 1, 'Auditory/Right': 2}
tmin, tmax = -0.2, 0.5

events = mne.read_events(event_fname)

# %%
# And pick MEG channels for repairing. Currently, :mod:`autoreject` can repair
# only one channel type at a time.

raw.info['bads'] = []
picks = mne.pick_types(raw.info, meg='grad', eeg=False, stim=False, eog=False,
                       include=[], exclude=[])

# %%
# Now, we can create epochs. The ``reject`` params will be set to ``None``
# because we do not want epochs to be dropped when instantiating
# :class:`mne.Epochs`.
raw.del_proj()  # remove proj, don't proj while interpolating
epochs = mne.Epochs(raw, events, event_id, tmin, tmax,
                    baseline=(None, 0), reject=None,
                    verbose=False, detrend=0, preload=True)

# %%
# :class:`autoreject.AutoReject` internally does cross-validation to
# determine the optimal values :math:`\rho^{*}` and :math:`\kappa^{*}`

# %%
# Note that :class:`autoreject.AutoReject` by design supports
# multiple channels.
# If no picks are passed, separate solutions will be computed for each channel
# type and internally combined. This then readily supports cleaning
# unseen epochs from the different channel types used during fit.
# Here we only use a subset of channels to save time.

ar = AutoReject(n_interpolates, consensus_percs, picks=picks,
                thresh_method='random_search', random_state=42)

# Note that fitting and transforming can be done on different compatible
# portions of data if needed.
ar.fit(epochs['Auditory/Left'])
epochs_clean = ar.transform(epochs['Auditory/Left'])
evoked_clean = epochs_clean.average()
evoked = epochs['Auditory/Left'].average()

# %%
# Now, we will manually mark the bad channels just for plotting.

evoked.info['bads'] = ['MEG 2443']
evoked_clean.info['bads'] = ['MEG 2443']

# %%
# Let us plot the results.

import matplotlib.pyplot as plt  # noqa
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

# %%
# To top things up, we can also visualize the bad sensors for each trial using
# a heatmap.

ar.get_reject_log(epochs['Auditory/Left']).plot()
