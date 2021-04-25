"""
=============================
Plot channel-level thresholds
=============================

This example demonstrates how to use :mod:`autoreject` to find
channel-wise thresholds.
"""

# Author: Mainak Jas <mjas@mgh.harvard.edu>
# License: BSD (3-clause)

###############################################################################
# Let us first load the `raw` data using :func:`mne.io.read_raw_fif`.

import numpy as np
import os.path as op
import matplotlib.pyplot as plt

import mne
from mne import io
from mne.datasets import sample
import autoreject

data_path = sample.data_path()
sample_dir = op.join(data_path, 'MEG', 'sample')
raw_fname = op.join(sample_dir, 'sample_audvis_filt-0-40_raw.fif')
raw = io.read_raw_fif(raw_fname, preload=True)

###############################################################################
# We can extract the events (or triggers) for epoching our signal.

event_fname = op.join(sample_dir, 'sample_audvis_filt-0-40_raw-eve.fif')
event_id = {'Auditory/Left': 1}
tmin, tmax = -0.2, 0.5
events = mne.read_events(event_fname)

###############################################################################
# Now that we have the events, we can extract the trials for the selection
# of channels defined by ``picks``.

epochs = mne.Epochs(raw, events, event_id, tmin, tmax,
                    baseline=(None, 0),
                    reject=None, verbose=False, preload=True)

picks = mne.pick_types(epochs.info, meg='grad', eeg=False, stim=False,
                       eog=False, exclude='bads')


###############################################################################
# Now, we compute the channel-level thresholds using
# :func:`autoreject.compute_thresholds`. The `method` parameter will determine
# how we will search for thresholds over a range of potential candidates.

# Get a dictionary of rejection thresholds
threshes = autoreject.compute_thresholds(
    epochs, picks=picks, method='random_search', random_state=42,
    augment=False, verbose='progressbar')

###############################################################################
# Finally, let us plot a histogram of the channel-level thresholds to verify
# that the thresholds are indeed different for different sensors.

autoreject.set_matplotlib_defaults(plt)

unit = r'fT/cm'
scaling = 1e13

plt.figure(figsize=(6, 5))
plt.tick_params(axis='x', which='both', bottom='off', top='off')
plt.tick_params(axis='y', which='both', left='off', right='off')

plt.hist(scaling * np.array(list(threshes.values())), 30,
         color='g', alpha=0.4)
plt.xlabel('Threshold (%s)' % unit)
plt.ylabel('Number of sensors')
plt.xlim((100, 950))
plt.tight_layout()
plt.show()
