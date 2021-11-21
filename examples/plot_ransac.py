"""
===============================
Detect bad sensors using RANSAC
===============================

This example demonstrates how to use RANSAC [1]_ from the PREP pipeline to
detect bad sensors and repair them. Note that this implementation in
:mod:`autoreject` [2]_ is an extension of the original implementation and
works for MEG sensors as well.

References
----------
.. [1] Bigdely-Shamlo, N., Mullen, T., Kothe, C., Su, K. M., & Robbins, K. A.
       (2015). The PREP pipeline: standardized preprocessing for large-scale
       EEG analysis. Frontiers in neuroinformatics, 9, 16.
.. [2] Jas, M., Engemann, D. A., Bekhti, Y., Raimondo, F., & Gramfort, A.
       (2017). Autoreject: Automated artifact rejection for MEG and EEG data.
       NeuroImage, 159, 417-429.
"""

# Author: Mainak Jas <mainak.jas@telecom-paristech.fr>
# License: BSD-3-Clause

###############################################################################
# For the purposes of this example, we shall use the MNE sample dataset.
# Therefore, let us make some MNE related imports.

import mne
from mne import io
from mne import Epochs
from mne.datasets import sample

###############################################################################
# Let us now read in the raw `fif` file for MNE sample dataset.

data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
raw = io.read_raw_fif(raw_fname, preload=True)

###############################################################################
# We can then read in the events

event_fname = data_path + ('/MEG/sample/sample_audvis_filt-0-40_raw-'
                           'eve.fif')
event_id = {'Auditory/Left': 1}
tmin, tmax = -0.2, 0.5

events = mne.read_events(event_fname)

###############################################################################
# And pick MEG channels for repairing. Currently, :mod:`autoreject` can repair
# only one channel type at a time.

raw.info['bads'] = []

###############################################################################
# Now, we can create epochs. The ``reject`` params will be set to ``None``
# because we do not want epochs to be dropped when instantiating
# :class:`mne.Epochs`.

raw.del_proj()  # remove proj, don't proj while interpolating
epochs = Epochs(raw, events, event_id, tmin, tmax,
                baseline=(None, 0), reject=None,
                verbose=False, detrend=0, preload=True)
picks = mne.pick_types(epochs.info, meg='grad', eeg=False,
                       stim=False, eog=False,
                       include=[], exclude=[])


###############################################################################
# We import ``Ransac`` and run the familiar ``fit_transform`` method.
from autoreject import Ransac  # noqa
from autoreject.utils import interpolate_bads  # noqa

ransac = Ransac(verbose=True, picks=picks, n_jobs=1)
epochs_clean = ransac.fit_transform(epochs)

###############################################################################
# We can also get the list of bad channels computed by ``Ransac``.

print('\n'.join(ransac.bad_chs_))

###############################################################################
# Then we compute the ``evoked`` before and after interpolation.

evoked = epochs.average()
evoked_clean = epochs_clean.average()

###############################################################################
# We will manually mark the bad channels just for plotting.

evoked.info['bads'] = ['MEG 2443']
evoked_clean.info['bads'] = ['MEG 2443']

###############################################################################
# Let us plot the results.

from autoreject.utils import set_matplotlib_defaults  # noqa
import matplotlib.pyplot as plt  # noqa
set_matplotlib_defaults(plt)

fig, axes = plt.subplots(2, 1, figsize=(6, 6))

for ax in axes:
    ax.tick_params(axis='x', which='both', bottom='off', top='off')
    ax.tick_params(axis='y', which='both', left='off', right='off')

ylim = dict(grad=(-170, 200))
evoked.pick_types(meg='grad', exclude=[])
evoked.plot(exclude=[], axes=axes[0], ylim=ylim, show=False)
axes[0].set_title('Before RANSAC')
evoked_clean.pick_types(meg='grad', exclude=[])
evoked_clean.plot(exclude=[], axes=axes[1], ylim=ylim)
axes[1].set_title('After RANSAC')
fig.tight_layout()

###############################################################################
# To top things up, we can also visualize the bad sensors for each trial using
# a heatmap.

ch_names = [epochs.ch_names[ii] for ii in ransac.picks][7::10]
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.imshow(ransac.bad_log, cmap='Reds',
          interpolation='nearest')
ax.grid(False)
ax.set_xlabel('Sensors')
ax.set_ylabel('Trials')
plt.setp(ax, xticks=range(7, len(ransac.picks), 10),
         xticklabels=ch_names)
plt.setp(ax.get_yticklabels(), rotation=0)
plt.setp(ax.get_xticklabels(), rotation=90)
ax.tick_params(axis=u'both', which=u'both', length=0)
fig.tight_layout(rect=[None, None, None, 1.1])
plt.show()
