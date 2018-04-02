"""
===============================
Visualize bad sensors per trial
===============================

This example demonstrates how to use :mod:`autoreject` to
visualize the bad sensors in each trial
"""

# Author: Mainak Jas <mainak.jas@telecom-paristech.fr>
#         Denis A. Engemann <denis.engemann@gmail.com>
# License: BSD (3-clause)

###############################################################################
# First, we download the data from OpenfMRI. We will download the tarfile,
# extract the necessary files and delete the tar from the disk

import os
import tarfile

import autoreject
from autoreject.utils import fetch_file

subject_id = 16  # OpenfMRI format of subject numbering

src_url = ('http://openfmri.s3.amazonaws.com/tarballs/'
           'ds117_R0.1.1_sub016_raw.tgz')
subject = "sub%03d" % subject_id

print("processing subject: %s" % subject)
base_path = os.path.join(
    os.path.dirname(autoreject.__file__), '..', 'examples')
target = os.path.join(base_path, 'ds117_R0.1.1_sub016_raw.tgz')
if not os.path.exists(os.path.join(base_path, 'ds117')):
    if not os.path.exists(target):
        fetch_file(src_url, target)
    tf = tarfile.open(target)
    print('Extracting files. This may take a while ...')
    tf.extractall(path=base_path, members=tf.getmembers()[-25:-9:3])
    os.remove(target)

###############################################################################
# We will create epochs with data starting 200 ms before trigger onset
# and continuing up to 800 ms after that. The data contains visual stimuli for
# famous faces, unfamiliar faces, as well as scrambled faces.

tmin, tmax = -0.2, 0.8
events_id = {'famous/first': 5, 'famous/immediate': 6, 'famous/long': 7}

###############################################################################
# Let us now load all the epochs into memory and concatenate them

import mne  # noqa

epochs = list()
for run in range(3, 7):
    run_fname = os.path.join(base_path, 'ds117', 'sub%03d' % subject_id, 'MEG',
                             'run_%02d_raw.fif' % run)
    raw = mne.io.read_raw_fif(run_fname, preload=True)
    raw.pick_types(eeg=True, meg=False, stim=True)  # less memory + computation
    raw.filter(1., 40., l_trans_bandwidth=0.5, n_jobs=1, verbose='INFO')

    raw.set_channel_types({'EEG061': 'eog', 'EEG062': 'eog',
                           'EEG063': 'ecg', 'EEG064': 'misc'})
    raw.rename_channels({'EEG061': 'EOG061', 'EEG062': 'EOG062',
                         'EEG063': 'ECG063', 'EEG064': 'MISC'})

    events = mne.find_events(raw, stim_channel='STI101',
                             consecutive='increasing',
                             min_duration=0.003, verbose=True)
    # Read epochs
    mne.io.set_eeg_reference(raw)

    epoch = mne.Epochs(raw, events, events_id, tmin, tmax, proj=True,
                       baseline=None,
                       preload=False, reject=None, decim=4)
    epochs.append(epoch)

    # Same `dev_head_t` for all runs so that we can concatenate them.
    epoch.info['dev_head_t'] = epochs[0].info['dev_head_t']


epochs = mne.epochs.concatenate_epochs(epochs)
###############################################################################
# Now, we apply autoreject

from autoreject import LocalAutoRejectCV, compute_thresholds  # noqa
from functools import partial  # noqa

this_epoch = epochs['famous']
exclude = []  # XXX
picks = mne.pick_types(epochs.info, meg=False, eeg=True, stim=False,
                       eog=False, exclude=exclude)

thresh_func = partial(compute_thresholds, random_state=42, n_jobs=1)

###############################################################################
# Note that :class:`autoreject.LocalAutoRejectCV` by design supports multiple
# channels. If no picks are passed separate solutions will be computed for each
# channel type and internally combines. This then readily supports cleaning
# unseen epochs from the different channel types used during fit.
# Here we only use a subset of channels to save time.

###############################################################################
# Also note that once the parameters are learned, any data can be repaired
# that contains channels that were used during fit. This also means that time
# may be saved by fitting :class:`autoreject.LocalAutoRejectCV` on a
# representative subsample of the data.


ar = LocalAutoRejectCV(thresh_func=thresh_func, verbose='tqdm', picks=picks)

ar.fit(this_epoch)
epochs_ar = ar.transform(this_epoch)

epochs_ar, reject_log = ar.fit_transform(this_epoch, return_log=True)
###############################################################################
# We can visualize the cross validation curve over two variables

import numpy as np  # noqa
import matplotlib.pyplot as plt  # noqa
import matplotlib.patches as patches  # noqa
from autoreject import set_matplotlib_defaults  # noqa

set_matplotlib_defaults(plt, style='seaborn-white')
loss = ar.loss_['eeg'].mean(axis=-1)  # losses are stored by channel type.

plt.matshow(loss.T * 1e6, cmap=plt.get_cmap('viridis'))
plt.xticks(range(len(ar.consensus)), ar.consensus)
plt.yticks(range(len(ar.n_interpolates)), ar.n_interpolates)

# Draw rectangle at location of best parameters
ax = plt.gca()
idx, jdx = np.unravel_index(loss.argmin(), loss.shape)
rect = patches.Rectangle((idx - 0.5, jdx - 0.5), 1, 1, linewidth=2,
                         edgecolor='r', facecolor='none')
ax.add_patch(rect)
ax.xaxis.set_ticks_position('bottom')
plt.xlabel(r'Consensus percentage $\kappa$')
plt.ylabel(r'Max sensors interpolated $\rho$')
plt.title('Mean cross validation error (x 1e6)')
plt.colorbar()
plt.show()

###############################################################################
# ... and visualize the bad epochs and sensors. Bad sensors which have been
# interpolated are in blue. Bad sensors which are not interpolated are in red.
# Bad trials are also in red.

scalings = dict(eeg=40e-6)


reject_log.plot_epochs(this_epoch, scalings=scalings)

###############################################################################
# ... and the epochs after cleaning with autoreject

epochs_ar.plot(scalings=scalings)


###############################################################################
# The epochs dropped by autoreject are also stored in epochs.drop_log

epochs_ar.plot_drop_log()

###############################################################################
# Finally, the evoked before and after autoreject, for sanity check. We use
# the ``spatial_colors`` argument from MNE as it allows us to see that
# the eyeblinks have not yet been cleaned but the bad channels have been
# repaired.

ylim = dict(eeg=(-15, 15))
epochs.average().plot(ylim=ylim, spatial_colors=True)
epochs_ar.average().plot(ylim=ylim, spatial_colors=True)
