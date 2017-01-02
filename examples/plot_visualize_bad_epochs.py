"""
===============================
Visualize bad sensors per trial
===============================

This example demonstrates how to use :mod:`autoreject` to
visualize the bad sensors in each trial
"""

# Author: Mainak Jas <mainak.jas@telecom-paristech.fr>
# License: BSD (3-clause)

###############################################################################
# First, we download the data from OpenfMRI. We will download the tarfile,
# extract the necessary files and delete the tar from the disk

###############################################################################

import os
import tarfile

import autoreject
from autoreject.utils import fetch_file

subject_id = 16  # OpenfMRI format of subject numbering

src_url = ('http://openfmri.s3.amazonaws.com/tarballs/'
           'ds117_R0.1.1_sub016_raw.tgz')
subject = "sub%03d" % subject_id

print("processing subject: %s" % subject)
base_path = os.path.join(os.path.dirname(autoreject.__file__), '..', 'examples')
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

###############################################################################

tmin, tmax = -0.2, 0.8
events_id = {'famous/first': 5, 'famous/immediate': 6, 'famous/long': 7}

###############################################################################
# Let us now load all the epochs into memory and concatenate them

###############################################################################
import mne

epochs = list()
for run in range(3, 7):
    run_fname = os.path.join(base_path, 'ds117', 'sub%03d' % subject_id, 'MEG',
                             'run_%02d_raw.fif' % run)
    raw = mne.io.read_raw_fif(run_fname, preload=True, add_eeg_ref=False)
    mne.io.set_eeg_reference(raw, [])
    raw.pick_types(eeg=True, meg=False, stim=True)  # less memory + computation
    raw.filter(1, 40, l_trans_bandwidth=0.5, n_jobs=1, verbose='INFO')

    raw.set_channel_types({'EEG061': 'eog', 'EEG062': 'eog',
                           'EEG063': 'ecg', 'EEG064': 'misc'})
    raw.rename_channels({'EEG061': 'EOG061', 'EEG062': 'EOG062',
                         'EEG063': 'ECG063', 'EEG064': 'MISC'})

    exclude = []  # XXX
    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False,
                           eog=False, exclude=exclude)
    events = mne.find_events(raw, stim_channel='STI101',
                             consecutive='increasing',
                             min_duration=0.003, verbose=True)
    # Read epochs
    epoch = mne.Epochs(raw, events, events_id, tmin, tmax, proj=True,
                       add_eeg_ref=True, picks=picks, baseline=None,
                       preload=False, reject=None, decim=4)
    epochs.append(epoch)

    # Same `dev_head_t` for all runs so that we can concatenate them.
    epoch.info['dev_head_t'] = epochs[0].info['dev_head_t']
epochs = mne.epochs.concatenate_epochs(epochs)

###############################################################################
# Now, we apply autoreject

###############################################################################
from autoreject import LocalAutoRejectCV, compute_thresholds
from functools import partial

this_epoch = epochs['famous']
thresh_func = partial(compute_thresholds, random_state=42)

ar = LocalAutoRejectCV(thresh_func=thresh_func, verbose='tqdm')
epochs_ar = ar.fit_transform(this_epoch.copy())

###############################################################################
# ... and visualize the bad epochs and sensors. Bad sensors which have been
# interpolated are in blue. Bad sensors which are not interpolated are in red.
# Bad trials are also in red.

###############################################################################
from autoreject import plot_epochs
plot_epochs(this_epoch, bad_epochs_idx=ar.bad_epochs_idx,
            fix_log=ar.fix_log, scalings=dict(eeg=40e-6),
            title='')

###############################################################################
# ... and the epochs after cleaning with autoreject

###############################################################################
epochs_ar.plot(scalings=dict(eeg=40e-6))

###############################################################################
# Finally, the evoked before and after autoreject, for sanity check. We use
# the ``spatial_colors`` argument from MNE as it allows us to see that
# the eyeblinks have not yet been cleaned but the bad channels have been
# repaired.

###############################################################################
ylim = dict(eeg=(-15, 15))
epochs.average().plot(ylim=ylim, spatial_colors=True)
epochs_ar.average().plot(ylim=ylim, spatial_colors=True)
