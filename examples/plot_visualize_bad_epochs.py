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
# First, we download the data from openfMRI. We will download the tarfile,
# extract the necessary files and delete the tar from the disk
###############################################################################

import os
import os.path as op
import mne
import tarfile

import autoreject
from autoreject import LocalAutoRejectCV, plot_epochs
from autoreject.utils import fetch_file

subject_id = 16  # Openfmri format
tmin, tmax = -0.2, 0.8
events_id = {'famous/first': 5, 'famous/immediate': 6, 'famous/long': 7}

src_url = ('http://openfmri.s3.amazonaws.com/tarballs/'
           'ds117_R0.1.1_sub016_raw.tgz')

subject = "sub%03d" % subject_id
print("processing subject: %s" % subject)

base_path = op.join(op.dirname(autoreject.__file__), '..', 'examples')
target = op.join(base_path, 'ds117_R0.1.1_sub016_raw.tgz')
if not op.exists(op.join(base_path, 'ds117')):
    if not op.exists(target):
        fetch_file(src_url, target)
    tf = tarfile.open(target)
    print('Extracting files. This may take a while ...')
    tf.extractall(path=base_path, members=tf.getmembers()[-25:-9:3])
    os.remove(target)

epochs = list()
for run in range(1, 7):
    run_fname = op.join(base_path, 'ds117',
                        'sub%03d' % subject_id, 'MEG',
                        'run_%02d_raw.fif' % run)
    raw = mne.io.Raw(run_fname, preload=True, add_eeg_ref=False)
    mne.io.set_eeg_reference(raw, [])
    # less memory + computation
    raw.pick_types(eeg=True, meg=False, stim=True)
    raw.filter(1, 40, n_jobs=1, verbose='INFO')

    raw.set_channel_types({'EEG061': 'eog', 'EEG062': 'eog',
                           'EEG063': 'ecg'})
    raw.rename_channels({'EEG061': 'EOG061', 'EEG062': 'EOG062',
                         'EEG063': 'ECG063'})

    exclude = []  # XXX
    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False,
                           eog=True, exclude=exclude)
    events = mne.find_events(raw, stim_channel='STI101',
                             consecutive='increasing',
                             min_duration=0.003, verbose=True)
    # Read epochs
    epoch = mne.Epochs(raw, events, events_id, tmin, tmax, proj=False,
                       add_eeg_ref=False, picks=picks, baseline=None,
                       preload=True, reject=None, decim=4)
    epochs.append(epoch)
    mne.io.set_eeg_reference(epoch)
    if run > 1:
        epoch.info['dev_head_t'] = epochs[0].info['dev_head_t']
epochs = mne.epochs.concatenate_epochs(epochs)

this_epoch = epochs['famous']
ar = LocalAutoRejectCV()
epochs_ar = ar.fit_transform(this_epoch.copy())

plot_epochs(this_epoch, bad_epochs_idx=ar.bad_epochs_idx,
            fix_log=ar.fix_log, scalings=dict(eeg=40e-6),
            title='')
