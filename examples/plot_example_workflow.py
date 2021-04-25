"""
===============================
Example ``autoreject`` workflow
===============================

This example demonstrates how to visualize data when preprocessing
with :mod:`autoreject` and discusses decisions about when and which
other preprocessing steps to use in combination.
"""

# Author: Alex Rockhill <aprockhill@mailbox.org>
#         Mainak Jas <mainak.jas@telecom-paristech.fr>
#         Apoorva Karekal <apoorvak@uoregon.edu>
#
# License: BSD (3-clause)

# sphinx_gallery_thumbnail_number = 2

# %%
# First, we download resting-state EEG data from a Parkinson's patient
# from OpenNeuro. We will do this using ``openneuro-py`` which can be
# installed with the command ``pip install openneuro-py``.

import os
import os.path as op
import matplotlib.pyplot as plt
import openneuro

import mne
import autoreject

dataset = 'ds002778'  # The id code on OpenNeuro for this example dataset
subject_id = 'pd14'

target_dir = os.path.join(
    os.path.dirname(autoreject.__file__), '..', 'examples', dataset)
if not os.path.isdir(target_dir):
    os.makedirs(target_dir)

openneuro.download(dataset=dataset, target_dir=target_dir,
                   include=[f'sub-{subject_id}/ses-off'])

# %%
# We will now load in the raw data from the bdf file downloaded from OpenNeuro
# and, since this is resting-state data without any events, make regularly
# spaced events with which to epoch the raw data. In evoked plot (the plot of
# the average of the epochs) we can see that there may be some eyeblink
# artifact contamination but, overall, the data is typical of
# resting-state EEG.

raw = mne.io.read_raw_bdf(op.join(target_dir, f'sub-{subject_id}',
                                  'ses-off', 'eeg',
                                  'sub-pd14_ses-off_task-rest_eeg.bdf'))
raw.drop_channels(['EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5',
                   'EXG6', 'EXG7', 'EXG8', 'Status'])  # drop extra channels
dig_montage = mne.channels.make_standard_montage('biosemi32')
raw.set_montage(dig_montage)  # use the standard montage
raw.load_data()
events = mne.make_fixed_length_events(raw, duration=3)
epochs = mne.Epochs(raw, events, tmin=0, baseline=None, preload=True)

# plot the data
epochs.average().detrend().plot_joint()

# %%
# Now, we'll naively apply autoreject as our first preprocessing step.
#
# As we can see in the plot of the rejected epochs, there are many eyeblinks
# that caused the epoch to be dropped. This resulted in a lot of the data
# being lost.

# the data looks fairly clean already and we don't want to interpolate
# more than a few sensors since we only have 32 to start, so the
# number of channels to interpolate was set to check low numbers
ar = autoreject.AutoReject(n_interpolate=[1, 2, 3, 4], random_state=11,
                           n_jobs=1, verbose='tqdm')
ar.fit(epochs[:20])  # fit on the first 20 epochs to save time
epochs_ar, reject_log = ar.transform(epochs, return_log=True)

# visualize the dropped epochs
epochs[reject_log.bad_epochs].plot(scalings=dict(eeg=1e-4))

# %%
# The data may be very valuable and the time for the experiment
# limited and so we may want to take steps to reduce the number of
# epochs dropped by first using other steps to preprocess the data.
#
# We will use a highpass filter first to remove slow drift that could
# cause epochs to be dropped. As we can see in the plot, this reduced
# the number of epochs marked as bad by autoreject substantially.
#
# When making this decision to filter the data, we do want to be careful
# because filtering can spread sharp, high-frequency transients and
# distort the phase of the signal. Most evoked response potential
# analyses use filtering since the interest is in the time series, but
# if you are doing a frequency based analysis, filtering before the
# Fourier transform could potentially be avoided by detrending instead.

raw.filter(l_freq=1, h_freq=None)
epochs = mne.Epochs(raw, events, tmin=0, baseline=None, preload=True)
ar = autoreject.AutoReject(n_interpolate=[1, 2, 3, 4], random_state=11,
                           n_jobs=1, verbose='tqdm')
ar.fit(epochs[:20])  # fit on the first 20 epochs to save time
epochs_ar, reject_log = ar.transform(epochs, return_log=True)

# visualize the dropped epochs
epochs[reject_log.bad_epochs].plot(scalings=dict(eeg=1e-4))

# %%
# Finally, we can apply independent components analysis (ICA) to remove
# eyeblinks from the data. If our analysis were to be very dependent on
# sensors at the front of the head or frequency components near the
# frequency of eyeblink potentials or if we wanted to be extra careful
# or if we had lots of dataand didn't want to go through the trouble,
# we could skip ICA and use the previous result. However, ICA can increase
# the amount of usable data and is one of the most commonly used methods
# in EEG analyses.
#
# We can see in the plots below that ICA effectively removed eyeblink
# artifact, and, a bit counterintuitively, in doing so caused a few more
# epochs to be dropped. This is because the threshold was no longer biased
# upwards by the eyeblinks and so was more stringent.
#
# These are the basic steps for a workflow with decisions that must be
# made based on what the data is being used for. Following this may help
# you optimize your use of ``autoreject`` in preprocessing.

ica = mne.preprocessing.ICA(random_state=99)
ica.fit(epochs)
# plot with and without eyeblink component
ica.plot_overlay(epochs.average(), exclude=[0])
ica.apply(epochs, exclude=[0])

# run ``autoreject`` on more time
ar = autoreject.AutoReject(n_interpolate=[1, 2, 3, 4], random_state=11,
                           n_jobs=1, verbose='tqdm')
ar.fit(epochs[:20])  # fit on the first 20 epochs to save time
epochs_ar, reject_log = ar.transform(epochs, return_log=True)
epochs[reject_log.bad_epochs].plot(scalings=dict(eeg=1e-4))

# visualize the final outcome of the data
fig, axes = plt.subplots(2, 1, figsize=(6, 6))
ylim = dict(eeg=(-5, 5))
epochs.average().plot(exclude=[], axes=axes[0], ylim=ylim, show=False)
axes[0].set_title('Before autoreject')
epochs_ar.average().plot(exclude=[], axes=axes[1], ylim=ylim)
axes[1].set_title('After autoreject')
fig.tight_layout()
