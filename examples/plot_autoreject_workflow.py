"""
===================================================
Preprocessing workflow with ``autoreject`` and ICA
===================================================
This example demonstrates how to visualize data when preprocessing
with :mod:`autoreject` and discusses decisions about when and which
other preprocessing steps to use in combination.
"""

# Author: Alex Rockhill <aprockhill@mailbox.org>
#         Mainak Jas <mjas@mgh.harvard.edu>
#         Apoorva Karekal <apoorvak@uoregon.edu>
#
# License: BSD (3-clause)

# sphinx_gallery_thumbnail_number = 9

# %%
# .. toctree::
#
# First, we download resting-state EEG data from a Parkinson's patient
# from OpenNeuro. We will do this using ``openneuro-py`` which can be
# installed with the command ``pip install openneuro-py``.

import numpy as np
import os.path as op
import matplotlib.pyplot as plt
import openneuro

import mne
import autoreject

dataset = 'ds002778'  # The id code on OpenNeuro for this example dataset
subject_id = 'pd14'

target_dir = mne.utils._TempDir()
openneuro.download(dataset=dataset, target_dir=target_dir,
                   include=[f'sub-{subject_id}/ses-off'])

# %%
# We will now load in the raw data from the bdf file downloaded from OpenNeuro
# and, since this is resting-state data without any events, make regularly
# spaced events with which to epoch the raw data. In evoked plot (the plot of
# the average of the epochs) we can see that there may be some eyeblink
# artifact contamination but, overall, the data is typical of
# resting-state EEG.

raw_fname = op.join(target_dir, f'sub-{subject_id}',
                    'ses-off', 'eeg', 'sub-pd14_ses-off_task-rest_eeg.bdf')
raw = mne.io.read_raw_bdf(raw_fname, preload=True)
dig_montage = mne.channels.make_standard_montage('biosemi32')
raw.drop_channels([ch for ch in raw.ch_names
                   if ch not in dig_montage.ch_names])
raw.set_montage(dig_montage)  # use the standard montage
epochs = mne.make_fixed_length_epochs(raw, duration=3, preload=True)

# plot the data
epochs.average().detrend().plot_joint()

# %%
# Autoreject first pass
# ---------------------
# Now, we'll naively apply autoreject as our first preprocessing step.
#
# As we can see in the plot of the rejected epochs, there are many eyeblinks
# that caused the epoch to be dropped. This resulted in a lot of the data
# being lost.
#
# The data looks fairly clean already and we don't want to interpolate
# more than a few sensors since we only have 32 to start, so the
# number of channels to interpolate was set to check some low numbers
ar = autoreject.AutoReject(n_interpolate=[1, 2, 3, 4], random_state=11,
                           n_jobs=1, verbose='tqdm')
ar.fit(epochs[:20])  # fit on a few epochs to save time
epochs_ar, reject_log = ar.transform(epochs, return_log=True)

# %%
# visualize the dropped epochs
epochs[reject_log.bad_epochs].plot(scalings=dict(eeg=100e-6))

# %%
# and the reject log
reject_log.plot('horizontal')

# %%
# Highpass filter
# ---------------
# The data may be very valuable and the time for the experiment
# limited and so we may want to take steps to reduce the number of
# epochs dropped by first using other steps to preprocess the data.
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
epochs = mne.make_fixed_length_epochs(raw, duration=3, preload=True)
ar = autoreject.AutoReject(n_interpolate=[1, 2, 3, 4], random_state=11,
                           n_jobs=1, verbose='tqdm')
ar.fit(epochs[:20])  # fit on a few epochs to save time
epochs_ar, reject_log = ar.transform(epochs, return_log=True)

# %%
# visualize the dropped epochs
epochs[reject_log.bad_epochs].plot(scalings=dict(eeg=100e-6))

# %%
# and the reject log
reject_log.plot('horizontal')

# %%
# ICA
# ---
# Finally, we can apply independent components analysis (ICA) to remove
# eyeblinks from the data. If our analysis were to be very dependent on
# sensors at the front of the head, we could skip ICA and use the previous
# result. However, ICA can increase the amount of usable data and is one
# of the most commonly used methods in EEG analyses.
#
# We first find the global rejection threshold and only then run ICA, and then
# finally run the local rejection threshold for each channel. This sequence
# is recommended.
#
# We can see in the plots below that ICA effectively removed eyeblink
# artifact, and, in doing so, reduced the number of epochs that were dropped.

# find global rejection threshold
reject = autoreject.get_rejection_threshold(epochs)

# compute ICA
ica = mne.preprocessing.ICA(random_state=99)
ica.fit(epochs, reject=reject)

# %%
# plot source components to see which is made up of blinks
ica.plot_sources(epochs)

# exclude components with eyeblink artifact
ica.exclude = [0,  # blinks
               1  # saccades
               ]

# %%
# plot with and without eyeblink component
ica.plot_overlay(epochs.average(), exclude=ica.exclude)
ica.apply(epochs, exclude=ica.exclude)

# %%
# Autoreject final pass
# ---------------------
# We can see in this section that preprocessing, especially ICA, can be made
# to do a lot of the heavy lifting. There isn't a huge difference when viewing
# the averaged data (the evoked) because the ICA effectively limited the number
# of epochs that had to be dropped. However, there are still artifacts such as
# non-stereotypical blinks that weren't able to be removed by ICA, channel
# "pops" (sharp transients with exponential RC decay), muscle artifact such as
# jaw clenches and gross movement artifact that could still impact analyses.
#
# These are the basic steps for a workflow with decisions that must be
# made based on what the data is being used for. Following this may help
# you optimize your use of ``autoreject`` in preprocessing.

# compute channel-level rejections
ar = autoreject.AutoReject(n_interpolate=[1, 2, 3, 4], random_state=11,
                           n_jobs=1, verbose='tqdm')
ar.fit(epochs[:20])  # fit on the first 20 epochs to save time
epochs_ar, reject_log = ar.transform(epochs, return_log=True)
epochs[reject_log.bad_epochs].plot(scalings=dict(eeg=100e-6))

# %%
# We will do a few more visualizations to see that removing the bad epochs
# found by ``autoreject`` is still important even with preprocessing first.
# This is especially important if your analyses include trial-level statistics
# such as looking for bursting activity. We'll visualize why autoreject
# excluded these epochs and the effect that including these bad epochs would
# have on the data.
#
# First, we will visualize the reject log
reject_log.plot('horizontal')

# %%
# Next, we will visualize the average data
fig, axes = plt.subplots(2, 1, figsize=(6, 6))
ylim = dict(eeg=(-5, 5))
epochs.average().plot(exclude=[], axes=axes[0], ylim=ylim, show=False)
axes[0].set_title('Before autoreject')
epochs_ar.average().plot(exclude=[], axes=axes[1], ylim=ylim)
axes[1].set_title('After autoreject')
fig.tight_layout()
