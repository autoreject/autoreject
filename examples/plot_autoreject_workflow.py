"""

.. _plot_autoreject_workflow:

===================================================
Preprocessing workflow with ``autoreject`` and ICA
===================================================
This example demonstrates how to visualize data when preprocessing
with :mod:`autoreject` and discusses decisions about when and which
other preprocessing steps to use in combination.

**tldr**: We recommend that you first highpass filter the data,
then run autoreject (local) and supply the bad epochs detected by it
to the ICA algorithm for a robust fit, and finally run
autoreject (local) again.
"""

# Author: Alex Rockhill <aprockhill@mailbox.org>
#         Mainak Jas <mjas@mgh.harvard.edu>
#         Apoorva Karekal <apoorvak@uoregon.edu>
#
# License: BSD-3-Clause

# sphinx_gallery_thumbnail_number = 9

# %%
# .. contents:: Table of Contents
#    :local:
#
# First, we download resting-state EEG data from a Parkinson's patient
# from OpenNeuro. We will do this using ``openneuro-py`` which can be
# installed with the command ``pip install openneuro-py``.

import os.path as op

import numpy as np
import matplotlib.pyplot as plt
import openneuro

import mne
import autoreject

dataset = 'ds002778'  # The id code on OpenNeuro for this example dataset
subject_id = 'pd14'

target_dir = op.join(op.dirname(autoreject.__file__), '..', 'examples')
openneuro.download(dataset=dataset, target_dir=target_dir,
                   include=[f'sub-{subject_id}/ses-off'])

# %%
# We will now load in the raw data from the bdf file downloaded from OpenNeuro
# and, since this is resting-state data without any events, make regularly
# spaced events with which to epoch the raw data. In the averaged plot,
# we can see that there may be some eyeblink
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
# Autoreject without any other preprocessing
# ------------------------------------------
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
                           n_jobs=1, verbose=True)
ar.fit(epochs[:20])  # fit on a few epochs to save time
epochs_ar, reject_log = ar.transform(epochs, return_log=True)

# %%
# visualize the dropped epochs
epochs[reject_log.bad_epochs].plot(scalings=dict(eeg=100e-6))

# %%
# and the reject log
reject_log.plot('horizontal')

# %%
# Autoreject with high-pass filter
# --------------------------------
# The data may be very valuable and the time for the experiment
# limited and so we may want to take steps to reduce the number of
# epochs dropped by first using other steps to preprocess the data.
# We will use a high-pass filter first to remove slow drift that could
# cause epochs to be dropped.
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
                           n_jobs=1, verbose=True)
ar.fit(epochs[:20])  # fit on a few epochs to save time
epochs_ar, reject_log = ar.transform(epochs, return_log=True)

# %%
# visualize the dropped epochs
epochs[reject_log.bad_epochs].plot(scalings=dict(eeg=100e-6))

# %%
# and the reject log. As we can see in the plot, high-pass filtering reduced
# the number of epochs marked as bad by autoreject substantially.
reject_log.plot('horizontal')

# %%
# ICA
# ---
# Finally, we can apply independent components analysis (ICA) to remove
# eyeblinks from the data. If our analysis were to be very dependent on
# sensors at the front of the head, we could skip ICA and use the previous
# result. However, ICA can increase the amount of usable data by applying
# a spatial filter that downscales the data in sensors most affected by eyeblink
# artifacts.
#
# Note that ICA works best if bad segments of the data are removed
# Hence, we will remove the bad segments from the
# previous run of autoreject for the benefit of the ICA algorithm.

# compute ICA
ica = mne.preprocessing.ICA(random_state=99)
ica.fit(epochs[~reject_log.bad_epochs])

# %%
# We can see in the plots below that ICA effectively removed eyeblink
# artifact.
#
# plot source components to see which is made up of blinks
exclude = [0,  # blinks
           2  # saccades
           ]
ica.plot_components(exclude)
ica.exclude = exclude

# %%
# plot with and without eyeblink component
ica.plot_overlay(epochs.average(), exclude=ica.exclude)
ica.apply(epochs, exclude=ica.exclude)

# %%
# Autoreject with highpass filter and ICA
# ---------------------------------------
# We can see in this section that preprocessing, especially ICA, can be made
# to do a lot of the heavy lifting. There isn't a huge difference when viewing
# the averaged data because the ICA effectively limited the number
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
                           n_jobs=1, verbose=True)
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
# Next, we will visualize the cleaned average data and compare it against
# the bad segments.
evoked_bad = epochs[reject_log.bad_epochs].average()
plt.figure()
plt.plot(evoked_bad.times, evoked_bad.data.T * 1e6, 'r', zorder=-1)
epochs_ar.average().plot(axes=plt.gca())

# %%
# Finally, don't forget that we are working with resting state data
# here. Here we used long epochs of 3 seconds so that frequency-domain
# analysis was possible with the epochs. However, this could also lead
# to longer segments of the data being rejected. If you want more
# fine-grained control over the artifacts, you can
# construct shorter epochs and use the autoreject log to mark
# annotations in MNE that can be used to reject the data during doing
# time-frequency analysis. We want to emphasize that there
# is no subsitute for visual inspection. Irrespective of the rejection
# method used, we highly recommend users to inspect their preprocessed
# data before further analyses.
