# Author: Mainak Jas <mainak.jas@telecom-paristech.fr>
# License: BSD (3-clause)

import numpy as np
from numpy.testing import assert_array_equal

import mne
from mne.datasets import sample
from mne import io

from autoreject import GlobalAutoReject, LocalAutoReject, validation_curve
from autoreject.utils import clean_by_interp
from autoreject.viz import plot_epochs

from nose.tools import assert_raises

import matplotlib
matplotlib.use('Agg')

data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
raw = io.read_raw_fif(raw_fname, preload=False)
raw.crop(0, 10)
raw.info['projs'] = list()


def test_autoreject():
    """Some basic tests for autoreject."""

    event_id = {'Visual/Left': 3}
    tmin, tmax = -0.2, 0.5
    events = mne.find_events(raw)

    include = []
    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False,
                           eog=False, include=include, exclude='bads')
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax,
                        picks=picks, baseline=(None, 0),
                        reject=None, verbose=False)

    ar = GlobalAutoReject()
    X = epochs.get_data()
    n_epochs, n_channels, n_times = X.shape
    X = X.reshape(n_epochs, -1)

    assert_raises(ValueError, ar.fit, X)
    assert_raises(ValueError, ar.fit, X, n_channels)

    param_name = 'thresh'
    param_range = np.linspace(40e-6, 200e-6, 30)
    assert_raises(ValueError, validation_curve, ar, X, None,
                  param_name, param_range)

    ar = LocalAutoReject()
    assert_raises(NotImplementedError, validation_curve, ar, epochs, None,
                  param_name, param_range)


def test_utils():
    """Test utils."""

    event_id = {'Visual/Left': 3}
    tmin, tmax = -0.2, 0.5
    events = mne.find_events(raw)
    picks = mne.pick_channels(raw.info['ch_names'],
                              ['MEG 2443', 'MEG 2442', 'MEG 2441'])
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax,
                        picks=picks, baseline=(None, 0),
                        reject=None, preload=True)

    this_epoch = epochs.copy()
    epochs_clean = clean_by_interp(this_epoch)
    assert_array_equal(this_epoch.get_data(), epochs.get_data())
    assert_raises(AssertionError, assert_array_equal, epochs_clean.get_data(),
                  this_epoch.get_data())


def test_viz():
    """Test viz."""
    import matplotlib.pyplot as plt

    events = mne.find_events(raw)
    picks = mne.pick_channels(raw.info['ch_names'],
                              ['MEG 2443', 'MEG 2442', 'MEG 2441'])
    epochs = mne.Epochs(raw, events, picks=picks, baseline=(None, 0),
                        reject=None, preload=True)
    bad_epochs_idx = [0, 1, 3]
    n_epochs, n_channels, _ = epochs.get_data().shape
    fix_log = np.zeros((n_epochs, n_channels))

    plot_epochs(epochs, bad_epochs_idx=bad_epochs_idx, fix_log=fix_log)
    plt.close('all')
