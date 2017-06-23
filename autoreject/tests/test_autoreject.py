# Author: Mainak Jas <mainak.jas@telecom-paristech.fr>
# License: BSD (3-clause)

import numpy as np

import mne
from mne.datasets import sample
from mne import io

from autoreject import (GlobalAutoReject, LocalAutoReject, LocalAutoRejectCV,
                        compute_thresholds, validation_curve,
                        get_rejection_threshold)

from nose.tools import assert_raises, assert_true, assert_equal

import matplotlib
matplotlib.use('Agg')


data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
raw = io.read_raw_fif(raw_fname, preload=False)
raw.crop(0, 20)
raw.info['projs'] = list()


def test_autoreject():
    """Some basic tests for autoreject."""

    event_id = {'Visual/Left': 3}
    tmin, tmax = -0.2, 0.5
    events = mne.find_events(raw)

    ##########################################################################
    # picking epochs
    include = [u'EEG %03d' % i for i in range(1, 45, 3)]
    picks = mne.pick_types(raw.info, meg=False, eeg=False, stim=False,
                           eog=True, include=include, exclude=[])

    # raise error if preload is false
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax,
                        picks=picks, baseline=(None, 0), decim=8,
                        reject=None, preload=False)
    ar = LocalAutoReject()
    assert_raises(ValueError, ar.fit, epochs)
    epochs.load_data()
    ar.fit(epochs)
    assert_true(len(ar.picks) == len(picks) - 1)

    ##########################################################################
    # picking AutoReject
    picks = mne.pick_types(raw.info, meg=False, eeg=False, stim=False,
                           eog=False, include=include, exclude=[])

    epochs = mne.Epochs(raw, events, event_id, tmin, tmax,
                        baseline=(None, 0), decim=8,
                        reject=None, preload=True)

    X = epochs.get_data()
    n_epochs, n_channels, n_times = X.shape
    X = X.reshape(n_epochs, -1)

    ar = GlobalAutoReject()
    assert_raises(ValueError, ar.fit, X)
    ar = GlobalAutoReject(n_channels=n_channels)
    assert_raises(ValueError, ar.fit, X)
    ar = GlobalAutoReject(n_times=n_times)
    assert_raises(ValueError, ar.fit, X)
    ar = GlobalAutoReject(n_channels=n_channels, n_times=n_times,
                          thresh=40e-6)
    ar.fit(X)

    reject = get_rejection_threshold(epochs)
    assert_true(reject, isinstance(reject, dict))

    param_name = 'thresh'
    param_range = np.linspace(40e-6, 200e-6, 10)
    assert_raises(ValueError, validation_curve, ar, X, None,
                  param_name, param_range)

    ar = LocalAutoReject(picks=picks)
    assert_raises(NotImplementedError, validation_curve, ar, epochs, None,
                  param_name, param_range)

    ar = LocalAutoRejectCV(cv=3, picks=picks)
    assert_raises(AttributeError, ar.fit, X)
    assert_raises(ValueError, ar.transform, X)
    assert_raises(ValueError, ar.transform, epochs)

    ar.fit(epochs)
    assert_true(isinstance(ar.threshes_, dict))
    assert_true(len(ar.picks) == len(picks))
    assert_true(len(ar.threshes_.keys()) == len(ar.picks))
    pick_eog = mne.pick_types(epochs.info, meg=False, eeg=False, eog=True)
    assert_true(epochs.ch_names[pick_eog] not in ar.threshes_.keys())
    assert_raises(
        IndexError, ar.transform,
        epochs.copy().pick_channels(
            [epochs.ch_names[pp] for pp in picks[:3]]))

    assert_equal(ar.bad_segments.shape[1], len(epochs.ch_names))
    assert_true(np.any(ar.bad_segments[:, picks]))
    anti_picks = np.ones(len(epochs.ch_names), dtype=bool)
    anti_picks[picks] = False
    assert_true(not np.any(ar.bad_segments[:, anti_picks]))

    epochs.load_data()
    assert_raises(ValueError, compute_thresholds, epochs, 'dfdfdf')
    for method in ['random_search', 'bayesian_optimization']:
        compute_thresholds(epochs, picks=picks, method=method)
