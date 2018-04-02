# Author: Mainak Jas <mainak.jas@telecom-paristech.fr>
#         Denis A. Engemann <denis.engemann@gmail.com>
# License: BSD (3-clause)

from functools import partial

import numpy as np
from numpy.testing import assert_array_equal

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
raw.crop(0, 60)
raw.info['projs'] = list()


def test_global_autoreject():
    """Test global autoreject."""
    event_id = None
    tmin, tmax = -0.2, 0.5
    events = mne.find_events(raw)

    picks = mne.pick_types(raw.info, meg=True, eeg=True, stim=False,
                           eog=True, exclude=[])
    # raise error if preload is false
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax,
                        picks=picks, baseline=(None, 0),
                        reject=None, preload=False)

    # Test get_rejection_thresholds.
    reject1 = get_rejection_threshold(epochs, decim=1, random_state=42)
    reject2 = get_rejection_threshold(epochs, decim=1, random_state=42)
    reject3 = get_rejection_threshold(epochs, decim=2, random_state=42)
    tols = dict(eeg=5e-6, eog=5e-6, grad=10e-12, mag=5e-15)
    assert_true(reject1, isinstance(reject1, dict))
    for key, value in list(reject1.items()):
        assert_equal(reject1[key], reject2[key])
        assert_true(abs(reject1[key] - reject3[key]) < tols[key])


def test_autoreject():
    """Test basic LocalAutoReject functionality."""

    event_id = None
    tmin, tmax = -0.2, 0.5
    events = mne.find_events(raw)

    ##########################################################################
    # picking epochs
    include = [u'EEG %03d' % i for i in range(1, 45, 3)]
    picks = mne.pick_types(raw.info, meg=False, eeg=False, stim=False,
                           eog=True, include=include, exclude=[])
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax,
                        picks=picks, baseline=(None, 0), decim=10,
                        reject=None, preload=False)[:10]

    ar = LocalAutoReject()
    assert_raises(ValueError, ar.fit, epochs)
    epochs.load_data()

    ar.fit(epochs)
    assert_true(len(ar.picks_) == len(picks) - 1)

    # epochs with no picks.
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax,
                        baseline=(None, 0), decim=10,
                        reject=None, preload=True)[:20]
    # let's drop some channels to speed up
    pre_picks = mne.pick_types(epochs.info, meg=True, eeg=True)
    pre_picks = np.r_[
        mne.pick_types(epochs.info, meg='mag', eeg=False)[::15],
        mne.pick_types(epochs.info, meg='grad', eeg=False)[::60],
        mne.pick_types(epochs.info, meg=False, eeg=True)[::16],
        mne.pick_types(epochs.info, meg=False, eeg=False, eog=True)]
    pick_ch_names = [epochs.ch_names[pp] for pp in pre_picks]
    epochs.pick_channels(pick_ch_names)
    epochs_fit = epochs[:12]  # make sure to use different size of epochs
    epochs_new = epochs[12:]

    X = epochs_fit.get_data()
    n_epochs, n_channels, n_times = X.shape
    X = X.reshape(n_epochs, -1)

    ar = GlobalAutoReject()
    assert_raises(ValueError, ar.fit, X)
    ar = GlobalAutoReject(n_channels=n_channels)
    assert_raises(ValueError, ar.fit, X)
    ar = GlobalAutoReject(n_times=n_times)
    assert_raises(ValueError, ar.fit, X)
    ar_global = GlobalAutoReject(
        n_channels=n_channels, n_times=n_times, thresh=40e-6)
    ar_global.fit(X)

    param_name = 'thresh'
    param_range = np.linspace(40e-6, 200e-6, 10)
    assert_raises(ValueError, validation_curve, ar_global, X, None,
                  param_name, param_range)

    ##########################################################################
    # picking AutoReject

    picks = mne.pick_types(
        epochs.info, meg='mag', eeg=True, stim=False, eog=False,
        include=[], exclude=[])
    non_picks = mne.pick_types(
        epochs.info, meg='grad', eeg=False, stim=False, eog=False,
        include=[], exclude=[])
    ch_types = ['mag', 'eeg']

    ar = LocalAutoReject(picks=picks)  # XXX : why do we need this??
    assert_raises(NotImplementedError, validation_curve, ar, epochs, None,
                  param_name, param_range)

    thresh_func = partial(compute_thresholds,
                          method='bayesian_optimization',
                          random_state=42)
    ar = LocalAutoRejectCV(cv=3, picks=picks, thresh_func=thresh_func,
                           n_interpolates=[1, 2],
                           consensus=[0.5, 1])
    assert_raises(AttributeError, ar.fit, X)
    assert_raises(ValueError, ar.transform, X)
    assert_raises(ValueError, ar.transform, epochs)

    ar.fit(epochs_fit)
    annot = ar.get_reject_log(epochs_fit)
    for ch_type in ch_types:
        # test that kappa & rho are selected
        assert_true(
            ar.n_interpolate_[ch_type] in ar.n_interpolates)
        assert_true(
            ar.consensus_[ch_type] in ar.consensus)

    # test complementarity of goods and bads
    assert_array_equal(
        np.sort(np.r_[annot['bad_epochs_idx'],
                      annot['good_epochs_idx']]),
        np.arange(len(epochs_fit)))

    # test that transform does not change state of ar
    epochs_clean = ar.transform(epochs_fit)  # apply same data
    annot2 = ar.get_reject_log(epochs_fit)
    assert_array_equal(annot['fix_log'], annot2['fix_log'])
    assert_array_equal(annot['bad_epochs_idx'], annot2['bad_epochs_idx'])
    assert_array_equal(annot['good_epochs_idx'], annot2['good_epochs_idx'])

    epochs_new_clean = ar.transform(epochs_new)  # apply to new data

    annot_new = ar.get_reject_log(epochs_new)
    assert_array_equal(
        np.sort(np.r_[annot_new['bad_epochs_idx'],
                      annot_new['good_epochs_idx']]),
        np.arange(len(epochs_new)))

    assert_true(
        len(annot_new['good_epochs_idx']) != len(annot['good_epochs_idx']))

    # test fix_log and picks /channel types tracking
    ch_types_orig, picks_orig = zip(*annot_new['picks_by_type'])
    picks_orig = np.concatenate(picks_orig)
    assert_equal(ch_types, list(ch_types_orig))
    assert_array_equal(picks, picks_orig)

    # test correct entries in fix log
    assert_true(
        np.isnan(annot_new['fix_log'][:, non_picks]).sum() > 0)
    assert_true(
        np.isnan(annot_new['fix_log'][:, picks]).sum() == 0)
    assert_equal(annot_new['fix_log'].shape,
                 (len(epochs_new), len(epochs_new.ch_names)))

    # test correct interpolations by type
    for ch_type, this_picks in annot_new['picks_by_type']:
        interp_counts = np.sum(
            annot_new['fix_log'][:, this_picks] == 2, axis=1)
        interp_channels = [len(cc) for cc in
                           annot_new['interp_channels'][ch_type]]
        # print(annot_new['interp_channels'][ch_type])
        assert_array_equal(interp_counts, interp_channels)

    is_same = epochs_new_clean.get_data() == epochs_new.get_data()
    if not np.isscalar(is_same):
        is_same = np.isscalar(is_same)
    assert_true(not is_same)
    assert_equal(epochs_clean.ch_names, epochs_fit.ch_names)

    assert_true(isinstance(ar.threshes_, dict))
    assert_true(len(ar.picks) == len(picks))
    assert_true(len(ar.threshes_.keys()) == len(ar.picks))
    pick_eog = mne.pick_types(epochs.info, meg=False, eeg=False, eog=True)[0]
    assert_true(epochs.ch_names[pick_eog] not in ar.threshes_.keys())
    assert_raises(
        IndexError, ar.transform,
        epochs.copy().pick_channels(
            [epochs.ch_names[pp] for pp in picks[:3]]))

    epochs.load_data()
    assert_raises(ValueError, compute_thresholds, epochs, 'dfdfdf')
    index, ch_names = zip(*[(ii, epochs_fit.ch_names[pp])
                          for ii, pp in enumerate(picks)])
    threshes_a = compute_thresholds(
        epochs_fit, picks=picks, method='random_search')
    assert_equal(set(threshes_a.keys()), set(ch_names))
    threshes_b = compute_thresholds(
        epochs_fit, picks=picks, method='bayesian_optimization')
    assert_equal(set(threshes_b.keys()), set(ch_names))
