"""Test autoreject."""
# Author: Mainak Jas <mainak.jas@telecom-paristech.fr>
#         Denis A. Engemann <denis.engemann@gmail.com>
#         Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#
# License: BSD-3-Clause

import os.path as op
import pickle
import platform

import numpy as np
from numpy.testing import assert_array_equal
import pytest

import mne
from mne.datasets import testing
from mne import io
from mne.utils import _TempDir

from autoreject import (_GlobalAutoReject, _AutoReject, AutoReject,
                        compute_thresholds, validation_curve,
                        get_rejection_threshold, read_auto_reject,
                        read_reject_log)
from autoreject.utils import _get_picks_by_type
from autoreject.autoreject import _get_interp_chs

data_path = testing.data_path(download=False)
raw_fname = data_path / 'MEG' / 'sample' / 'sample_audvis_trunc_raw.fif'

fname_nirx = op.join(
    data_path, 'NIRx', 'nirscout', 'nirx_15_2_recording_w_short')

ignore_decim = pytest.mark.filterwarnings(
    'ignore:The measurement information indicates a low-pass.*:RuntimeWarning')


@ignore_decim
@testing.requires_testing_data
def test_global_autoreject():
    """Test global autoreject."""
    raw = io.read_raw_fif(raw_fname, preload=False)
    raw.del_proj()
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
    if platform.system().lower().startswith("win"):  # pragma: no cover
        # XXX: When testing on Windows, the precision seemed to be lower. Why?
        tols = dict(eeg=9e-5, eog=9e-5, grad=10e-12, mag=5e-15)
    assert reject1, isinstance(reject1, dict)
    for key, value in list(reject1.items()):
        assert reject1[key] == reject2[key]
        assert abs(reject1[key] - reject3[key]) < tols[key]

    reject = get_rejection_threshold(epochs, decim=4, ch_types='eeg')
    assert 'eog' not in reject
    assert 'eeg' in reject
    pytest.raises(ValueError, get_rejection_threshold, epochs,
                  decim=4, ch_types=5)


@ignore_decim
@testing.requires_testing_data
def test_autoreject():
    """Test basic _AutoReject functionality."""
    raw = io.read_raw_fif(raw_fname, preload=False)
    raw.del_proj()
    raw.info['bads'] = []
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

    ar = _AutoReject()
    pytest.raises(ValueError, ar.fit, epochs)
    epochs.load_data()

    ar.fit(epochs)
    assert len(ar.picks_) == len(picks) - 1

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
    bad_ch_names = [epochs.ch_names[ix] for ix in range(len(epochs.ch_names))
                    if ix not in pre_picks]
    epochs_with_bads = epochs.copy()
    epochs_with_bads.info['bads'] = bad_ch_names
    epochs.pick_channels(pick_ch_names)

    epochs_fit = epochs[:12]  # make sure to use different size of epochs
    epochs_new = epochs[12:]
    epochs_with_bads_fit = epochs_with_bads[:12]

    X = epochs_fit.get_data()
    n_epochs, n_channels, n_times = X.shape
    X = X.reshape(n_epochs, -1)

    ar = _GlobalAutoReject()
    pytest.raises(ValueError, ar.fit, X)
    ar = _GlobalAutoReject(n_channels=n_channels)
    pytest.raises(ValueError, ar.fit, X)
    ar = _GlobalAutoReject(n_times=n_times)
    pytest.raises(ValueError, ar.fit, X)
    ar_global = _GlobalAutoReject(
        n_channels=n_channels, n_times=n_times, thresh=40e-6)
    ar_global.fit(X)

    param_range = np.linspace(40e-6, 200e-6, 10)

    train_scores, test_scores = \
        validation_curve(epochs_fit, param_range=param_range)
    assert len(train_scores) == len(test_scores)

    train_scores, test_scores, param_range = \
        validation_curve(epochs_fit, return_param_range=True)
    assert len(train_scores) == len(test_scores) == len(param_range)

    pytest.raises(ValueError, validation_curve, X, param_range=param_range)

    ##########################################################################
    # picking AutoReject

    picks = mne.pick_types(
        epochs.info, meg='mag', eeg=True, stim=False, eog=False,
        include=[], exclude=[])
    non_picks = mne.pick_types(
        epochs.info, meg='grad', eeg=False, stim=False, eog=False,
        include=[], exclude=[])
    ch_types = ['mag', 'eeg']

    ar = _AutoReject(picks=picks)  # XXX : why do we need this??

    ar = AutoReject(cv=3, picks=picks, random_state=42,
                    n_interpolate=[1, 2], consensus=[0.5, 1])
    pytest.raises(AttributeError, ar.fit, X)
    pytest.raises(ValueError, ar.transform, X)
    pytest.raises(ValueError, ar.transform, epochs)
    epochs_nochs = epochs_fit.copy()
    # just one channel loc is nan or all channel locs are 0.
    # Should raise error in both cases
    epochs_nochs.info['chs'][1]['loc'][:] = np.nan
    pytest.raises(RuntimeError, ar.fit, epochs_nochs)
    for ch in epochs_nochs.info['chs']:
        ch['loc'] = np.zeros_like(ch['loc'])
    pytest.raises(RuntimeError, ar.fit, epochs_nochs)
    ar2 = AutoReject(cv=3, picks=picks, random_state=42,
                     n_interpolate=[1, 2], consensus=[0.5, 1],
                     verbose='blah')
    with pytest.raises(ValueError, match='boolean'):
        ar2.fit(epochs_fit)

    ar.fit(epochs_fit)
    reject_log = ar.get_reject_log(epochs_fit)
    for ch_type in ch_types:
        # test that kappa & rho are selected
        assert ar.n_interpolate_[ch_type] in ar.n_interpolate
        assert ar.consensus_[ch_type] in ar.consensus

        assert (ar.n_interpolate_[ch_type] ==
                ar.local_reject_[ch_type].n_interpolate_[ch_type])
        assert (ar.consensus_[ch_type] ==
                ar.local_reject_[ch_type].consensus_[ch_type])

    # test complementarity of goods and bads
    assert_array_equal(len(reject_log.bad_epochs), len(epochs_fit))

    # test that transform does not change state of ar
    epochs_clean = ar.transform(epochs_fit)  # apply same data
    assert repr(ar)
    assert repr(ar.local_reject_)
    reject_log2 = ar.get_reject_log(epochs_fit)
    assert_array_equal(reject_log.labels, reject_log2.labels)
    assert_array_equal(reject_log.bad_epochs, reject_log2.bad_epochs)
    assert_array_equal(reject_log.ch_names, reject_log2.ch_names)

    epochs_new_clean = ar.transform(epochs_new)  # apply to new data

    reject_log_new = ar.get_reject_log(epochs_new)
    assert_array_equal(len(reject_log_new.bad_epochs), len(epochs_new))

    assert len(reject_log_new.bad_epochs) != len(reject_log.bad_epochs)

    picks_by_type = _get_picks_by_type(epochs.info, ar.picks)
    # test correct entries in fix log
    assert np.isnan(reject_log_new.labels[:, non_picks]).sum() > 0
    assert np.isnan(reject_log_new.labels[:, picks]).sum() == 0
    assert (reject_log_new.labels.shape ==
            (len(epochs_new), len(epochs_new.ch_names)))

    # test correct interpolations by type
    for ch_type, this_picks in picks_by_type:
        interp_counts = np.sum(
            reject_log_new.labels[:, this_picks] == 2, axis=1)
        labels = reject_log_new.labels.copy()
        not_this_picks = np.setdiff1d(np.arange(labels.shape[1]), this_picks)
        labels[:, not_this_picks] = np.nan
        interp_channels = _get_interp_chs(
            labels, reject_log.ch_names, this_picks)
        assert_array_equal(
            interp_counts, [len(cc) for cc in interp_channels])

    assert len(epochs_new_clean.get_data()) != len(epochs_new.get_data())

    # test that transform can take new reject_log
    reject_log1 = ar.get_reject_log(epochs)
    assert reject_log1.bad_epochs.sum() > 0  # ensure > 1 bad epoch
    reject_log1.bad_epochs[:] = False
    epochs_nobad = ar.transform(epochs, reject_log=reject_log1)
    assert len(epochs_nobad) == len(epochs)
    pytest.raises(ValueError, ar.transform, epochs, reject_log='blah')

    # test that transform ignores bad channels
    epochs_with_bads_fit.pick_types(meg='mag', eeg=True, eog=True, exclude=[])
    ar_bads = AutoReject(cv=3, random_state=42,
                         n_interpolate=[1, 2], consensus=[0.5, 1])
    with pytest.warns(UserWarning, match='151 channels are marked as bad'):
        ar_bads.fit(epochs_with_bads_fit)
    with pytest.warns(UserWarning, match='151 channels are marked as bad'):
        epochs_with_bads_clean = ar_bads.transform(epochs_with_bads_fit)

    good_w_bads_ix = mne.pick_types(epochs_with_bads_clean.info,
                                    meg='mag', eeg=True, eog=True,
                                    exclude='bads')
    good_wo_bads_ix = mne.pick_types(epochs_clean.info,
                                     meg='mag', eeg=True, eog=True,
                                     exclude='bads')
    assert_array_equal(epochs_with_bads_clean.get_data()[:, good_w_bads_ix, :],
                       epochs_clean.get_data()[:, good_wo_bads_ix, :])

    bad_ix = [epochs_with_bads_clean.ch_names.index(ch)
              for ch in epochs_with_bads_clean.info['bads']]
    epo_ix = ~ar_bads.get_reject_log(epochs_with_bads_fit).bad_epochs
    assert_array_equal(
        epochs_with_bads_clean.get_data()[:, bad_ix, :],
        epochs_with_bads_fit.get_data()[epo_ix, :, :][:, bad_ix, :])

    assert epochs_clean.ch_names == epochs_fit.ch_names

    assert isinstance(ar.threshes_, dict)
    assert len(ar.picks) == len(picks)
    assert len(ar.threshes_.keys()) == len(ar.picks)
    pick_eog = mne.pick_types(epochs.info, meg=False, eeg=False, eog=True)[0]
    assert epochs.ch_names[pick_eog] not in ar.threshes_.keys()
    pytest.raises(
        IndexError, ar.transform,
        epochs.copy().pick_channels(
            [epochs.ch_names[pp] for pp in picks[:3]]))

    epochs.load_data()
    pytest.raises(ValueError, compute_thresholds, epochs, 'dfdfdf')
    index, ch_names = zip(*[(ii, epochs_fit.ch_names[pp])
                            for ii, pp in enumerate(picks)])
    threshes_a = compute_thresholds(
        epochs_fit, picks=picks, method='random_search')
    assert set(threshes_a.keys()) == set(ch_names)
    threshes_b = compute_thresholds(
        epochs_fit, picks=picks, method='bayesian_optimization')
    assert set(threshes_b.keys()) == set(ch_names)


@ignore_decim
@testing.requires_testing_data
def test_io():
    """Test IO functionality."""
    raw = io.read_raw_fif(raw_fname, preload=False)
    raw.del_proj()
    event_id = None
    tmin, tmax = -0.2, 0.5
    events = mne.find_events(raw)
    savedir = _TempDir()
    fname = op.join(savedir, 'autoreject.hdf5')

    include = [u'EEG %03d' % i for i in range(1, 45, 3)]
    picks = mne.pick_types(raw.info, meg=False, eeg=False, stim=False,
                           eog=True, include=include, exclude=[])

    # raise error if preload is false
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax,
                        picks=picks, baseline=(None, 0), decim=4,
                        reject=None, preload=True)[:10]
    ar = AutoReject(cv=2, random_state=42, n_interpolate=[1],
                    consensus=[0.5], verbose=False)
    ar.save(fname)  # save without fitting
    pkl_ar = pickle.dumps(ar)  # also, pickle without fitting

    # check that fit after saving is the same as fit without saving/pickling
    ar2 = read_auto_reject(fname)
    ar3 = pickle.loads(pkl_ar)
    ar.fit(epochs)
    ar2.fit(epochs)
    ar3.fit(epochs)
    assert np.sum([ar.threshes_[k] - ar2.threshes_[k]
                   for k in ar.threshes_.keys()]) == 0.
    assert np.sum([ar.threshes_[k] - ar3.threshes_[k]
                   for k in ar.threshes_.keys()]) == 0.

    pytest.raises(ValueError, ar.save, fname)
    ar.save(fname, overwrite=True)
    pkl_ar2 = pickle.dumps(ar)

    ar4 = read_auto_reject(fname)
    ar5 = pickle.loads(pkl_ar2)

    epochs_clean1, reject_log1 = ar.transform(epochs, return_log=True)
    epochs_clean2, reject_log2 = ar4.transform(epochs, return_log=True)
    epochs_clean3, reject_log3 = ar5.transform(epochs, return_log=True)

    assert_array_equal(epochs_clean1.get_data(), epochs_clean2.get_data())
    assert_array_equal(epochs_clean1.get_data(), epochs_clean3.get_data())
    assert_array_equal(reject_log1.labels, reject_log2.labels)
    assert_array_equal(reject_log1.labels, reject_log3.labels)

    reject_log1.save(op.join(savedir, 'reject_log.npz'))
    reject_log4 = read_reject_log(op.join(savedir, 'reject_log.npz'))
    assert_array_equal(reject_log1.labels, reject_log4.labels)
    assert_array_equal(reject_log1.bad_epochs, reject_log4.bad_epochs)
    assert all(reject_log1.ch_names == reject_log4.ch_names)


@testing.requires_testing_data
def test_fnirs():
    """Test that autoreject runs on fNIRS data."""
    raw = mne.io.read_raw_nirx(fname_nirx)
    raw = mne.preprocessing.nirs.optical_density(raw)
    raw = mne.preprocessing.nirs.beer_lambert_law(raw)
    events, _ = mne.events_from_annotations(raw)
    epochs = mne.Epochs(raw, events,
                        tmin=-1, tmax=1,
                        proj=True, baseline=(None, 0), preload=True,
                        detrend=None, verbose=True)
    # Test autoreject
    ar = AutoReject(cv=3, consensus=[0.1])
    assert len(epochs) == 3
    epochs_clean = ar.fit_transform(epochs)
    assert len(epochs_clean) < len(epochs)
    # Test threshold extraction
    reject = get_rejection_threshold(epochs, cv=3)
    assert "hbo" in reject.keys()
    assert "hbr" in reject.keys()
    assert reject["hbo"] < 0.001  # This is a very high value as sanity check
    assert reject["hbr"] < 0.001
    assert reject["hbr"] > 0.0


@testing.requires_testing_data
@pytest.mark.parametrize('ch_type', ['ecog', 'seeg'])
def test_ecog(ch_type):
    """Test that autoreject runs on ECoG and sEEG data."""
    raw = mne.io.read_raw(raw_fname).del_proj().pick('eeg')
    raw.info['bads'] = []
    # setting the channel types
    ch_dict = {ch: ch_type for ch in raw.ch_names}
    raw.set_channel_types(ch_dict)

    # make events
    epochs = mne.make_fixed_length_epochs(raw).load_data()
    n1 = len(epochs)
    reject = get_rejection_threshold(epochs)
    epochs.drop_bad(reject=reject)
    n2 = len(epochs)
    assert ch_type in reject.keys()
    assert reject[ch_type] > 0.0
    assert reject[ch_type] < 0.01
    assert n2 < n1

    # testing that compute_thresholds is working without location data
    epochs.set_montage(None)
    rejects = compute_thresholds(epochs, augment=False)
    assert set(rejects.keys()) == set(raw.ch_names)
