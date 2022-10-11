# Author: Mainak Jas <mainak.jas@telecom-paristech.fr>
# License: BSD-3-Clause
import sys

import pytest

import numpy as np
import mne
from mne.datasets import testing
from mne import io

from autoreject import Ransac

data_path = testing.data_path(download=False)
raw_fname = data_path / 'MEG' / 'sample' / 'sample_audvis_trunc_raw.fif'


@testing.requires_testing_data
def test_ransac():
    """Some basic tests for ransac."""
    raw = io.read_raw_fif(raw_fname, preload=False)
    raw.crop(0, 15)
    raw.del_proj()
    raw.info['bads'] = []

    event_id = {'Visual/Left': 3}
    tmin, tmax = -0.2, 0.5

    events = mne.find_events(raw)
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax,
                        baseline=(None, 0), decim=8,
                        reject=None, preload=True, verbose='error')
    del raw

    # normal case
    picks = mne.pick_types(epochs.info, meg='mag', eeg=False, stim=False,
                           eog=False, exclude=[])
    ransac = Ransac(picks=picks, random_state=np.random.RandomState(42))
    epochs_clean = ransac.fit_transform(epochs)
    assert len(epochs_clean) == len(epochs)

    # Pass string instead of array of idx
    picks = 'eeg'
    ransac = Ransac(picks=picks)
    ransac.fit(epochs[:2])
    expected = mne.pick_types(epochs.info, meg=False, eeg=True, stim=False,
                              eog=False, exclude='bads')
    assert (expected == ransac.picks).all()

    # Pass numpy instead of epochs
    X = epochs.get_data()
    pytest.raises(AttributeError, ransac.fit, X)

    # should not contain both channel types
    picks = mne.pick_types(epochs.info, meg=True, eeg=False, stim=False,
                           eog=False, exclude=[])
    ransac = Ransac(picks=picks)
    pytest.raises(ValueError, ransac.fit, epochs)

    # should not contain other channel types.
    picks = mne.pick_types(epochs.info, meg=False, eeg=True, stim=True,
                           eog=False, exclude=[])
    ransac = Ransac(picks=picks)
    pytest.raises(ValueError, ransac.fit, epochs)


@testing.requires_testing_data
def test_ransac_multiprocessing():
    """test on real data

    test on real data with 5 random channels augmented with strong
    50 Hz line noise.
    """
    raw = io.read_raw_fif(raw_fname, preload=False)

    raw.del_proj()

    event_id = {'Visual/Left': 3}
    tmin, tmax = -0.2, 0.5

    events = mne.find_events(raw)
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax,
                        baseline=(None, 0), reject=None, preload=True)
    # see if results are consistent when using different n_jobs
    # amplify some of the channels with noise as sanity check
    rng = np.random.RandomState(0)

    # create some artifically noisy channels with strong line noise
    mag_idx = mne.pick_types(epochs.info, meg='mag')
    noisy_idx = rng.choice(mag_idx, 5, replace=False)
    noisy_idx = sorted(noisy_idx)

    times = np.arange(0, epochs.tmax - epochs.tmin + 1 / epochs.info['sfreq'],
                      1 / epochs.info['sfreq'])
    sinewave = 10 * np.sin(2 * np.pi * 50 * times)

    epochs._data[:, noisy_idx, :] *= sinewave

    # atomic tests for _get_mappings to check if they return the same
    # result when run "parallel" for subsets or all-in-once, sanity check
    ransac = Ransac(picks=mag_idx, random_state=np.random.RandomState(42),
                    n_jobs=1, n_resample=2, min_channels=0.1)
    ch_subset = ransac._get_random_subsets(epochs.info)
    ransac.ch_type = 'meg'
    mappings = ransac._get_mappings(epochs, ch_subset)
    mappings_sub1 = ransac._get_mappings(epochs, [ch_subset[0]])
    mappings_sub2 = ransac._get_mappings(epochs, [ch_subset[1]])

    np.testing.assert_array_equal(mappings, np.concatenate([mappings_sub1,
                                                            mappings_sub2]))
    # atomic tests for _iterate_epochs to check if they return the same
    # result when run "parallel" for subsets or all-in-once, sanity check
    ransac.mappings_ = mappings
    epoch_idxs_splits = np.array_split(np.arange(len(epochs)), 2)
    corr = ransac._iterate_epochs(epochs, np.arange(len(epochs)))
    corr_sub1 = ransac._iterate_epochs(epochs, epoch_idxs_splits[0])
    corr_sub2 = ransac._iterate_epochs(epochs, epoch_idxs_splits[1])
    np.testing.assert_array_equal(corr, np.concatenate([corr_sub1,
                                                        corr_sub2]))

    # we can make it so that none are found
    ransac = Ransac(random_state=42, n_resample=10)
    epochs_nobad = epochs[:2].pick(mag_idx[5:20], exclude='bads')
    ransac.fit(epochs_nobad)
    assert len(ransac.bad_chs_) == 0

    # now test across different jobs
    mappings = dict()
    corrs = dict()
    bad_chs = dict()

    if sys.platform.startswith("win"):
        use_jobs = [1]  # joblib unpickling issues on Windows
    else:
        use_jobs = [1, 2, 3]

    for n_jobs in use_jobs:
        ransac = Ransac(picks=mag_idx, random_state=np.random.RandomState(42),
                        n_jobs=n_jobs, n_resample=50)
        with pytest.warns(UserWarning, match='2 channels are marked as'):
            ransac.fit(epochs)

        # the corr_ variable should be of shape (n_epochs, n_channels)
        assert ransac.corr_.shape == (len(epochs), len(mag_idx))
        corrs[n_jobs] = ransac.corr_
        bad_chs[n_jobs] = ransac.bad_chs_
        mappings[n_jobs] = ransac.mappings_

        # should find all noisy bad channels
        bads = [epochs.ch_names.index(ch) for ch in ransac.bad_chs_]
        for noisy in noisy_idx:
            assert noisy in bads, \
                f'bad channel {noisy} not detected, bads={bads}'

    # test if results are consistent across different jobs
    for n_jobs in corrs:
        msg = f'n_jobs={n_jobs} did not produce same results as n_jobs=1'
        # simply compare to the last computed value
        np.testing.assert_allclose(mappings[1], mappings[n_jobs], err_msg=msg)
        np.testing.assert_allclose(corrs[1], corrs[n_jobs], err_msg=msg)
        assert bad_chs[1] == bad_chs[n_jobs], msg
