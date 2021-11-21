# Author: Mainak Jas <mainak.jas@telecom-paristech.fr>
# License: BSD-3-Clause

import pytest

import numpy as np
import mne
from mne.datasets import sample
from mne import io

from autoreject import Ransac

import matplotlib
matplotlib.use('Agg')

data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
raw = io.read_raw_fif(raw_fname, preload=False)
raw.crop(0, 15)
raw.del_proj()


def test_ransac():
    """Some basic tests for ransac."""
    event_id = {'Visual/Left': 3}
    tmin, tmax = -0.2, 0.5

    events = mne.find_events(raw)
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax,
                        baseline=(None, 0), decim=8,
                        reject=None, preload=True)
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
    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=True,
                           eog=False, exclude=[])
    ransac = Ransac(picks=picks)
    pytest.raises(ValueError, ransac.fit, epochs)
