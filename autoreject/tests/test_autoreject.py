# Author: Mainak Jas <mainak.jas@telecom-paristech.fr>
# License: BSD (3-clause)

import mne
from mne.datasets import sample
from mne import io

from autoreject import GlobalAutoReject

from nose.tools import assert_raises

event_id = {'Visual/Left': 3}
tmin, tmax = -0.2, 0.5

data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
raw = io.read_raw_fif(raw_fname, preload=False)
raw.crop(0, 10)
events = mne.find_events(raw)

include = []
picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False,
                       eog=False, include=include, exclude='bads')
epochs = mne.Epochs(raw, events, event_id, tmin, tmax,
                    picks=picks, baseline=(None, 0),
                    reject=None, verbose=False)


def test_autoreject():
    """Some basic tests for autoreject."""

    ar = GlobalAutoReject()
    X = epochs.get_data()
    n_epochs, n_channels, n_times = X.shape
    X = X.reshape(n_epochs, -1)

    assert_raises(ValueError, ar.fit, X)
    assert_raises(ValueError, ar.fit, X, n_channels)
