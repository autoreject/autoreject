# Author: Mainak Jas <mainak.jas@telecom-paristech.fr>
# License: BSD (3-clause)

import mne
from mne.datasets import sample
from mne import io

from autoreject import Ransac

from nose.tools import assert_raises, assert_true

data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
raw = io.read_raw_fif(raw_fname, add_eeg_ref=False, preload=False)
raw.crop(0, 15)
raw.info['projs'] = list()


def test_ransac():
    """Some basic tests for autoreject and ransac."""
    ransac = Ransac()

    event_id = {'Visual/Left': 3}
    tmin, tmax = -0.2, 0.5
    events = mne.find_events(raw)
    include = [u'EEG %03d' % i for i in range(1, 15)]
    picks = mne.pick_types(raw.info, meg=True, eeg=True, stim=False,
                           eog=False, include=include, exclude=[])
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax,
                        picks=picks, baseline=(None, 0), decim=8,
                        reject=None, add_eeg_ref=False)

    X = epochs.get_data()
    assert_raises(ValueError, ransac.fit, X)
    # should not contain both channel types
    assert_raises(ValueError, ransac.fit, epochs)
    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=True,
                           eog=False, include=include, exclude=[])
    # should not contain other channel types
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax,
                        picks=picks, baseline=(None, 0), decim=8,
                        reject=None, add_eeg_ref=False)
    assert_raises(ValueError, ransac.fit, epochs)
    # now with only one channel type
    picks = mne.pick_types(raw.info, meg=False, eeg=True)
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax,
                        picks=picks, baseline=(None, 0), decim=8,
                        reject=None, add_eeg_ref=False, preload=True)
    epochs_clean = ransac.fit_transform(epochs)
    assert_true(len(epochs_clean) == len(epochs))
