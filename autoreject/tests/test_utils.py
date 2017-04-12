# Author: Mainak Jas <mainak.jas@telecom-paristech.fr>
# License: BSD (3-clause)

from numpy.testing import assert_array_equal

import mne
from mne.datasets import sample
from mne import io

from autoreject.utils import clean_by_interp

from nose.tools import assert_raises

data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
raw = io.read_raw_fif(raw_fname, preload=False)
raw.crop(0, 15)
raw.info['projs'] = list()


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
