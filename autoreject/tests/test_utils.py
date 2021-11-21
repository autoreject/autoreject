# Author: Mainak Jas <mainak.jas@telecom-paristech.fr>
# License: BSD-3-Clause

from numpy.testing import assert_array_equal, assert_array_almost_equal
import pytest

import mne
from mne.datasets import sample
from mne.bem import _check_origin
from mne import io

from autoreject.utils import clean_by_interp, interpolate_bads
from autoreject.utils import _interpolate_bads_eeg
import mne.channels.interpolation

data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
raw = io.read_raw_fif(raw_fname, preload=False)
raw.crop(0, 15)
raw.del_proj()

evoked_fname = data_path + '/MEG/sample/sample_audvis-ave.fif'
evoked = mne.read_evokeds(evoked_fname, condition='Left Auditory',
                          baseline=(None, 0))


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
    assert this_epoch.info['bads'] == ['MEG 2443']
    epochs_clean = clean_by_interp(this_epoch)
    assert this_epoch.info['bads'] == ['MEG 2443']
    assert_array_equal(this_epoch.get_data(), epochs.get_data())
    pytest.raises(AssertionError, assert_array_equal, epochs_clean.get_data(),
                  this_epoch.get_data())

    picks_meg = mne.pick_types(evoked.info, meg='grad', eeg=False, exclude=[])
    picks_eeg = mne.pick_types(evoked.info, meg=False, eeg=True, exclude=[])
    picks_bad_meg = mne.pick_channels(evoked.ch_names, include=['MEG 2443'])
    picks_bad_eeg = mne.pick_channels(evoked.ch_names, include=['EEG 053'])
    evoked_orig = evoked.copy()
    for picks, picks_bad in zip([picks_meg, picks_eeg],
                                [picks_bad_meg, picks_bad_eeg]):
        evoked_autoreject = interpolate_bads(evoked, picks=picks,
                                             reset_bads=False)
        evoked.interpolate_bads(reset_bads=False)
        assert_array_equal(evoked.data[picks_bad],
                           evoked_autoreject.data[picks_bad])
        pytest.raises(AssertionError, assert_array_equal,
                      evoked_orig.data[picks_bad], evoked.data[picks_bad])

    # test that autoreject EEG interpolation code behaves the same as MNE
    evoked_ar = evoked_orig.copy()
    evoked_mne = evoked_orig.copy()

    origin = _check_origin('auto', evoked_ar.info)
    _interpolate_bads_eeg(evoked_ar, picks=None)
    mne.channels.interpolation._interpolate_bads_eeg(evoked_mne, origin=origin)
    assert_array_almost_equal(evoked_ar.data, evoked_mne.data)


def test_interpolate_bads():
    """Test interpolate bads."""
    event_id = None
    events = mne.find_events(raw)
    tmin, tmax = -0.2, 0.5
    for ii, ch_name in enumerate(raw.info['ch_names'][:14]):
        raw.set_channel_types({ch_name: 'bio'})
        raw.rename_channels({ch_name: 'BIO%02d' % ii})

    picks = mne.pick_types(raw.info, meg='grad', eeg=False, eog=False)
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax,
                        baseline=(None, 0), decim=10,
                        reject=None, preload=True)[:10]
    epochs.info['bads'] = ['MEG 2212']
    interpolate_bads(epochs, picks)
