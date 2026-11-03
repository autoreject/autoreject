# Author: Mainak Jas <mainak.jas@telecom-paristech.fr>
# License: BSD-3-Clause

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

import mne
import mne.channels.interpolation
from mne.datasets import testing
from mne import io

from autoreject.utils import clean_by_interp, interpolate_bads
from autoreject.utils import _interpolate_bads_eeg

data_path = testing.data_path(download=False)
raw_fname = data_path / 'MEG' / 'sample' / 'sample_audvis_trunc_raw.fif'
evoked_fname = data_path / 'MEG' / 'sample' / 'sample_audvis-ave.fif'


@testing.requires_testing_data
def test_utils():
    """Test utils."""
    raw = io.read_raw_fif(raw_fname, preload=False)
    raw.crop(0, 15)
    raw.del_proj()
    evoked = mne.read_evokeds(evoked_fname, condition='Left Auditory',
                              baseline=(None, 0))
    event_id = {'Visual/Left': 3}
    tmin, tmax = -0.2, 0.5
    events = mne.find_events(raw)
    picks = mne.pick_channels(raw.info['ch_names'],
                              ['MEG 2443', 'MEG 2442', 'MEG 2441'],
                              ordered=False)
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax,
                        picks=picks, baseline=(None, 0),
                        reject=None, preload=True)

    this_epoch = epochs.copy()
    assert this_epoch.info['bads'] == ['MEG 2443']
    epochs_clean = clean_by_interp(this_epoch)
    assert this_epoch.info['bads'] == ['MEG 2443']
    assert_array_equal(this_epoch.get_data(copy=False), epochs.get_data(copy=False))
    with pytest.raises(AssertionError):
        assert_array_equal(
            epochs_clean.get_data(copy=False),
            this_epoch.get_data(copy=False),
        )

    picks_meg = mne.pick_types(evoked.info, meg='grad', eeg=False, exclude=[])
    picks_eeg = mne.pick_types(evoked.info, meg=False, eeg=True, exclude=[])
    picks_bad_meg = mne.pick_channels(evoked.ch_names, include=['MEG 2443'])
    picks_bad_eeg = mne.pick_channels(evoked.ch_names, include=['EEG 053'])
    evoked_mne = evoked.copy().interpolate_bads(reset_bads=False)
    for picks, picks_bad in zip([picks_meg, picks_eeg],
                                [picks_bad_meg, picks_bad_eeg]):
        evoked_ar = interpolate_bads(evoked.copy(), picks=picks,
                                     reset_bads=False)
        assert evoked_ar.info['bads'] == evoked.info['bads']
        others = np.setdiff1d(np.arange(len(evoked.ch_names)), picks_bad)
        if picks is picks_eeg:  # MEG 2443 is in picks_meg, but not in picks
            assert_array_equal(evoked_ar.data[others], evoked.data[others])
        assert not np.array_equal(evoked_ar.data[picks_bad],
                                  evoked.data[picks_bad])
        # MNE uses a different origin (and all MEG channels, not only grads)
        corr = np.corrcoef(evoked_ar.data[picks_bad[0]],
                           evoked_mne.data[picks_bad[0]])[0, 1]
        assert corr > 0.99

    # autoreject's EEG interpolation is MNE's with the origin at zero
    evoked_ar = evoked.copy()
    evoked_mne = evoked.copy()
    _interpolate_bads_eeg(evoked_ar, picks=None)
    mne.channels.interpolation._interpolate_bads_eeg(
        evoked_mne, origin=(0., 0., 0.))
    assert_allclose(evoked_ar.data, evoked_mne.data, rtol=1e-7, atol=1e-20)


@testing.requires_testing_data
def test_interpolate_bads():
    """Test interpolate bads."""
    raw = io.read_raw_fif(raw_fname, preload=False)
    raw.crop(0, 15)
    raw.del_proj()
    event_id = None
    events = mne.find_events(raw)
    tmin, tmax = -0.2, 0.5
    for ii, ch_name in enumerate(raw.info['ch_names'][:14]):
        raw.set_channel_types({ch_name: 'bio'}, verbose='error')
        raw.rename_channels({ch_name: 'BIO%02d' % ii})

    picks = mne.pick_types(raw.info, meg='grad', eeg=False, eog=False)
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax,
                        baseline=(None, 0), decim=10,
                        reject=None, preload=True, verbose='error')[:10]
    epochs.info['bads'] = ['MEG 2212']
    interpolate_bads(epochs, picks)
