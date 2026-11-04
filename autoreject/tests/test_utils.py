# Author: Mainak Jas <mainak.jas@telecom-paristech.fr>
# License: BSD-3-Clause

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

import mne
import mne.channels.interpolation
from mne.datasets import testing
from mne import io

from autoreject import utils
from autoreject.autoreject import _interpolate_bad_epochs
from autoreject.utils import clean_by_interp, interpolate_bads
from autoreject.utils import (_clean_by_interp, _handle_picks,
                              _interpolate_bads_eeg, _precompute_interp)

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
    with pytest.raises(ValueError, match='Unrecognized type'):
        clean_by_interp(raw)

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
def test_interp_matrices_checks(monkeypatch):
    """Test that the Info is not checked again for every set of bads."""
    if not hasattr(mne.Info, '_skip_checks'):  # pragma: no cover
        pytest.skip('Requires MNE-Python 1.13+')
    raw = io.read_raw_fif(raw_fname, preload=False)
    raw.pick('eeg', exclude='bads').crop(0, 5).load_data()
    epochs = mne.make_fixed_length_epochs(raw, duration=1., preload=True)
    n_checks = [0]
    check = mne.Info._check_consistency

    def counted(self, *args, **kwargs):
        if not getattr(self, '_no_check', False):
            n_checks[0] += 1
        return check(self, *args, **kwargs)

    monkeypatch.setattr(mne.Info, '_check_consistency', counted)
    _clean_by_interp(epochs, verbose=False)
    assert len(epochs.ch_names) > 50
    assert n_checks[0] < 10  # not once per channel


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


def _assert_close_by_type(info, got, want):
    """Compare data with a tolerance relative to the scale of each type."""
    for kind in ('meg', 'eeg'):
        picks = mne.pick_types(info, meg=kind == 'meg', eeg=kind == 'eeg')
        scale = np.abs(want[..., picks, :]).max()
        assert scale > 0
        assert_allclose(got[..., picks, :], want[..., picks, :],
                        rtol=1e-7, atol=1e-10 * scale)


@testing.requires_testing_data
@pytest.mark.parametrize('interp_data', (False, True))
def test_interpolation_one_at_a_time(interp_data, monkeypatch):
    """Test the interpolation against doing one channel or epoch at a time."""
    if interp_data and not utils._has_interp_data():
        pytest.skip('Requires MNE-Python 1.14+')
    monkeypatch.setattr(utils, '_has_interp_data', lambda: interp_data)
    raw = io.read_raw_fif(raw_fname, preload=False).crop(0, 5)
    bads = ['MEG 2443', 'EEG 053']
    assert raw.info['bads'] == bads  # kept, but not picked
    picks = np.concatenate([
        mne.pick_types(raw.info, meg='grad', exclude=())[::6],
        mne.pick_types(raw.info, meg=False, eeg=True, exclude=())[::3],
        mne.pick_types(raw.info, meg=False, eog=True)])
    epochs = mne.make_fixed_length_epochs(raw, duration=1.0, preload=True,
                                          verbose='error')
    epochs.pick([epochs.ch_names[p] for p in picks] + bads)
    # EOG first, so positions among the MEG channels differ from indices
    names = epochs.ch_names
    epochs.reorder_channels(names[-3:-2] + names[:-3] + names[-2:])
    assert epochs.ch_names[0] == 'EOG 061'
    picks = _handle_picks(epochs.info, None)
    assert not {epochs.ch_names[p] for p in picks} & set(bads)
    dots = _precompute_interp(epochs.info, picks)

    # leave-one-out, for both epochs and evoked
    for inst in (epochs, epochs.average()):
        want = inst.copy()
        for pick in picks:
            this = inst.copy()
            this.info['bads'] = [inst.ch_names[pick]]
            interpolate_bads(this, picks=picks, dots=dots, mode='fast')
            want._data[..., pick, :] = this._data[..., pick, :]
        orig = inst._data.copy()
        got = _clean_by_interp(inst, picks=picks, dots=dots, verbose=False)
        assert_array_equal(inst._data, orig)  # left as is
        assert got.info['bads'] == inst.info['bads'] == bads
        _assert_close_by_type(inst.info, got._data, want._data)

    # repairing epochs, with an epoch to leave as is and a repeated set
    names = [epochs.ch_names[p] for p in picks]
    interp_channels = [names[:2], [], names[-3:], names[:2], [names[5]]]
    want = epochs.copy()
    for ii, interp_chs in enumerate(interp_channels):
        if interp_chs:
            this = epochs[ii]
            this.info['bads'] = list(interp_chs)
            interpolate_bads(this, picks=picks, dots=dots)
            want._data[ii] = this._data[0]
    got = epochs.copy()
    _interpolate_bad_epochs(got, interp_channels=interp_channels,
                            picks=picks, dots=dots, verbose=False)
    assert got.info['bads'] == bads
    assert_array_equal(got._data[1], epochs._data[1])
    _assert_close_by_type(epochs.info, got._data, want._data)
