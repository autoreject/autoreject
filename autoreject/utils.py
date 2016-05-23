import numpy as np

import mne
from mne.channels import read_montage

from joblib import Memory

mem = Memory(cachedir='cachedir')


def scorer(evoked_gt, epochs_test, good_epochs_idx=None):
    """ Returns the error percentage. """
    import numpy as np
    if good_epochs_idx is None:
        evoked_test = epochs_test.average()
        return np.sqrt(np.mean((np.mean(evoked_gt.data, axis=0)
                       - evoked_test.data) ** 2))
    elif len(good_epochs_idx) > 0:
        epochs_test = epochs_test[good_epochs_idx]
        evoked_test = epochs_test.average()
        return np.sqrt(np.mean((np.mean(evoked_gt.data, axis=0)
                       - evoked_test.data) ** 2))
    else:
        return np.inf


def get_bad_idx(epochs, reject):
    """Get indices of bad epochs.
    """
    deltas = np.array([(np.max(d, axis=1) - np.min(d, axis=1))
                      for d in epochs.get_data()])
    bad_epochs_score = np.max(deltas, axis=-1)
    bad_epochs_idx = np.where(bad_epochs_score > reject)[0]
    sorted_epoch_idx = bad_epochs_score.argsort()[::-1]
    return bad_epochs_idx, sorted_epoch_idx


def clean_by_interp(inst):
    """Clean epochs/evoked by LOOCV
    """
    inst_interp = inst.copy()
    for ch_idx, ch in enumerate(inst.info['ch_names']):
        print('Cleaning ch %s' % ch)
        if isinstance(inst, mne.Evoked):
            ch_orig = inst.data[ch_idx].copy()
        elif isinstance(inst, mne.Epochs):
            ch_orig = inst._data[:, ch_idx].copy()

        inst.info['bads'] = [ch]
        interpolate_bads(inst, reset_bads=True, mode='fast')

        if isinstance(inst, mne.Evoked):
            inst_interp.data[ch_idx] = inst.data[ch_idx]
            inst.data[ch_idx] = ch_orig
        elif isinstance(inst, mne.Epochs):
            inst_interp._data[:, ch_idx] = inst._data[:, ch_idx]
            inst._data[:, ch_idx] = ch_orig

    return inst_interp


def _set_raw_montage(raw):
    """Set montage for the raw object."""
    montage = read_montage('standard_1005')
    raw.rename_channels(lambda x: str(x.strip('.')).upper()
                        if str(x.strip('.')).upper() in
                        montage.ch_names else str(x.strip('.')))
    raw.rename_channels(dict(Cpz='CPz', Poz='POz', Fcz='FCz', Afz='AFz'))
    raw.set_channel_types({'Fp1': 'eog'})  # artificially make an EOG channel
    raw.set_montage(montage)


def _epochs_to_rawarray(epochs):
    """Concatenate the epochs in a raw object."""
    from mne.io import RawArray
    from mne.decoding import EpochsVectorizer
    # bring channels to the first dimension
    data = epochs.get_data().transpose((1, 0, 2))
    raw_data = EpochsVectorizer().transform(data)
    raw_array = RawArray(raw_data, epochs.info.copy())
    return raw_array


def _interpolate_epochs(epochs, bad_log):
    """Interpolate channels epochwise.

    Parameters
    ----------
    epochs : instance of mne.Epochs
        The epochs to be interpolated.
    bad_log : array, shape (n_epochs, n_channels)
    """
    from utils import interpolate_bads
    from progressbar import ProgressBar, SimpleProgress

    if len(epochs) != bad_log.shape[0]:
        raise ValueError('Shape of bad_log and epochs do not match')
    if len(epochs.info['ch_names']) != bad_log.shape[1]:
        raise ValueError('Shape of bad_log and ch_names do not match')

    pbar = ProgressBar(widgets=[SimpleProgress()])
    # XXX: last epoch may be smaller than window size
    print('Repairing epochs: ')
    for epoch_idx in pbar(range(len(epochs))):
        bad_idx = np.where(bad_log[epoch_idx] == 1)[0]
        bad_chs = [epochs.info['ch_names'][p] for p in bad_idx]
        epoch = epochs[epoch_idx].copy()
        epoch.info['bads'] = bad_chs
        interpolate_bads(epoch, reset_bads=True)
        epochs._data[epoch_idx] = epoch._data


def _interpolate_bads_meg_fast(inst, mode='accurate', verbose=None):
    """Interpolate bad channels from data in good channels.
    """
    from mne.channels.interpolation import _do_interp_dots
    from mne import pick_types, pick_channels

    picks_meg = pick_types(inst.info, meg=True, eeg=False, exclude=[])
    ch_names = [inst.info['ch_names'][p] for p in picks_meg]
    picks_good = pick_types(inst.info, meg=True, eeg=False, exclude='bads')

    # select the bad meg channel to be interpolated
    if len(inst.info['bads']) == 0:
        picks_bad = []
    else:
        picks_bad = pick_channels(ch_names, inst.info['bads'],
                                  exclude=[])

    # return without doing anything if there are no meg channels
    if len(picks_meg) == 0 or len(picks_bad) == 0:
        return

    mapping = _fast_map_meg_channels(inst, picks_good, picks_bad, mode=mode)

    _do_interp_dots(inst, mapping, picks_good, picks_bad)


def interpolate_bads(inst, reset_bads=True, mode='accurate'):
    """Interpolate bad MEG and EEG channels.
    """
    import mne
    from mne.channels.interpolation import _interpolate_bads_eeg
    mne.set_log_level('WARNING')  # to prevent cobyla printf error

    if getattr(inst, 'preload', None) is False:
        raise ValueError('Data must be preloaded.')

    _interpolate_bads_eeg(inst)
    _interpolate_bads_meg_fast(inst, mode=mode)

    if reset_bads is True:
        inst.info['bads'] = []

    return inst


def _fast_map_meg_channels(inst, pick_from, pick_to, mode='fast'):
    from mne.io.pick import pick_info
    from mne.forward._field_interpolation import _setup_dots
    from mne.forward._field_interpolation import _compute_mapping_matrix
    from mne.forward._make_forward import _create_meg_coils, _read_coil_defs
    from mne.forward._lead_dots import _do_self_dots, _do_cross_dots
    from mne.bem import _check_origin

    miss = 1e-4  # Smoothing criterion for MEG

    def _compute_dots(info, mode='fast'):
        """Compute all-to-all dots.
        """
        templates = _read_coil_defs()
        coils = _create_meg_coils(info['chs'], 'normal', info['dev_head_t'],
                                  templates)
        my_origin = _check_origin((0., 0., 0.04), info_from)
        int_rad, noise, lut_fun, n_fact = _setup_dots(mode, coils, 'meg')
        self_dots = _do_self_dots(int_rad, False, coils, my_origin, 'meg',
                                  lut_fun, n_fact, n_jobs=1)
        cross_dots = _do_cross_dots(int_rad, False, coils, coils,
                                    my_origin, 'meg', lut_fun, n_fact).T
        return self_dots, cross_dots

    _compute_fast_dots = mem.cache(_compute_dots)
    info = inst.info.copy()
    info['bads'] = []  # if bads is different, hash will be different

    info_from = pick_info(info, pick_from, copy=True)
    templates = _read_coil_defs()
    coils_from = _create_meg_coils(info_from['chs'], 'normal',
                                   info_from['dev_head_t'], templates)
    my_origin = _check_origin((0., 0., 0.04), info_from)
    int_rad, noise, lut_fun, n_fact = _setup_dots(mode, coils_from, 'meg')

    self_dots, cross_dots = _compute_fast_dots(info, mode=mode)

    cross_dots = cross_dots[pick_to, :][:, pick_from]
    self_dots = self_dots[pick_from, :][:, pick_from]

    ch_names = [c['ch_name'] for c in info_from['chs']]
    fmd = dict(kind='meg', ch_names=ch_names,
               origin=my_origin, noise=noise, self_dots=self_dots,
               surface_dots=cross_dots, int_rad=int_rad, miss=miss)
    fmd['data'] = _compute_mapping_matrix(fmd, info_from)

    return fmd['data']


def map_meg_loocv_channels(inst, pick_from, pick_to, self_dots=None,
                           cross_dots=None, mode='fast'):

    from mne.io.pick import pick_info
    from mne.forward._lead_dots import _do_self_dots, _do_cross_dots
    from mne.forward._make_forward import _create_meg_coils, _read_coil_defs
    from mne.forward._field_interpolation import _setup_dots
    from mne.forward._field_interpolation import _compute_mapping_matrix
    from mne.bem import _check_origin

    info_from = pick_info(inst.info, pick_from, copy=True)
    info_to = pick_info(inst.info, pick_to, copy=True)

    # no need to apply trans because both from and to coils are in device
    # coordinates
    templates = _read_coil_defs(verbose=False)
    coils_from = _create_meg_coils(info_from['chs'], 'normal',
                                   info_from['dev_head_t'], templates)
    coils_to = _create_meg_coils(info_to['chs'], 'normal',
                                 info_to['dev_head_t'], templates)
    miss = 1e-4  # Smoothing criterion for MEG

    int_rad, noise, lut_fun, n_fact = _setup_dots(mode, coils_from, 'meg')
    my_origin = _check_origin((0., 0., 0.04), info_from)

    if self_dots is None:
        self_dots = _do_self_dots(int_rad, False, coils_from, my_origin, 'meg',
                                  lut_fun, n_fact, n_jobs=1)
    if cross_dots is None:
        cross_dots = _do_cross_dots(int_rad, False, coils_from, coils_to,
                                    my_origin, 'meg', lut_fun, n_fact).T

    ch_names = [c['ch_name'] for c in info_from['chs']]
    fmd = dict(kind='meg', ch_names=ch_names,
               origin=my_origin, noise=noise, self_dots=self_dots,
               surface_dots=cross_dots, int_rad=int_rad, miss=miss)
    fmd['data'] = _compute_mapping_matrix(fmd, info_from)

    return fmd['data'], self_dots, cross_dots
