"""Utility functions for autoreject."""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>

import mne
from joblib import Memory

from mne.utils import ProgressBar

mem = Memory(cachedir='cachedir')


def set_matplotlib_defaults(plt):
    """Set publication quality defaults for matplotlib.

    Parameters
    ----------
    plt : instance of matplotlib.pyplot
        The plt instance.
    """
    import matplotlib
    matplotlib.style.use('ggplot')

    fontsize = 17
    params = {'axes.labelsize': fontsize + 2,
              'text.fontsize': fontsize,
              'legend.fontsize': fontsize,
              'xtick.labelsize': fontsize,
              'ytick.labelsize': fontsize,
              'axes.titlesize': fontsize + 2}
    plt.rcParams.update(params)


def clean_by_interp(inst):
    """Clean epochs/evoked by LOOCV
    """
    inst_interp = inst.copy()
    mesg = 'Creating augmented epochs'
    pbar = ProgressBar(len(inst.info['ch_names']) - 1, mesg=mesg,
                       spinner=True)
    for ch_idx, ch in enumerate(inst.info['ch_names']):
        pbar.update(ch_idx + 1)

        inst_clean = inst.copy()
        inst_clean.info['bads'] = [ch]
        interpolate_bads(inst_clean, reset_bads=True, mode='fast')

        if isinstance(inst, mne.Evoked):
            inst_interp.data[ch_idx] = inst_clean.data[ch_idx]
        elif isinstance(inst, mne.Epochs):
            inst_interp._data[:, ch_idx] = inst_clean._data[:, ch_idx]

    return inst_interp


def fetch_file(url, file_name, resume=True, timeout=10.):
    """Load requested file, downloading it if needed or requested

    Parameters
    ----------
    url: string
        The url of file to be downloaded.
    file_name: string
        Name, along with the path, of where downloaded file will be saved.
    resume: bool, optional
        If true, try to resume partially downloaded files.
    timeout : float
        The URL open timeout.
    """
    from mne.utils import _fetch_file
    return _fetch_file(url=url, file_name=file_name, print_destination=True,
                       resume=resume, hash_=None, timeout=timeout)


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
