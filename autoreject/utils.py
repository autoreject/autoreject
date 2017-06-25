"""Utility functions for autoreject."""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>

import numpy as np
import mne
from sklearn.externals.joblib import Memory

mem = Memory(cachedir='cachedir')


def _handle_picks(info, picks):
    """Pick the data channls or return picks."""
    if picks is None:
        out = mne.pick_types(
            info, meg=True, eeg=True, ref_meg=False, exclude='bads')
    else:
        out = picks
    return out


def set_matplotlib_defaults(plt, style='ggplot'):
    """Set publication quality defaults for matplotlib.

    Parameters
    ----------
    plt : instance of matplotlib.pyplot
        The plt instance.
    """
    import matplotlib
    matplotlib.style.use(style)

    fontsize = 17
    params = {'axes.labelsize': fontsize + 2,
              'text.fontsize': fontsize,
              'legend.fontsize': fontsize,
              'xtick.labelsize': fontsize,
              'ytick.labelsize': fontsize,
              'axes.titlesize': fontsize + 2}
    plt.rcParams.update(params)


def _pbar(iterable, desc, leave=True, position=None, verbose='progressbar'):
    if verbose == 'progressbar':
        from mne.utils import ProgressBar

        _ProgressBar = ProgressBar
        if not mne.utils.check_version('mne', '0.14dev0'):
            class _ProgressBar(ProgressBar):
                def __iter__(self):
                    """Iterate to auto-increment the pbar with 1."""
                    self.max_value = len(iterable)
                    for obj in iterable:
                        yield obj
                    self.update_with_increment_value(1)

        pbar = _ProgressBar(iterable, mesg=desc, spinner=True)
    elif verbose == 'tqdm':
        from tqdm import tqdm
        pbar = tqdm(iterable, desc=desc, leave=leave, position=position,
                    dynamic_ncols=True)
    elif verbose == 'tqdm_notebook':
        from tqdm import tqdm_notebook
        pbar = tqdm_notebook(iterable, desc=desc, leave=leave,
                             position=position, dynamic_ncols=True)
    elif verbose is False:
        pbar = iterable
    return pbar


def _get_epochs_type():
    if hasattr(mne.epochs, '_BaseEpochs'):
        BaseEpochs = mne.epochs._BaseEpochs
    else:
        BaseEpochs = mne.epochs.BaseEpochs
    return BaseEpochs


def clean_by_interp(inst, picks=None, verbose='progressbar'):
    """Clean epochs/evoked by LOOCV.

    Parameters
    ----------
    inst : instance of mne.Evoked or mne.Epochs
        The evoked or epochs object.
    picks : ndarray, shape(n_channels,) | None
        The channels to be considered for autoreject. If None, defaults
        to data channels {'meg', 'eeg'}.
    verbose : 'tqdm', 'tqdm_notebook', 'progressbar' or False
        The verbosity of progress messages.
        If `'progressbar'`, use `mne.utils.ProgressBar`.
        If `'tqdm'`, use `tqdm.tqdm`.
        If `'tqdm_notebook'`, use `tqdm.tqdm_notebook`.
        If False, suppress all output messages.

    Returns
    -------
    inst_clean : instance of mne.Evoked or mne.Epochs
        Instance after interpolation of bad channels.
    """
    inst_interp = inst.copy()
    mesg = 'Creating augmented epochs'
    picks = _handle_picks(info=inst_interp.info, picks=picks)

    BaseEpochs = _get_epochs_type()
    ch_names = [inst.info['ch_names'][p] for p in picks]
    for ch_idx, (pick, ch) in enumerate(_pbar(list(zip(picks, ch_names)),
                                        desc=mesg, verbose=verbose)):
        inst_clean = inst.copy().pick_channels(ch_names)
        inst_clean.info['bads'] = [ch]
        interpolate_bads(inst_clean, picks=picks, reset_bads=True, mode='fast')

        pick_interp = mne.pick_channels(inst_clean.info['ch_names'], [ch])[0]

        if isinstance(inst, mne.Evoked):
            inst_interp.data[pick] = inst_clean.data[pick_interp]
        elif isinstance(inst, BaseEpochs):
            inst_interp._data[:, pick] = inst_clean._data[:, pick_interp]
        else:
            raise ValueError('Unrecognized type for inst')
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
    _fetch_file(url=url, file_name=file_name, print_destination=True,
                resume=resume, hash_=None, timeout=timeout)


def interpolate_bads(inst, picks, reset_bads=True, mode='accurate'):
    """Interpolate bad MEG and EEG channels.
    """
    import mne
    mne.set_log_level('WARNING')  # to prevent cobyla printf error

    # this needs picks, assume our instance is complete and intact
    _interpolate_bads_eeg(inst, picks=picks)
    _interpolate_bads_meg_fast(inst, picks=picks, mode=mode)

    if reset_bads is True:
        inst.info['bads'] = []

    return inst


def _interpolate_bads_eeg(inst, picks):
    """Interpolate bad EEG channels.

    Operates in place.

    Parameters
    ----------
    inst : mne.io.Raw, mne.Epochs or mne.Evoked
        The data to interpolate. Must be preloaded.
    """
    from mne import pick_types
    from mne.bem import _fit_sphere
    from mne.utils import logger, warn
    from mne.channels.interpolation import (_make_interpolation_matrix,
                                            _do_interp_dots)

    # we map the full instance but we have already picked the isntance
    bads_idx = np.zeros(len(inst.ch_names), dtype=np.bool)
    goods_idx = np.zeros(len(inst.ch_names), dtype=np.bool)

    picks = mne.pick_types(inst.info, meg=True, eeg=False)
    inst.info._check_consistency()
    bads_idx[picks] = [
        inst.info['ch_names'][ch] in inst.info['bads'] for ch in picks]

    if len(picks) == 0 or len(bads_idx) == 0:
        return

    goods_idx[picks] = True
    goods_idx[bads_idx] = False

    pos = inst._get_channel_positions(picks)

    # Make sure only EEG are used
    bads_idx_pos = bads_idx[picks]
    goods_idx_pos = goods_idx[picks]

    pos_good = pos[goods_idx_pos]
    pos_bad = pos[bads_idx_pos]

    # test spherical fit
    radius, center = _fit_sphere(pos_good)
    distance = np.sqrt(np.sum((pos_good - center) ** 2, 1))
    distance = np.mean(distance / radius)
    if np.abs(1. - distance) > 0.1:
        warn('Your spherical fit is poor, interpolation results are '
             'likely to be inaccurate.')

    logger.info('Computing interpolation matrix from {0} sensor '
                'positions'.format(len(pos_good)))

    interpolation = _make_interpolation_matrix(pos_good, pos_bad)

    logger.info('Interpolating {0} sensors'.format(len(pos_bad)))
    _do_interp_dots(inst, interpolation, goods_idx, bads_idx)


def _interpolate_bads_meg_fast(inst, picks, mode='accurate', verbose=None):
    """Interpolate bad channels from data in good channels.
    """
    from mne.channels.interpolation import _do_interp_dots
    from mne import pick_types, pick_channels, pick_info

    # We can have pre-picked instances or not.
    # And we need to handle it.

    if len(inst.ch_names) > len(picks):
        picked_info = pick_info(inst.info, picks)
    else:
        picked_info = inst.info.copy()

    def get_picks_bad_good(info):
        picks_meg = pick_types(info, meg=True, eeg=False, exclude=[],
                               stim=False)
        ch_names = [info['ch_names'][p] for p in picks_meg]
        picks_good = pick_types(info, meg=True, eeg=False, exclude='bads',
                                stim=False)

        # select the bad meg channel to be interpolated
        if len(info['bads']) == 0:
            picks_bad = []
        else:
            picks_bad = pick_channels(ch_names, info['bads'],
                                      exclude=[])
        return picks_meg, picks_good, picks_bad

    picks_meg, picks_good, picks_bad = get_picks_bad_good(picked_info)

    # return without doing anything if there are no meg channels
    if len(picks_meg) == 0 or len(picks_bad) == 0:
        return

    mapping = _fast_map_meg_channels(
        picked_info, pick_from=picks_good, pick_to=picks_bad,
        mode=mode)

    # recompute picks_good, picks_bad
    _, _, picks_bad_ = get_picks_bad_good(inst.info)
    # mapping_ = np.empty(len(ch_names), len(ch_))

    _do_interp_dots(inst, mapping, picks_good, picks_bad_)


def _fast_map_meg_channels(info, pick_from, pick_to,
                           mode='fast'):
    from mne.io.pick import pick_info
    from mne.forward._field_interpolation import _setup_dots
    from mne.forward._field_interpolation import _compute_mapping_matrix
    from mne.forward._make_forward import _create_meg_coils, _read_coil_defs
    from mne.forward._lead_dots import _do_self_dots, _do_cross_dots
    from mne.bem import _check_origin

    miss = 1e-4  # Smoothing criterion for MEG

    # XXX: hack to silence _compute_mapping_matrix
    verbose = mne.get_config('MNE_LOGGING_LEVEL', 'INFO')
    mne.set_log_level('WARNING')


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

    _compute_fast_dots = mem.cache(_compute_dots, verbose=0)
    info['bads'] = []  # if bads is different, hash will be different

    info_from = pick_info(info, pick_from, copy=True)
    templates = _read_coil_defs()
    coils_from = _create_meg_coils(info_from['chs'], 'normal',
                                   info_from['dev_head_t'], templates)
    my_origin = _check_origin((0., 0., 0.04), info_from)
    int_rad, noise, lut_fun, n_fact = _setup_dots(mode, coils_from, 'meg')

    # this function needsa clean input. It hates the presence of other
    # channels
    self_dots, cross_dots = _compute_fast_dots(
        info, mode=mode)

    cross_dots = cross_dots[pick_to, :][:, pick_from]
    self_dots = self_dots[pick_from, :][:, pick_from]

    ch_names = [c['ch_name'] for c in info_from['chs']]
    fmd = dict(kind='meg', ch_names=ch_names,
               origin=my_origin, noise=noise, self_dots=self_dots,
               surface_dots=cross_dots, int_rad=int_rad, miss=miss)

    fmd['data'] = _compute_mapping_matrix(fmd, info_from)
    mne.set_log_level(verbose)

    return fmd['data']
