"""Utility functions for autoreject."""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>

import mne
import warnings
from sklearn.externals.joblib import Memory

mem = Memory(cachedir='cachedir')


def _get_ch_type_from_picks(picks, info):
    """Get the channel types from picks."""
    keys = list()
    for pp in picks:
        key = mne.io.pick.channel_type(info=info, idx=pp)
        if key not in keys:
            keys.append(key)
    return keys


def _check_data(epochs, picks, ch_constraint='data_channels',
                verbose='progressbar'):
    BaseEpochs = _get_epochs_type()
    if not isinstance(epochs, BaseEpochs):
        raise ValueError('Only accepts MNE epochs objects.')

    if epochs.preload is False:
        raise ValueError('Data must be preloaded.')
    n_bads = len(epochs.info['bads'])

    picked_info = mne.io.pick.pick_info(epochs.info, picks)
    ch_types_picked = {
        mne.io.meas_info.channel_type(picked_info, idx)
        for idx in range(len(picks))}

    if ch_constraint == 'data_channels':
        if not all(ch in ('mag', 'grad', 'eeg') for ch in ch_types_picked):
            raise ValueError('AutoReject only supports mag, grad, and eeg '
                             'at this point.')
    elif ch_constraint == 'single_channel_type':
        if sum(ch in ch_types_picked for ch in ('mag', 'grad', 'eeg')) > 1:
            raise ValueError('AutoReject only supports mag, grad, and eeg '
                             'at this point.')
    else:
        raise ValueError('bad value for ch_constraint.')

    if n_bads > 0:
        if verbose is not False:
            warnings.warn(
                '%i channels are marked as bad. These will be ignored.'
                'If you want them to be considered by autoreject please '
                'remove them from epochs.info["bads"].' % n_bads)


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
        inst_clean = inst.copy()
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
    """Load requested file, downloading it if needed or requested.

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
    """Interpolate bad MEG and EEG channels."""
    import mne
    from mne.channels.interpolation import _interpolate_bads_eeg

    mne.set_log_level('WARNING')  # to prevent cobyla printf error

    # this needs picks, assume our instance is complete and intact
    _interpolate_bads_eeg(inst)
    _interpolate_bads_meg_fast(inst, picks=picks, mode=mode)

    if reset_bads is True:
        inst.info['bads'] = []

    return inst


def _interpolate_bads_meg_fast(inst, picks, mode='accurate', verbose=None):
    """Interpolate bad channels from data in good channels."""
    from mne import pick_types, pick_channels, pick_info
    from mne.channels.interpolation import _do_interp_dots
    # We can have pre-picked instances or not.
    # And we need to handle it.

    inst_picked = True
    if len(inst.ch_names) > len(picks):
        picked_info = pick_info(inst.info, picks)
        inst_picked = False
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

    # we need to make sure that only meg channels are passed here
    # as the MNE interpolation code is not fogriving.
    # This is why we picked the info.
    mapping = _fast_map_meg_channels(
        picked_info.copy(), pick_from=picks_good, pick_to=picks_bad,
        mode=mode)
    # the downside is that the mapping matrix now does not match
    # the unpicked info of the data.
    # Since we may have picked the info, we need to double map
    # the indices.
    _, picks_good_, picks_bad_orig = get_picks_bad_good(inst.info.copy())
    ch_names_a = [picked_info['ch_names'][pp] for pp in picks_bad]
    ch_names_b = [inst.info['ch_names'][pp] for pp in picks_bad_orig]
    assert ch_names_a == ch_names_b
    if not inst_picked:
        picks_good_ = [pp for pp in picks if pp in picks_good_]
    assert len(picks_good_) == len(picks_good)
    ch_names_a = [picked_info['ch_names'][pp] for pp in picks_good]
    ch_names_b = [inst.info['ch_names'][pp] for pp in picks_good_]
    assert ch_names_a == ch_names_b
    _do_interp_dots(inst, mapping, picks_good_, picks_bad_orig)


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
        """Compute all-to-all dots."""
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

    # This function needs a clean input. It hates the presence of other
    # channels than MEG channels. Make sure all is picked.
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
