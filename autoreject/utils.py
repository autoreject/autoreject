"""Utility functions for autoreject."""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Denis A. Engemann <denis.engemann@gmail.com>

from collections import defaultdict
import warnings

import numpy as np

import mne
from mne import pick_types, pick_info
from mne.io.pick import _picks_to_idx
from mne.channels.interpolation import _do_interp_dots
from mne.forward._field_interpolation import _setup_dots


def _check_ch_locs(chs):
    """Check if channel locations exist.

    Parameters
    ----------
    chs : dict
        The channels from info['chs']
    """
    locs3d = np.array([ch['loc'][:3] for ch in chs])
    return not ((locs3d == 0).all() or
                (~np.isfinite(locs3d)).any() or
                np.allclose(locs3d, 0.))


def _check_data(epochs, picks, ch_constraint='data_channels', check_loc=True,
                verbose=True):
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

    if check_loc and not _check_ch_locs(picked_info['chs']):
        raise RuntimeError('Valid channel positions are needed '
                           'for autoreject to work')

    # XXX : ch_constraint -> allow_many_types=True | False
    supported_channels = ['mag', 'grad', 'eeg', 'hbo', 'hbr', 'ecog', 'seeg']
    channel_text = ', '.join(x for x in supported_channels)
    message = f'AutoReject only supports {channel_text} currenly.'
    if ch_constraint == 'data_channels':
        if not all(ch in supported_channels for ch in ch_types_picked):
            raise ValueError(message)
    elif ch_constraint == 'single_channel_type':
        if sum(ch in ch_types_picked for ch in supported_channels) > 1:
            raise ValueError(message)  # XXX: to check
    else:
        raise ValueError('bad value for ch_constraint.')

    if n_bads > 0:
        if verbose is not False:
            warnings.warn(
                '%i channels are marked as bad. These will be ignored. '
                'If you want them to be considered by autoreject please '
                'remove them from epochs.info["bads"].' % n_bads)


def _handle_picks(info, picks):
    """Pick the data channls or return picks."""
    if picks is None:
        out = mne.pick_types(info, meg=True, eeg=True, ref_meg=False,
                             fnirs=True, ecog=True, seeg=True, exclude='bads')
    else:
        out = _picks_to_idx(info, picks, exclude='bads')
    return out


def _get_picks_by_type(info, picks):
    """Get the picks grouped by channel type."""
    # do magic here
    sub_picks_ = defaultdict(list)
    keys = list()
    for pp in picks:
        key = mne.io.pick.channel_type(info=info, idx=pp)
        sub_picks_[key].append(pp)
        if key not in keys:
            keys.append(key)
    picks_by_type = [(kk, sub_picks_[kk]) for kk in keys]
    return picks_by_type


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
              'legend.fontsize': fontsize,
              'xtick.labelsize': fontsize,
              'ytick.labelsize': fontsize,
              'axes.titlesize': fontsize + 2}
    plt.rcParams.update(params)


def _pprint(params, offset=0, printer=repr):
    """Pretty print the dictionary 'params' (copied from sklearn)

    Parameters
    ----------
    params : dict
        The dictionary to pretty print
    offset : int
        The offset in characters to add at the begin of each line.
    printer : callable
        The function to convert entries to strings, typically
        the builtin str or repr

    Returns
    -------
    lines : str
        The pretty print of the dictionary as a string.
    """
    # Do a multi-line justified repr:
    options = np.get_printoptions()
    np.set_printoptions(precision=5, threshold=64, edgeitems=2)
    params_list = list()
    this_line_length = offset
    line_sep = ',\n' + (1 + offset // 2) * ' '
    for i, (k, v) in enumerate(sorted(params.items())):
        if type(v) is float:
            # use str for representing floating point numbers
            # this way we get consistent representation across
            # architectures and versions.
            this_repr = '%s=%s' % (k, str(v))
        else:
            # use repr of the rest
            this_repr = '%s=%s' % (k, printer(v))
        if len(this_repr) > 500:
            this_repr = this_repr[:300] + '...' + this_repr[-100:]
        if i > 0:
            if (this_line_length + len(this_repr) >= 75 or '\n' in this_repr):
                params_list.append(line_sep)
                this_line_length = len(line_sep)
            else:
                params_list.append(', ')
                this_line_length += 2
        params_list.append(this_repr)
        this_line_length += len(this_repr)

    np.set_printoptions(**options)
    lines = ''.join(params_list)
    # Strip trailing space to avoid nightmare in doctests
    lines = '\n'.join(li.rstrip(' ') for li in lines.split('\n'))
    return lines


def _pbar(iterable, desc, verbose=True, **kwargs):
    raise_error = False
    if isinstance(verbose, str):
        if verbose not in {"tqdm", "tqdm_notebook", "progressbar"}:
            raise_error = True
        verbose = bool(verbose)
    elif isinstance(verbose, (int, bool)):
        verbose = bool(verbose)  # this can happen with pickling
    else:
        raise_error = True
        warnings.warn(
            (f"verbose flag only supports boolean inputs. Option {verbose} "
             f"coerced into type {bool(verbose)}"), DeprecationWarning)
        verbose = bool(verbose)
    if raise_error:
        raise ValueError(
            f"verbose must be a boolean value. Got {repr(verbose)}")
    if verbose:
        from mne.utils.progressbar import ProgressBar
        pbar = ProgressBar(iterable, mesg=desc, **kwargs)
    else:
        pbar = iterable
    return pbar


def _get_epochs_type():
    if hasattr(mne.epochs, '_BaseEpochs'):
        BaseEpochs = mne.epochs._BaseEpochs
    else:
        BaseEpochs = mne.epochs.BaseEpochs
    return BaseEpochs


def clean_by_interp(inst, picks=None, verbose=True):
    """Clean epochs/evoked by LOOCV.

    Parameters
    ----------
    inst : mne.Evoked | mne.Epochs
        The evoked or epochs object.
    picks : str | list | slice | None
        Channels to include. Slices and lists of integers will be interpreted
        as channel indices. In lists, channel *type* strings (e.g.,
        ``['meg', 'eeg']``) will pick channels of those types, channel *name*
        strings (e.g., ``['MEG0111', 'MEG2623']`` will pick the given channels.
        Can also be the string values ``'all'`` to pick all channels, or
        ``'data'`` to pick data channels. None (default) will pick data
        channels {'meg', 'eeg'}. Note that channels in ``info['bads']`` *will
        be included* if their names or indices are explicitly provided.
    verbose : bool
        The verbosity of progress messages.
        If False, suppress all output messages.

    Returns
    -------
    inst_clean : instance of mne.Evoked or mne.Epochs
        Instance after interpolation of bad channels.
    """
    return _clean_by_interp(inst, picks=picks, verbose=verbose)


def _clean_by_interp(inst, picks=None, dots=None, verbose=True):
    inst_interp = inst.copy()
    mesg = 'Creating augmented epochs'
    picks = _handle_picks(info=inst_interp.info, picks=picks)

    BaseEpochs = _get_epochs_type()
    ch_names = [inst.info['ch_names'][p] for p in picks]
    for ch_idx, (pick, ch) in enumerate(_pbar(list(zip(picks, ch_names)),
                                        desc=mesg, verbose=verbose)):
        inst.info['bads'] = [ch]
        pick_interp = mne.pick_channels(inst.info['ch_names'], [ch])[0]
        data_orig = inst._data[:, pick_interp].copy()

        interpolate_bads(inst, picks=picks, dots=dots,
                         reset_bads=True, mode='fast')

        if isinstance(inst, mne.Evoked):
            inst_interp.data[pick] = inst.data[pick_interp]
        elif isinstance(inst, BaseEpochs):
            inst_interp._data[:, pick] = inst._data[:, pick_interp]
        else:
            raise ValueError('Unrecognized type for inst')
        inst._data[:, pick_interp] = data_orig.copy()
    inst.info['bads'] = inst_interp.info['bads'].copy()
    return inst_interp


def interpolate_bads(inst, picks, dots=None, reset_bads=True, mode='accurate'):
    """Interpolate bad MEG and EEG channels."""
    import mne
    eeg_picks = set(pick_types(inst.info, meg=False, eeg=True, exclude=[]))
    eeg_picks_interp = [p for p in picks if p in eeg_picks]
    meg_picks = set(pick_types(inst.info, meg=True, eeg=False, exclude=[]))
    meg_picks_interp = [p for p in picks if p in meg_picks]
    # to prevent cobyla printf error
    with mne.utils.use_log_level('error'):
        if len(eeg_picks_interp) > 0:
            _interpolate_bads_eeg(inst, picks=eeg_picks_interp)
        if len(meg_picks_interp) > 0:
            _interpolate_bads_meg_fast(inst, picks=meg_picks_interp,
                                       dots=dots, mode=mode)
    if reset_bads is True:
        inst.info['bads'] = []
    return inst


def _interpolate_bads_eeg(inst, picks=None):
    """ Interpolate bad EEG channels.

    Operates in place.

    Parameters
    ----------
    inst : mne.io.Raw, mne.Epochs or mne.Evoked
        The data to interpolate. Must be preloaded.
    picks : str | list | slice | None
        Channels to include for interpolation. Slices and lists of integers
        will be interpreted as channel indices. In lists, channel *name*
        strings (e.g., ``['EEG 01', 'EEG 02']``) will pick the given channels.
        None (default) will pick all EEG channels. Note that channels in
        ``info['bads']`` *will be included* if their names or indices are
        explicitly provided.
    """
    from mne.bem import _fit_sphere
    from mne.utils import logger, warn
    from mne.channels.interpolation import _do_interp_dots
    from mne.channels.interpolation import _make_interpolation_matrix
    import numpy as np

    inst.info._check_consistency()
    if picks is None:
        picks = pick_types(inst.info, meg=False, eeg=True, exclude=[])
    else:
        picks = _handle_picks(inst.info, picks)

    bads_idx = np.zeros(len(inst.ch_names), dtype=bool)
    goods_idx = np.zeros(len(inst.ch_names), dtype=bool)
    bads_idx[picks] = [inst.ch_names[ch] in inst.info['bads'] for ch in picks]

    if len(picks) == 0 or bads_idx.sum() == 0:
        return

    goods_idx[picks] = True
    goods_idx[bads_idx] = False

    pos = inst._get_channel_positions(picks)

    # Make sure only good EEG are used
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


def _interpolate_bads_meg_fast(inst, picks, mode='accurate',
                               dots=None):
    """Interpolate bad channels from data in good channels."""
    # We can have pre-picked instances or not.
    # And we need to handle it.

    inst_picked = True
    if len(inst.ch_names) > len(picks):
        picked_info = pick_info(inst.info, picks)
        dots = _pick_dots(dots, picks, picks)
        inst_picked = False
    else:
        picked_info = inst.info.copy()

    def get_picks_bad_good(info, picks_meg):
        picks_good = [p for p in picks_meg
                      if info['ch_names'][p] not in info['bads']]

        # select the bad meg channel to be interpolated
        if len(info['bads']) == 0:
            picks_bad = []
        else:
            picks_bad = [p for p in picks_meg
                         if info['ch_names'][p] in info['bads']]
        return picks_meg, picks_good, picks_bad

    picks_meg, picks_good, picks_bad = get_picks_bad_good(
        picked_info, range(picked_info['nchan']))
    # return without doing anything if there are no meg channels
    if len(picks_meg) == 0 or len(picks_bad) == 0:
        return

    # we need to make sure that only meg channels are passed here
    # as the MNE interpolation code is not fogriving.
    # This is why we picked the info.
    mapping = _fast_map_meg_channels(
        picked_info, pick_from=picks_good, pick_to=picks_bad,
        dots=dots, mode=mode)
    # the downside is that the mapping matrix now does not match
    # the unpicked info of the data.
    # Since we may have picked the info, we need to double map
    # the indices.
    _, picks_good_, picks_bad_orig = get_picks_bad_good(
        inst.info, picks)
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


def _compute_dots(info, mode='fast', *, templates):
    """Compute all-to-all dots."""
    from mne.forward._lead_dots import _do_self_dots, _do_cross_dots
    from mne.forward._make_forward import _create_meg_coils, _read_coil_defs
    from mne.bem import _check_origin

    if templates is None:
        templates = _read_coil_defs(verbose='error')
    coils = _create_meg_coils(info['chs'], 'normal', info['dev_head_t'],
                              templates)
    my_origin = _check_origin((0., 0., 0.04), info)
    int_rad, noise, lut_fun, n_fact = _setup_dots(mode, info, coils, 'meg')
    self_dots = _do_self_dots(int_rad, False, coils, my_origin, 'meg',
                              lut_fun, n_fact, n_jobs=1)
    cross_dots = _do_cross_dots(int_rad, False, coils, coils,
                                my_origin, 'meg', lut_fun, n_fact).T
    return self_dots, cross_dots


def _pick_dots(dots, pick_from, pick_to):
    if dots is None:
        return dots
    self_dots, cross_dots = dots
    self_dots = self_dots[pick_from, :][:, pick_from]
    cross_dots = cross_dots[pick_to, :][:, pick_from]
    return [self_dots, cross_dots]


def _fast_map_meg_channels(info, pick_from, pick_to,
                           dots=None, mode='fast'):
    from mne.io.pick import pick_info
    from mne.forward._field_interpolation import _compute_mapping_matrix
    from mne.forward._make_forward import _create_meg_coils, _read_coil_defs
    from mne.bem import _check_origin

    miss = 1e-4  # Smoothing criterion for MEG

    info_from = pick_info(info, pick_from, copy=True)
    templates = _read_coil_defs(verbose='error')
    coils_from = _create_meg_coils(info_from['chs'], 'normal',
                                   info_from['dev_head_t'], templates)
    my_origin = _check_origin((0., 0., 0.04), info_from)
    int_rad, noise, lut_fun, n_fact = _setup_dots(mode, info_from,
                                                  coils_from, 'meg')

    # This function needs a clean input. It hates the presence of other
    # channels than MEG channels. Make sure all is picked.
    if dots is None:
        dots = self_dots, cross_dots = _compute_dots(
            info, mode=mode, templates=templates)
    else:
        self_dots, cross_dots = dots

    self_dots, cross_dots = _pick_dots(dots, pick_from, pick_to)

    ch_names = [c['ch_name'] for c in info_from['chs']]
    fmd = dict(kind='meg', ch_names=ch_names,
               origin=my_origin, noise=noise, self_dots=self_dots,
               surface_dots=cross_dots, int_rad=int_rad, miss=miss)

    # XXX: hack to silence _compute_mapping_matrix
    with mne.utils.use_log_level('error'):
        fmd['data'] = _compute_mapping_matrix(fmd, info_from)

    return fmd['data']


def _get_channel_type(epochs, picks):
    """return whether a set of picks are all meg or all eeg channels."""
    picked_info = mne.io.pick.pick_info(epochs.info, picks)
    ch_types_picked = {
        mne.io.meas_info.channel_type(picked_info, idx)
        for idx in range(len(picks))}
    invalid_ch_types_present = [key for key in ch_types_picked
                                if key not in ['mag', 'grad', 'eeg'] and
                                key in epochs]
    if len(invalid_ch_types_present) > 0:
        raise ValueError('Invalid channel types present in epochs.'
                         ' Expected ONLY `meg` or ONLY `eeg`. Got %s'
                         % ', '.join(invalid_ch_types_present))

    has_meg = any(kk in ch_types_picked for kk in ('mag', 'grad'))
    if 'eeg' in ch_types_picked and has_meg:
        raise ValueError('Got mixed channel types. Pick either eeg or meg'
                         ' but not both')
    if 'eeg' in ch_types_picked:
        return 'eeg'
    elif has_meg:
        return 'meg'
    else:
        raise ValueError('Oh no! Your channel type is not known.')
