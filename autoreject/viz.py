"""Functions to plot epochs data
"""

# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#          Jaakko Leppakangas <jaeilepp@student.jyu.fi>
#
# License: Simplified BSD

from functools import partial
import copy

import numpy as np

from mne.utils import get_config, set_config, logger
from mne.io.pick import pick_types, channel_type
from mne.io.proj import setup_proj
from mne.viz.utils import (figure_nobar, _toggle_proj, _toggle_options,
                           _layout_figure, _channels_changed,
                           _plot_raw_onscroll, _onclick_help, plt_show,
                           _compute_scalings)
from mne.defaults import _handle_default


def _draw_epochs_axes(epoch_idx, good_ch_idx, bad_ch_idx, data, times, axes,
                      title_str, axes_handler):
    """Aux functioin"""
    this = axes_handler[0]
    for ii, data_, ax in zip(epoch_idx, data, axes):
        for l, d in zip(ax.lines, data_[good_ch_idx]):
            l.set_data(times, d)
        if bad_ch_idx is not None:
            bad_lines = [ax.lines[k] for k in bad_ch_idx]
            for l, d in zip(bad_lines, data_[bad_ch_idx]):
                l.set_data(times, d)
        if title_str is not None:
            # XXX : don't hack mpl default sizes
            ax.set_title(title_str % ii, fontsize=12)
        ax.set_ylim(data.min(), data.max())
        ax.set_yticks(list())
        ax.set_xticks(list())
        if vars(ax)[this]['reject'] is True:
            #  memorizing reject
            for l in ax.lines:
                l.set_color((0.8, 0.8, 0.8))
            ax.get_figure().canvas.draw()
        else:
            #  forgetting previous reject
            for k in axes_handler:
                if k == this:
                    continue
                if vars(ax).get(k, {}).get('reject', None) is True:
                    for l in ax.lines[:len(good_ch_idx)]:
                        l.set_color('k')
                    if bad_ch_idx is not None:
                        for l in ax.lines[-len(bad_ch_idx):]:
                            l.set_color('r')
                    ax.get_figure().canvas.draw()
                    break


def _epochs_navigation_onclick(event, params):
    """Aux function"""
    import matplotlib.pyplot as plt
    p = params
    here = None
    if event.inaxes == p['back'].ax:
        here = 1
    elif event.inaxes == p['next'].ax:
        here = -1
    elif event.inaxes == p['reject-quit'].ax:
        if p['reject_idx']:
            p['epochs'].drop(p['reject_idx'])
        plt.close(p['fig'])
        plt.close(event.inaxes.get_figure())

    if here is not None:
        p['idx_handler'].rotate(here)
        p['axes_handler'].rotate(here)
        this_idx = p['idx_handler'][0]
        _draw_epochs_axes(this_idx, p['good_ch_idx'], p['bad_ch_idx'],
                          p['data'][this_idx],
                          p['times'], p['axes'], p['title_str'],
                          p['axes_handler'])
        # XXX don't ask me why
        p['axes'][0].get_figure().canvas.draw()


def _epochs_axes_onclick(event, params):
    """Aux function"""
    reject_color = (0.8, 0.8, 0.8)
    ax = event.inaxes
    if event.inaxes is None:
        return
    p = params
    here = vars(ax)[p['axes_handler'][0]]
    if here.get('reject', None) is False:
        idx = here['idx']
        if idx not in p['reject_idx']:
            p['reject_idx'].append(idx)
            for l in ax.lines:
                l.set_color(reject_color)
            here['reject'] = True
    elif here.get('reject', None) is True:
        idx = here['idx']
        if idx in p['reject_idx']:
            p['reject_idx'].pop(p['reject_idx'].index(idx))
            good_lines = [ax.lines[k] for k in p['good_ch_idx']]
            for l in good_lines:
                l.set_color('k')
            if p['bad_ch_idx'] is not None:
                bad_lines = ax.lines[-len(p['bad_ch_idx']):]
                for l in bad_lines:
                    l.set_color('r')
            here['reject'] = False
    ax.get_figure().canvas.draw()


def plot_epochs(epochs, picks=None, scalings=None, n_epochs=20,
                n_channels=20, title=None, show=True, block=False,
                bad_epochs_idx=None, log_labels=None):
    """ Visualize epochs

    Bad epochs can be marked with a left click on top of the epoch. Bad
    channels can be selected by clicking the channel name on the left side of
    the main axes. Calling this function drops all the selected bad epochs as
    well as bad epochs marked beforehand with rejection parameters.

    Parameters
    ----------

    epochs : instance of Epochs
        The epochs object
    picks : array-like of int | None
        Channels to be included. If None only good data channels are used.
        Defaults to None
    scalings : dict | 'auto' | None
        Scaling factors for the traces. If any fields in scalings are 'auto',
        the scaling factor is set to match the 99.5th percentile of a subset of
        the corresponding data. If scalings == 'auto', all scalings fields are
        set to 'auto'. If any fields are 'auto' and data is not preloaded,
        a subset of epochs up to 100mb will be loaded. If None, defaults to::

            dict(mag=1e-12, grad=4e-11, eeg=20e-6, eog=150e-6, ecg=5e-4,
                 emg=1e-3, ref_meg=1e-12, misc=1e-3, stim=1, resp=1, chpi=1e-4)

    n_epochs : int
        The number of epochs per view. Defaults to 20.
    n_channels : int
        The number of channels per view. Defaults to 20.
    title : str | None
        The title of the window. If None, epochs name will be displayed.
        Defaults to None.
    show : bool
        Show figure if True. Defaults to True
    block : bool
        Whether to halt program execution until the figure is closed.
        Useful for rejecting bad trials on the fly by clicking on an epoch.
        Defaults to False.
    bad_epochs_idx : array-like | None
        Indices of bad epochs to show. No bad epochs to visualize if None.
    log_labels : array, shape (n_channels, n_epochs) | None
        The bad segments to show in red and the interpolated segments
        to show in gray.

    Returns
    -------
    fig : Instance of matplotlib.figure.Figure
        The figure.

    Notes
    -----
    The arrow keys (up/down/left/right) can be used to navigate between
    channels and epochs and the scaling can be adjusted with - and + (or =)
    keys, but this depends on the backend matplotlib is configured to use
    (e.g., mpl.use(``TkAgg``) should work). Full screen mode can be toggled
    with f11 key. The amount of epochs and channels per view can be adjusted
    with home/end and page down/page up keys. Butterfly plot can be toggled
    with ``b`` key. Right mouse click adds a vertical line to the plot.
    """
    epochs.drop_bad()
    scalings = _compute_scalings(scalings, epochs)
    scalings = _handle_default('scalings_plot_raw', scalings)

    projs = epochs.info['projs']

    bads = np.array(list(), dtype=int)
    if bad_epochs_idx is not None:
        bads = np.array(bad_epochs_idx).astype(int)

    params = {'epochs': epochs,
              'info': copy.deepcopy(epochs.info),
              'bad_color': (0.8, 0.8, 0.8),
              't_start': 0,
              'histogram': None,
              'bads': bads,
              'log_labels': log_labels}
    params['label_click_fun'] = partial(_pick_bad_channels, params=params)
    _prepare_mne_browse_epochs(params, projs, n_channels, n_epochs, scalings,
                               title, picks)
    _prepare_projectors(params)
    _layout_figure(params)

    callback_close = partial(_close_event, params=params)
    params['fig'].canvas.mpl_connect('close_event', callback_close)
    try:
        plt_show(show, block=block)
    except TypeError:  # not all versions have this
        plt_show(show)

    return params['fig']


def _prepare_mne_browse_epochs(params, projs, n_channels, n_epochs, scalings,
                               title, picks, order=None):
    """Helper for setting up the mne_browse_epochs window."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.collections import LineCollection
    from matplotlib.colors import colorConverter
    epochs = params['epochs']

    if picks is None:
        picks = _handle_picks_viz(epochs)
    if len(picks) < 1:
        raise RuntimeError('No appropriate channels found. Please'
                           ' check your picks')
    picks = sorted(picks)
    # Reorganize channels
    inds = list()
    types = list()
    for t in ['grad', 'mag']:
        idxs = pick_types(params['info'], meg=t, ref_meg=False, exclude=[])
        if len(idxs) < 1:
            continue
        mask = np.in1d(idxs, picks, assume_unique=True)
        inds.append(idxs[mask])
        types += [t] * len(inds[-1])
    pick_kwargs = dict(meg=False, ref_meg=False, exclude=[])
    if order is None:
        order = ['eeg', 'seeg', 'ecog', 'eog', 'ecg', 'emg', 'ref_meg', 'stim',
                 'resp', 'misc', 'chpi', 'syst', 'ias', 'exci']
    for ch_type in order:
        pick_kwargs[ch_type] = True
        idxs = pick_types(params['info'], **pick_kwargs)
        if len(idxs) < 1:
            continue
        mask = np.in1d(idxs, picks, assume_unique=True)
        inds.append(idxs[mask])
        types += [ch_type] * len(inds[-1])
        pick_kwargs[ch_type] = False
    inds = np.concatenate(inds).astype(int)
    if not len(inds) == len(picks):
        raise RuntimeError('Some channels not classified. Please'
                           ' check your picks')
    ch_names = [params['info']['ch_names'][x] for x in inds]
    if params['log_labels'] is not None:
        params['log_labels'] = params['log_labels'][:, inds]

    # set up plotting
    size = get_config('MNE_BROWSE_RAW_SIZE')
    n_epochs = min(n_epochs, len(epochs.events))
    duration = len(epochs.times) * n_epochs
    n_channels = min(n_channels, len(picks))
    if size is not None:
        size = size.split(',')
        size = tuple(float(s) for s in size)
    if title is None:
        try:
            title = epochs.name
        except AttributeError:
            title = epochs._name
        if title is None or len(title) == 0:
            title = ''
    fig = figure_nobar(facecolor='w', figsize=size, dpi=80)
    fig.canvas.set_window_title('mne_browse_epochs')
    ax = plt.subplot2grid((10, 15), (0, 1), colspan=13, rowspan=9)

    ax.annotate(title, xy=(0.5, 1), xytext=(0, ax.get_ylim()[1] + 15),
                ha='center', va='bottom', size=12, xycoords='axes fraction',
                textcoords='offset points')
    color = _handle_default('color', None)

    ax.axis([0, duration, 0, 200])
    ax2 = ax.twiny()
    ax2.set_zorder(-1)
    ax2.axis([0, duration, 0, 200])
    ax_hscroll = plt.subplot2grid((10, 15), (9, 1), colspan=13)
    ax_hscroll.get_yaxis().set_visible(False)
    ax_hscroll.set_xlabel('Epochs')
    ax_vscroll = plt.subplot2grid((10, 15), (0, 14), rowspan=9)
    ax_vscroll.set_axis_off()
    ax_vscroll.add_patch(mpl.patches.Rectangle((0, 0), 1, len(picks),
                                               facecolor='w', zorder=3))

    ax_help_button = plt.subplot2grid((10, 15), (9, 0), colspan=1)
    help_button = mpl.widgets.Button(ax_help_button, 'Help')
    help_button.on_clicked(partial(_onclick_help, params=params))

    # populate vertical and horizontal scrollbars
    for ci in range(len(picks)):
        if ch_names[ci] in params['info']['bads']:
            this_color = params['bad_color']
        else:
            this_color = color[types[ci]]
        ax_vscroll.add_patch(mpl.patches.Rectangle((0, ci), 1, 1,
                                                   facecolor=this_color,
                                                   edgecolor=this_color,
                                                   zorder=4))

    vsel_patch = mpl.patches.Rectangle((0, 0), 1, n_channels, alpha=0.5,
                                       edgecolor='w', facecolor='w', zorder=5)
    ax_vscroll.add_patch(vsel_patch)

    ax_vscroll.set_ylim(len(types), 0)
    ax_vscroll.set_title('Ch.')

    # populate colors list
    type_colors = [colorConverter.to_rgba(color[c]) for c in types]
    colors = list()
    for color_idx in range(len(type_colors)):
        colors.append([type_colors[color_idx]] * len(epochs.events))
    lines = list()
    n_times = len(epochs.times)

    for ch_idx in range(n_channels):
        if len(colors) - 1 < ch_idx:
            break
        lc = LineCollection(list(), antialiased=False, linewidths=0.5,
                            zorder=3, picker=3.)
        ax.add_collection(lc)
        lines.append(lc)

    times = epochs.times
    data = np.zeros((params['info']['nchan'], len(times) * n_epochs))

    ylim = (25., 0.)  # Hardcoded 25 because butterfly has max 5 rows (5*5=25).
    # make shells for plotting traces
    offset = ylim[0] / n_channels
    offsets = np.arange(n_channels) * offset + (offset / 2.)

    times = np.arange(len(times) * len(epochs.events))
    epoch_times = np.arange(0, len(times), n_times)

    ax.set_yticks(offsets)
    ax.set_ylim(ylim)
    ticks = epoch_times + 0.5 * n_times
    ax.set_xticks(ticks)
    ax2.set_xticks(ticks[:n_epochs])
    labels = list(range(1, len(ticks) + 1))  # epoch numbers
    ax.set_xticklabels(labels)
    ax2.set_xticklabels(labels)
    xlim = epoch_times[-1] + len(epochs.times)
    ax_hscroll.set_xlim(0, xlim)
    vertline_t = ax_hscroll.text(0, 1, '', color='y', va='bottom', ha='right')

    # fit horizontal scroll bar ticks
    hscroll_ticks = np.arange(0, xlim, xlim / 7.0)
    hscroll_ticks = np.append(hscroll_ticks, epoch_times[-1])
    hticks = list()
    for tick in hscroll_ticks:
        hticks.append(epoch_times.flat[np.abs(epoch_times - tick).argmin()])
    hlabels = [x / n_times + 1 for x in hticks]
    ax_hscroll.set_xticks(hticks)
    ax_hscroll.set_xticklabels(hlabels)

    for epoch_idx in range(len(epoch_times)):
        ax_hscroll.add_patch(mpl.patches.Rectangle((epoch_idx * n_times, 0),
                                                   n_times, 1,
                                                   facecolor=(0.8, 0.8, 0.8),
                                                   edgecolor=(0.8, 0.8, 0.8),
                                                   alpha=0.5))
    hsel_patch = mpl.patches.Rectangle((0, 0), duration, 1,
                                       edgecolor='k',
                                       facecolor=(0.5, 0.5, 0.5),
                                       alpha=0.25, linewidth=1, clip_on=False)
    ax_hscroll.add_patch(hsel_patch)
    text = ax.text(0, 0, 'blank', zorder=3, verticalalignment='baseline',
                   ha='left', fontweight='bold')
    text.set_visible(False)

    params.update({'fig': fig,
                   'ax': ax,
                   'ax2': ax2,
                   'ax_hscroll': ax_hscroll,
                   'ax_vscroll': ax_vscroll,
                   'vsel_patch': vsel_patch,
                   'hsel_patch': hsel_patch,
                   'lines': lines,
                   'projs': projs,
                   'ch_names': ch_names,
                   'n_channels': n_channels,
                   'n_epochs': n_epochs,
                   'scalings': scalings,
                   'duration': duration,
                   'ch_start': 0,
                   'colors': colors,
                   'def_colors': type_colors,  # don't change at runtime
                   'picks': picks,
                   'data': data,
                   'times': times,
                   'epoch_times': epoch_times,
                   'offsets': offsets,
                   'labels': labels,
                   'scale_factor': 1.0,
                   'butterfly_scale': 1.0,
                   'fig_proj': None,
                   'types': np.array(types),
                   'inds': inds,
                   'vert_lines': list(),
                   'vertline_t': vertline_t,
                   'butterfly': False,
                   'text': text,
                   'ax_help_button': ax_help_button,  # needed for positioning
                   'help_button': help_button,  # reference needed for clicks
                   'fig_options': None,
                   'settings': [True, True, True, True],
                   'image_plot': None})

    params['plot_fun'] = partial(_plot_traces, params=params)

    # callbacks
    callback_scroll = partial(_plot_onscroll, params=params)
    fig.canvas.mpl_connect('scroll_event', callback_scroll)
    callback_click = partial(_mouse_click, params=params)
    fig.canvas.mpl_connect('button_press_event', callback_click)
    callback_key = partial(_plot_onkey, params=params)
    fig.canvas.mpl_connect('key_press_event', callback_key)
    callback_resize = partial(_resize_event, params=params)
    fig.canvas.mpl_connect('resize_event', callback_resize)
    fig.canvas.mpl_connect('pick_event', partial(_onpick, params=params))
    params['callback_key'] = callback_key

    # Draw event lines for the first time.
    _plot_vert_lines(params)

    # Plot bad epochs
    if len(params['bads']) > 0:
        if params['bads'].max() > len(epochs.events):
            raise ValueError('You had a bad_epoch with index'
                             '%d but there are only %d epochs. Make sure'
                             ' to provide the epochs *before* running'
                             'autoreject.'
                             % (params['bads'].max(), len(epochs.events)))
    for epoch_idx in params['bads']:
        params['ax_hscroll'].patches[epoch_idx].set_color((1., 0., 0., 1.))
        params['ax_hscroll'].patches[epoch_idx].set_zorder(3)
        params['ax_hscroll'].patches[epoch_idx].set_edgecolor('w')
        for ch_idx in range(len(params['ch_names'])):
            params['colors'][ch_idx][epoch_idx] = (1., 0., 0., 1.)

    # Plot bad segments
    if params['log_labels'] is not None:
        if not params['log_labels'].shape[0] == len(epochs.events):
            raise ValueError('The number of epochs should match the number of'
                             'epochs *before* autoreject. Please provide'
                             'the epochs object before running autoreject')
        if not params['log_labels'].shape[1] == len(params['ch_names']):
            raise ValueError('The number of channels should match the number'
                             ' of channels before running autoreject.')

        for ch_idx in range(len(params['ch_names'])):
            for epoch_idx in range(len(epochs.events)):
                this_log = params['log_labels'][epoch_idx, ch_idx]
                if epoch_idx in params['bads']:
                    pass
                else:
                    if this_log == 1:
                        params['colors'][ch_idx][epoch_idx] = (1., 0., 0., 1.)
                    elif this_log == 2:
                        params['colors'][ch_idx][epoch_idx] = (0.6, 0.6, 0.6,
                                                               1.)

    params['plot_fun']()


def _prepare_projectors(params):
    """ Helper for setting up the projectors for epochs browser """
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    epochs = params['epochs']
    projs = params['projs']
    if len(projs) > 0 and not epochs.proj:
        ax_button = plt.subplot2grid((10, 15), (9, 14))
        opt_button = mpl.widgets.Button(ax_button, 'Proj')
        callback_option = partial(_toggle_options, params=params)
        opt_button.on_clicked(callback_option)
        params['opt_button'] = opt_button
        params['ax_button'] = ax_button

    # As here code is shared with plot_evoked, some extra steps:
    # first the actual plot update function
    params['plot_update_proj_callback'] = _plot_update_epochs_proj
    # then the toggle handler
    callback_proj = partial(_toggle_proj, params=params)
    # store these for use by callbacks in the options figure
    params['callback_proj'] = callback_proj
    callback_proj('none')


def _plot_traces(params):
    """ Helper for plotting concatenated epochs """
    params['text'].set_visible(False)
    ax = params['ax']
    butterfly = params['butterfly']
    if butterfly:
        ch_start = 0
        n_channels = len(params['picks'])
        data = params['data'] * params['butterfly_scale']
    else:
        ch_start = params['ch_start']
        n_channels = params['n_channels']
        data = params['data'] * params['scale_factor']
    offsets = params['offsets']
    lines = params['lines']
    epochs = params['epochs']

    n_times = len(epochs.times)
    tick_list = list()
    start_idx = int(params['t_start'] / n_times)
    end = params['t_start'] + params['duration']
    end_idx = int(end / n_times)
    xlabels = params['labels'][start_idx:]
    event_ids = params['epochs'].events[:, 2]
    params['ax2'].set_xticklabels(event_ids[start_idx:])
    ax.set_xticklabels(xlabels)
    ylabels = ax.yaxis.get_ticklabels()
    # do the plotting
    for line_idx in range(n_channels):
        ch_idx = line_idx + ch_start
        if line_idx >= len(lines):
            break
        elif ch_idx < len(params['ch_names']):
            if butterfly:
                ch_type = params['types'][ch_idx]
                if ch_type == 'grad':
                    offset = offsets[0]
                elif ch_type == 'mag':
                    offset = offsets[1]
                elif ch_type == 'eeg':
                    offset = offsets[2]
                elif ch_type == 'eog':
                    offset = offsets[3]
                elif ch_type == 'ecg':
                    offset = offsets[4]
                else:
                    lines[line_idx].set_segments(list())
            else:
                tick_list += [params['ch_names'][ch_idx]]
                offset = offsets[line_idx]
            this_data = data[ch_idx]

            # subtraction here gets correct orientation for flipped ylim
            ydata = offset - this_data
            xdata = params['times'][:params['duration']]
            num_epochs = np.min([params['n_epochs'],
                                len(epochs.events)])
            segments = np.split(np.array((xdata, ydata)).T, num_epochs)

            ch_name = params['ch_names'][ch_idx]
            if ch_name in params['info']['bads']:
                if not butterfly:
                    this_color = params['bad_color']
                    ylabels[line_idx].set_color(this_color)
                this_color = np.tile((params['bad_color']), (num_epochs, 1))
                for bad_idx in params['bads']:
                    if bad_idx < start_idx or bad_idx > end_idx:
                        continue
                    this_color[bad_idx - start_idx] = (1., 0., 0.)
                lines[line_idx].set_zorder(2)
            else:
                this_color = params['colors'][ch_idx][start_idx:end_idx]
                lines[line_idx].set_zorder(3)
                if not butterfly:
                    ylabels[line_idx].set_color('black')
            lines[line_idx].set_segments(segments)
            lines[line_idx].set_color(this_color)
        else:
            lines[line_idx].set_segments(list())

    # finalize plot
    ax.set_xlim(params['times'][0], params['times'][0] + params['duration'],
                False)
    params['ax2'].set_xlim(params['times'][0],
                           params['times'][0] + params['duration'], False)
    if butterfly:
        factor = -1. / params['butterfly_scale']
        labels = np.empty(20, dtype='S15')
        labels.fill('')
        ticks = ax.get_yticks()
        idx_offset = 1
        if 'grad' in params['types']:
            labels[idx_offset + 1] = '0.00'
            for idx in [idx_offset, idx_offset + 2]:
                labels[idx] = '{0:.2f}'.format((ticks[idx] - offsets[0]) *
                                               params['scalings']['grad'] *
                                               1e13 * factor)
            idx_offset += 4
        if 'mag' in params['types']:
            labels[idx_offset + 1] = '0.00'
            for idx in [idx_offset, idx_offset + 2]:
                labels[idx] = '{0:.2f}'.format((ticks[idx] - offsets[1]) *
                                               params['scalings']['mag'] *
                                               1e15 * factor)
            idx_offset += 4
        if 'eeg' in params['types']:
            labels[idx_offset + 1] = '0.00'
            for idx in [idx_offset, idx_offset + 2]:
                labels[idx] = '{0:.2f}'.format((ticks[idx] - offsets[2]) *
                                               params['scalings']['eeg'] *
                                               1e6 * factor)
            idx_offset += 4
        if 'eog' in params['types']:
            labels[idx_offset + 1] = '0.00'
            for idx in [idx_offset, idx_offset + 2]:
                labels[idx] = '{0:.2f}'.format((ticks[idx] - offsets[3]) *
                                               params['scalings']['eog'] *
                                               1e6 * factor)
            idx_offset += 4
        if 'ecg' in params['types']:
            labels[idx_offset + 1] = '0.00'
            for idx in [idx_offset, idx_offset + 2]:
                labels[idx] = '{0:.2f}'.format((ticks[idx] - offsets[4]) *
                                               params['scalings']['ecg'] *
                                               1e6 * factor)
        ax.set_yticklabels(labels, fontsize=12, color='black')
    else:
        ax.set_yticklabels(tick_list, fontsize=12)
    params['vsel_patch'].set_y(ch_start)
    params['fig'].canvas.draw()
    # XXX This is a hack to make sure this figure gets drawn last
    # so that when matplotlib goes to calculate bounds we don't get a
    # CGContextRef error on the MacOSX backend :(
    if params['fig_proj'] is not None:
        params['fig_proj'].canvas.draw()


def _plot_update_epochs_proj(params, bools=None):
    """Helper only needs to be called when proj is changed"""
    if bools is not None:
        inds = np.where(bools)[0]
        params['info']['projs'] = [copy.deepcopy(params['projs'][ii])
                                   for ii in inds]
        params['proj_bools'] = bools
    params['projector'], _ = setup_proj(params['info'],
                                        verbose=False)

    start = int(params['t_start'] / len(params['epochs'].times))
    n_epochs = params['n_epochs']
    end = start + n_epochs
    data = np.concatenate(params['epochs'][start:end].get_data(), axis=1)
    if params['projector'] is not None:
        data = np.dot(params['projector'], data)
    types = params['types']
    for pick, ind in enumerate(params['inds']):
        params['data'][pick] = data[ind] / params['scalings'][types[pick]]
    params['plot_fun']()


def _handle_picks_viz(epochs):
    """Aux function to handle picks."""
    if any('ICA' in k for k in epochs.ch_names):
        picks = pick_types(epochs.info, misc=True, ref_meg=False,
                           exclude=[])
    else:
        picks = pick_types(epochs.info, meg=True, eeg=True, eog=True, ecg=True,
                           seeg=True, ecog=True, ref_meg=False, exclude=[])
    return picks


def _plot_window(value, params):
    """Deal with horizontal shift of the viewport."""
    max_times = len(params['times']) - params['duration']
    if value > max_times:
        value = len(params['times']) - params['duration']
    if value < 0:
        value = 0
    if params['t_start'] != value:
        params['t_start'] = value
        params['hsel_patch'].set_x(value)
        params['plot_update_proj_callback'](params)


def _plot_vert_lines(params):
    """ Helper function for plotting vertical lines."""
    ax = params['ax']
    while len(ax.lines) > 0:
        ax.lines.pop()
    params['vert_lines'] = list()
    params['vertline_t'].set_text('')

    epochs = params['epochs']
    if params['settings'][3]:  # if zeroline visible
        t_zero = np.where(epochs.times == 0.)[0]
        if len(t_zero) == 1:
            for event_idx in range(len(epochs.events)):
                pos = [event_idx * len(epochs.times) + t_zero[0],
                       event_idx * len(epochs.times) + t_zero[0]]
                ax.plot(pos, ax.get_ylim(), 'g', zorder=4, alpha=0.4)
    for epoch_idx in range(len(epochs.events)):
        pos = [epoch_idx * len(epochs.times), epoch_idx * len(epochs.times)]
        ax.plot(pos, ax.get_ylim(), color='black', linestyle='--', zorder=2)


def _pick_bad_epochs(event, params):
    """Helper for selecting / dropping bad epochs"""
    if 'ica' in params:
        pos = (event.xdata, event.ydata)
        _pick_bad_channels(pos, params)
        return
    n_times = len(params['epochs'].times)
    start_idx = int(params['t_start'] / n_times)
    xdata = event.xdata
    xlim = event.inaxes.get_xlim()
    epoch_idx = start_idx + int(xdata / (xlim[1] / params['n_epochs']))
    total_epochs = len(params['epochs'].events)
    if epoch_idx > total_epochs - 1:
        return
    # remove bad epoch
    if epoch_idx in params['bads']:
        params['bads'] = params['bads'][(params['bads'] != epoch_idx)]
        for ch_idx in range(len(params['ch_names'])):
            params['colors'][ch_idx][epoch_idx] = params['def_colors'][ch_idx]
        params['ax_hscroll'].patches[epoch_idx].set_color('w')
        params['ax_hscroll'].patches[epoch_idx].set_zorder(2)
        params['plot_fun']()
        return
    # add bad epoch
    params['bads'] = np.append(params['bads'], epoch_idx)
    params['ax_hscroll'].patches[epoch_idx].set_color((1., 0., 0., 1.))
    params['ax_hscroll'].patches[epoch_idx].set_zorder(3)
    params['ax_hscroll'].patches[epoch_idx].set_edgecolor('w')
    for ch_idx in range(len(params['ch_names'])):
        params['colors'][ch_idx][epoch_idx] = (1., 0., 0., 1.)
    params['plot_fun']()


def _pick_bad_channels(pos, params):
    """Helper function for selecting bad channels."""
    text, ch_idx = _label2idx(params, pos)
    if text is None:
        return
    if text in params['info']['bads']:
        while text in params['info']['bads']:
            params['info']['bads'].remove(text)
        color = params['def_colors'][ch_idx]
        params['ax_vscroll'].patches[ch_idx + 1].set_color(color)
    else:
        params['info']['bads'].append(text)
        color = params['bad_color']
        params['ax_vscroll'].patches[ch_idx + 1].set_color(color)
    if 'ica' in params:
        params['plot_fun']()
    else:
        params['plot_update_proj_callback'](params)


def _plot_onscroll(event, params):
    """Function to handle scroll events."""
    if event.key == 'control':
        if event.step < 0:
            event.key = '-'
        else:
            event.key = '+'
        _plot_onkey(event, params)
        return
    if params['butterfly']:
        return
    _plot_raw_onscroll(event, params, len(params['ch_names']))


def _mouse_click(event, params):
    """Function to handle mouse click events."""
    if event.inaxes is None:
        if params['butterfly'] or not params['settings'][0]:
            return
        ax = params['ax']
        ylim = ax.get_ylim()
        pos = ax.transData.inverted().transform((event.x, event.y))
        if pos[0] > 0 or pos[1] < 0 or pos[1] > ylim[0]:
            return
        if event.button == 1:  # left click
            params['label_click_fun'](pos)
        elif event.button == 3:  # right click
            if 'ica' not in params:
                _, ch_idx = _label2idx(params, pos)
                if ch_idx is None:
                    return
                if channel_type(params['info'], ch_idx) not in ['mag', 'grad',
                                                                'eeg', 'eog']:
                    logger.info('Event related fields / potentials only '
                                'available for MEG and EEG channels.')
                    return
    elif event.button == 1:  # left click
        # vertical scroll bar changed
        if event.inaxes == params['ax_vscroll']:
            if params['butterfly']:
                return
            # Don't let scrollbar go outside vertical scrollbar limits
            # XXX: floating point exception on some machines if this happens.
            ch_start = min(
                max(int(event.ydata) - params['n_channels'] // 2, 0),
                len(params['ch_names']) - params['n_channels'])

            if params['ch_start'] != ch_start:
                params['ch_start'] = ch_start
                params['plot_fun']()
        # horizontal scroll bar changed
        elif event.inaxes == params['ax_hscroll']:
            # find the closest epoch time
            times = params['epoch_times']
            offset = 0.5 * params['n_epochs'] * len(params['epochs'].times)
            xdata = times.flat[np.abs(times - (event.xdata - offset)).argmin()]
            _plot_window(xdata, params)
        # main axes
        elif event.inaxes == params['ax']:
            _pick_bad_epochs(event, params)

    elif event.inaxes == params['ax'] and event.button == 2:  # middle click
        params['fig'].canvas.draw()
        if params['fig_proj'] is not None:
            params['fig_proj'].canvas.draw()
    elif event.inaxes == params['ax'] and event.button == 3:  # right click
        n_times = len(params['epochs'].times)
        xdata = int(event.xdata % n_times)
        prev_xdata = 0
        if len(params['vert_lines']) > 0:
            prev_xdata = params['vert_lines'][0][0].get_data()[0][0]
            while len(params['vert_lines']) > 0:
                params['ax'].lines.remove(params['vert_lines'][0][0])
                params['vert_lines'].pop(0)
        if prev_xdata == xdata:  # lines removed
            params['vertline_t'].set_text('')
            params['plot_fun']()
            return
        ylim = params['ax'].get_ylim()
        for epoch_idx in range(params['n_epochs']):  # plot lines
            pos = [epoch_idx * n_times + xdata, epoch_idx * n_times + xdata]
            params['vert_lines'].append(params['ax'].plot(pos, ylim, 'y',
                                                          zorder=5))
        params['vertline_t'].set_text('%0.3f' % params['epochs'].times[xdata])
        params['plot_fun']()


def _plot_onkey(event, params):
    """Function to handle key presses."""
    import matplotlib.pyplot as plt
    if event.key == 'down':
        if params['butterfly']:
            return
        params['ch_start'] += params['n_channels']
        _channels_changed(params, len(params['ch_names']))
    elif event.key == 'up':
        if params['butterfly']:
            return
        params['ch_start'] -= params['n_channels']
        _channels_changed(params, len(params['ch_names']))
    elif event.key == 'left':
        sample = params['t_start'] - params['duration']
        sample = np.max([0, sample])
        _plot_window(sample, params)
    elif event.key == 'right':
        sample = params['t_start'] + params['duration']
        sample = np.min([sample, params['times'][-1] - params['duration']])
        times = params['epoch_times']
        xdata = times.flat[np.abs(times - sample).argmin()]
        _plot_window(xdata, params)
    elif event.key == '-':
        if params['butterfly']:
            params['butterfly_scale'] /= 1.1
        else:
            params['scale_factor'] /= 1.1
        params['plot_fun']()
    elif event.key in ['+', '=']:
        if params['butterfly']:
            params['butterfly_scale'] *= 1.1
        else:
            params['scale_factor'] *= 1.1
        params['plot_fun']()
    elif event.key == 'f11':
        mng = plt.get_current_fig_manager()
        mng.full_screen_toggle()
    elif event.key == 'pagedown':
        if params['n_channels'] == 1 or params['butterfly']:
            return
        n_channels = params['n_channels'] - 1
        ylim = params['ax'].get_ylim()
        offset = ylim[0] / n_channels
        params['offsets'] = np.arange(n_channels) * offset + (offset / 2.)
        params['n_channels'] = n_channels
        params['ax'].collections.pop()
        params['ax'].set_yticks(params['offsets'])
        params['lines'].pop()
        params['vsel_patch'].set_height(n_channels)
        params['plot_fun']()
    elif event.key == 'pageup':
        if params['butterfly']:
            return
        from matplotlib.collections import LineCollection
        n_channels = params['n_channels'] + 1
        ylim = params['ax'].get_ylim()
        offset = ylim[0] / n_channels
        params['offsets'] = np.arange(n_channels) * offset + (offset / 2.)
        params['n_channels'] = n_channels
        lc = LineCollection(list(), antialiased=False, linewidths=0.5,
                            zorder=3, picker=3.)
        params['ax'].add_collection(lc)
        params['ax'].set_yticks(params['offsets'])
        params['lines'].append(lc)
        params['vsel_patch'].set_height(n_channels)
        params['plot_fun']()
    elif event.key == 'home':
        n_epochs = params['n_epochs'] - 1
        if n_epochs <= 0:
            return
        n_times = len(params['epochs'].times)
        ticks = params['epoch_times'] + 0.5 * n_times
        params['ax2'].set_xticks(ticks[:n_epochs])
        params['n_epochs'] = n_epochs
        params['duration'] -= n_times
        params['hsel_patch'].set_width(params['duration'])
        params['data'] = params['data'][:, :-n_times]
        params['plot_update_proj_callback'](params)
    elif event.key == 'end':
        n_epochs = params['n_epochs'] + 1
        n_times = len(params['epochs'].times)
        if n_times * n_epochs > len(params['times']):
            return
        ticks = params['epoch_times'] + 0.5 * n_times
        params['ax2'].set_xticks(ticks[:n_epochs])
        params['n_epochs'] = n_epochs
        if len(params['vert_lines']) > 0:
            ax = params['ax']
            pos = params['vert_lines'][0][0].get_data()[0] + params['duration']
            params['vert_lines'].append(ax.plot(pos, ax.get_ylim(), 'y',
                                                zorder=4))
        params['duration'] += n_times
        if params['t_start'] + params['duration'] > len(params['times']):
            params['t_start'] -= n_times
            params['hsel_patch'].set_x(params['t_start'])
        params['hsel_patch'].set_width(params['duration'])
        params['data'] = np.zeros((len(params['data']), params['duration']))
        params['plot_update_proj_callback'](params)
    elif event.key == 'b':
        if params['fig_options'] is not None:
            plt.close(params['fig_options'])
            params['fig_options'] = None
        _prepare_butterfly(params)
        _plot_traces(params)
    elif event.key == 'o':
        if not params['butterfly']:
            _open_options(params)
    elif event.key == 'h':
        _plot_histogram(params)
    elif event.key == '?':
        _onclick_help(event, params)
    elif event.key == 'escape':
        plt.close(params['fig'])


def _prepare_butterfly(params):
    """Helper function for setting up butterfly plot."""
    from matplotlib.collections import LineCollection
    butterfly = not params['butterfly']
    if butterfly:
        types = set(['grad', 'mag', 'eeg', 'eog',
                     'ecg']) & set(params['types'])
        if len(types) < 1:
            return
        params['ax_vscroll'].set_visible(False)
        ax = params['ax']
        labels = ax.yaxis.get_ticklabels()
        for label in labels:
            label.set_visible(True)
        ylim = (5. * len(types), 0.)
        ax.set_ylim(ylim)
        offset = ylim[0] / (4. * len(types))
        ticks = np.arange(0, ylim[0], offset)
        ticks = [ticks[x] if x < len(ticks) else 0 for x in range(20)]
        ax.set_yticks(ticks)
        used_types = 0
        params['offsets'] = [ticks[2]]
        if 'grad' in types:
            pos = (0, 1 - (ticks[2] / ylim[0]))
            params['ax2'].annotate('Grad (fT/cm)', xy=pos, xytext=(-70, 0),
                                   ha='left', size=12, va='center',
                                   xycoords='axes fraction', rotation=90,
                                   textcoords='offset points')
            used_types += 1
        params['offsets'].append(ticks[2 + used_types * 4])
        if 'mag' in types:
            pos = (0, 1 - (ticks[2 + used_types * 4] / ylim[0]))
            params['ax2'].annotate('Mag (fT)', xy=pos, xytext=(-70, 0),
                                   ha='left', size=12, va='center',
                                   xycoords='axes fraction', rotation=90,
                                   textcoords='offset points')
            used_types += 1
        params['offsets'].append(ticks[2 + used_types * 4])
        if 'eeg' in types:
            pos = (0, 1 - (ticks[2 + used_types * 4] / ylim[0]))
            params['ax2'].annotate('EEG (uV)', xy=pos, xytext=(-70, 0),
                                   ha='left', size=12, va='center',
                                   xycoords='axes fraction', rotation=90,
                                   textcoords='offset points')
            used_types += 1
        params['offsets'].append(ticks[2 + used_types * 4])
        if 'eog' in types:
            pos = (0, 1 - (ticks[2 + used_types * 4] / ylim[0]))
            params['ax2'].annotate('EOG (uV)', xy=pos, xytext=(-70, 0),
                                   ha='left', size=12, va='center',
                                   xycoords='axes fraction', rotation=90,
                                   textcoords='offset points')
            used_types += 1
        params['offsets'].append(ticks[2 + used_types * 4])
        if 'ecg' in types:
            pos = (0, 1 - (ticks[2 + used_types * 4] / ylim[0]))
            params['ax2'].annotate('ECG (uV)', xy=pos, xytext=(-70, 0),
                                   ha='left', size=12, va='center',
                                   xycoords='axes fraction', rotation=90,
                                   textcoords='offset points')
            used_types += 1

        while len(params['lines']) < len(params['picks']):
            lc = LineCollection(list(), antialiased=False, linewidths=0.5,
                                zorder=3, picker=3.)
            ax.add_collection(lc)
            params['lines'].append(lc)
    else:  # change back to default view
        labels = params['ax'].yaxis.get_ticklabels()
        for label in labels:
            label.set_visible(params['settings'][0])
        params['ax_vscroll'].set_visible(True)
        while len(params['ax2'].texts) > 0:
            params['ax2'].texts.pop()
        n_channels = params['n_channels']
        while len(params['lines']) > n_channels:
            params['ax'].collections.pop()
            params['lines'].pop()
        ylim = (25., 0.)
        params['ax'].set_ylim(ylim)
        offset = ylim[0] / n_channels
        params['offsets'] = np.arange(n_channels) * offset + (offset / 2.)
        params['ax'].set_yticks(params['offsets'])
    params['butterfly'] = butterfly


def _onpick(event, params):
    """Helper to add a channel name on click"""
    if event.mouseevent.button != 2 or not params['butterfly']:
        return  # text label added with a middle mouse button
    lidx = np.where([l is event.artist for l in params['lines']])[0][0]
    text = params['text']
    text.set_x(event.mouseevent.xdata)
    text.set_y(event.mouseevent.ydata)
    text.set_text(params['ch_names'][lidx])
    text.set_visible(True)
    # do NOT redraw here, since for butterfly plots hundreds of lines could
    # potentially be picked -- use _mouse_click (happens once per click)
    # to do the drawing


def _close_event(event, params):
    pass


def _resize_event(event, params):
    """Function to handle resize event"""
    size = ','.join([str(s) for s in params['fig'].get_size_inches()])
    set_config('MNE_BROWSE_RAW_SIZE', size, set_env=False)
    _layout_figure(params)


def _update_channels_epochs(event, params):
    """Function for changing the amount of channels and epochs per view."""
    from matplotlib.collections import LineCollection
    # Channels
    n_channels = int(np.around(params['channel_slider'].val))
    offset = params['ax'].get_ylim()[0] / n_channels
    params['offsets'] = np.arange(n_channels) * offset + (offset / 2.)
    while len(params['lines']) > n_channels:
        params['ax'].collections.pop()
        params['lines'].pop()
    while len(params['lines']) < n_channels:
        lc = LineCollection(list(), linewidths=0.5, antialiased=False,
                            zorder=3, picker=3.)
        params['ax'].add_collection(lc)
        params['lines'].append(lc)
    params['ax'].set_yticks(params['offsets'])
    params['vsel_patch'].set_height(n_channels)
    params['n_channels'] = n_channels

    # Epochs
    n_epochs = int(np.around(params['epoch_slider'].val))
    n_times = len(params['epochs'].times)
    ticks = params['epoch_times'] + 0.5 * n_times
    params['ax2'].set_xticks(ticks[:n_epochs])
    params['n_epochs'] = n_epochs
    params['duration'] = n_times * n_epochs
    params['hsel_patch'].set_width(params['duration'])
    params['data'] = np.zeros((len(params['data']), params['duration']))
    if params['t_start'] + n_times * n_epochs > len(params['times']):
        params['t_start'] = len(params['times']) - n_times * n_epochs
        params['hsel_patch'].set_x(params['t_start'])
    params['plot_update_proj_callback'](params)


def _toggle_labels(label, params):
    """Function for toggling axis labels on/off."""
    if label == 'Channel names visible':
        params['settings'][0] = not params['settings'][0]
        labels = params['ax'].yaxis.get_ticklabels()
        for label in labels:
            label.set_visible(params['settings'][0])
    elif label == 'Event-id visible':
        params['settings'][1] = not params['settings'][1]
        labels = params['ax2'].xaxis.get_ticklabels()
        for label in labels:
            label.set_visible(params['settings'][1])
    elif label == 'Epoch-id visible':
        params['settings'][2] = not params['settings'][2]
        labels = params['ax'].xaxis.get_ticklabels()
        for label in labels:
            label.set_visible(params['settings'][2])
    elif label == 'Zeroline visible':
        params['settings'][3] = not params['settings'][3]
        _plot_vert_lines(params)
    params['fig'].canvas.draw()
    if params['fig_proj'] is not None:
        params['fig_proj'].canvas.draw()


def _open_options(params):
    """Function for opening the option window."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    if params['fig_options'] is not None:
        # turn off options dialog
        plt.close(params['fig_options'])
        params['fig_options'] = None
        return
    width = 10
    height = 3
    fig_options = figure_nobar(figsize=(width, height), dpi=80)
    fig_options.canvas.set_window_title('View settings')
    params['fig_options'] = fig_options
    ax_channels = plt.axes([0.15, 0.1, 0.65, 0.1])
    ax_epochs = plt.axes([0.15, 0.25, 0.65, 0.1])
    ax_button = plt.axes([0.85, 0.1, 0.1, 0.25])
    ax_check = plt.axes([0.15, 0.4, 0.4, 0.55])
    plt.axis('off')
    params['update_button'] = mpl.widgets.Button(ax_button, 'Update')
    params['channel_slider'] = mpl.widgets.Slider(ax_channels, 'Channels', 1,
                                                  len(params['ch_names']),
                                                  valfmt='%0.0f',
                                                  valinit=params['n_channels'])
    params['epoch_slider'] = mpl.widgets.Slider(ax_epochs, 'Epochs', 1,
                                                len(params['epoch_times']),
                                                valfmt='%0.0f',
                                                valinit=params['n_epochs'])
    params['checkbox'] = mpl.widgets.CheckButtons(ax_check,
                                                  ['Channel names visible',
                                                   'Event-id visible',
                                                   'Epoch-id visible',
                                                   'Zeroline visible'],
                                                  actives=params['settings'])
    update = partial(_update_channels_epochs, params=params)
    params['update_button'].on_clicked(update)
    labels_callback = partial(_toggle_labels, params=params)
    params['checkbox'].on_clicked(labels_callback)
    close_callback = partial(_settings_closed, params=params)
    params['fig_options'].canvas.mpl_connect('close_event', close_callback)
    try:
        params['fig_options'].canvas.draw()
        params['fig_options'].show(warn=False)
        if params['fig_proj'] is not None:
            params['fig_proj'].canvas.draw()
    except Exception:
        pass


def _settings_closed(events, params):
    """Function to handle close event from settings dialog."""
    params['fig_options'] = None


def _plot_histogram(params):
    """Function for plotting histogram of peak-to-peak values."""
    import matplotlib.pyplot as plt
    epochs = params['epochs']
    p2p = np.ptp(epochs.get_data(), axis=2)
    types = list()
    data = list()
    if 'eeg' in params['types']:
        eegs = np.array([p2p.T[i] for i,
                         x in enumerate(params['types']) if x == 'eeg'])
        data.append(eegs.ravel())
        types.append('eeg')
    if 'mag' in params['types']:
        mags = np.array([p2p.T[i] for i,
                         x in enumerate(params['types']) if x == 'mag'])
        data.append(mags.ravel())
        types.append('mag')
    if 'grad' in params['types']:
        grads = np.array([p2p.T[i] for i,
                          x in enumerate(params['types']) if x == 'grad'])
        data.append(grads.ravel())
        types.append('grad')
    params['histogram'] = plt.figure()
    scalings = _handle_default('scalings')
    units = _handle_default('units')
    titles = _handle_default('titles')
    colors = _handle_default('color')
    for idx in range(len(types)):
        ax = plt.subplot(len(types), 1, idx + 1)
        plt.xlabel(units[types[idx]])
        plt.ylabel('count')
        color = colors[types[idx]]
        rej = None
        if epochs.reject is not None and types[idx] in epochs.reject:
            rej = epochs.reject[types[idx]] * scalings[types[idx]]
            rng = [0., rej * 1.1]
        else:
            rng = None
        plt.hist(data[idx] * scalings[types[idx]], bins=100, color=color,
                 range=rng)
        if rej is not None:
            ax.plot((rej, rej), (0, ax.get_ylim()[1]), color='r')
        plt.title(titles[types[idx]])
    params['histogram'].suptitle('Peak-to-peak histogram', y=0.99)
    params['histogram'].subplots_adjust(hspace=0.6)
    try:
        params['histogram'].show(warn=False)
    except Exception:
        pass
    if params['fig_proj'] is not None:
        params['fig_proj'].canvas.draw()


def _label2idx(params, pos):
    """Aux function for click on labels. Returns channel name and idx."""
    labels = params['ax'].yaxis.get_ticklabels()
    offsets = np.array(params['offsets']) + params['offsets'][0]
    line_idx = np.searchsorted(offsets, pos[1])
    text = labels[line_idx].get_text()
    if len(text) == 0:
        return None, None
    ch_idx = params['ch_start'] + line_idx
    return text, ch_idx
