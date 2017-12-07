"""Automated rejection and repair of trials in M/EEG."""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Denis A. Engemann <denis.engemann@gmail.com>

import warnings
import numpy as np
from scipy.stats.distributions import uniform

import mne
from mne import pick_types

from sklearn.base import BaseEstimator
from sklearn.model_selection import RandomizedSearchCV
from sklearn.cross_validation import KFold, StratifiedShuffleSplit
from sklearn.cross_validation import cross_val_score
from sklearn.externals.joblib import Memory, Parallel, delayed

from .utils import (clean_by_interp, interpolate_bads, _get_epochs_type, _pbar,
                    _handle_picks, _check_data, _get_ch_type_from_picks,
                    _check_sub_picks)
from .bayesopt import expected_improvement, bayes_opt

mem = Memory(cachedir='cachedir')
mem.clear(warn=False)


def _slicemean(obj, this_slice, axis):
    mean = np.nan
    if len(obj[this_slice]) > 0:
        mean = np.mean(obj[this_slice], axis=axis)
    return mean


def validation_curve(estimator, epochs, y, param_name, param_range, cv=None):
    """Validation curve on epochs.

    Parameters
    ----------
    estimator : object that implements "fit" and "predict" method.
        the estimator whose Validation curve must be found
    epochs : instance of mne.Epochs.
        The epochs.
    y : array
        The labels.
    param_name : str
        Name of the parameter that will be varied.
    param_range : array
        The values of the parameter that will be evaluated.
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation strategy.

    Returns
    -------
    train_scores : array
        The scores in the training set
    test_scores : array
        The scores in the test set
    """
    from sklearn.model_selection import validation_curve
    if not isinstance(estimator, GlobalAutoReject):
        msg = 'No guarantee that it will work on this estimator.'
        raise NotImplementedError(msg)

    BaseEpochs = _get_epochs_type()
    if not isinstance(epochs, BaseEpochs):
        raise ValueError('Only accepts MNE epochs objects.')

    data_picks = _handle_picks(epochs.info, picks=None)
    X = epochs.get_data()[:, data_picks, :]
    n_epochs, n_channels, n_times = X.shape

    estimator.n_channels = n_channels
    estimator.n_times = n_times

    train_scores, test_scores = \
        validation_curve(estimator, X.reshape(n_epochs, -1), y=y,
                         param_name="thresh", param_range=param_range,
                         cv=cv, n_jobs=1, verbose=0)

    return train_scores, test_scores


class BaseAutoReject(BaseEstimator):
    """Base class for rejection."""

    def score(self, X):
        """Score it."""
        if hasattr(self, 'n_channels'):
            X = X.reshape(-1, self.n_channels, self.n_times)
        if np.any(np.isnan(self.mean_)):
            return -np.inf
        else:
            return -np.sqrt(np.mean((np.median(X, axis=0) - self.mean_) ** 2))

    def fit_transform(self, epochs):
        """Estimate rejection params and find bad epochs.

        Parameters
        ----------
        epochs : instance of mne.Epochs
            The epochs object which must be cleaned.
        """
        return self.fit(epochs).transform(epochs)


class GlobalAutoReject(BaseAutoReject):
    """Class to compute global rejection thresholds.

    Parameters
    ----------
    n_channels : int | None
        The number of channels in the epochs. Defaults to None.
    n_times : int | None
        The number of time points in the epochs. Defaults to None.
    thresh : float
        Boilerplate API. The rejection threshold.
    """

    def __init__(self, n_channels=None, n_times=None, thresh=40e-6):
        """Init it."""
        self.thresh = thresh
        self.n_channels = n_channels
        self.n_times = n_times

    def fit(self, X, y=None):
        """Fit it."""
        if self.n_channels is None or self.n_times is None:
            raise ValueError('Cannot fit without knowing n_channels'
                             ' and n_times')
        X = X.reshape(-1, self.n_channels, self.n_times)
        deltas = np.array([np.ptp(d, axis=1) for d in X])
        epoch_deltas = deltas.max(axis=1)
        keep = epoch_deltas <= self.thresh
        self.mean_ = _slicemean(X, keep, axis=0)
        return self


def get_rejection_threshold(epochs, decim=1, random_state=None):
    """Compute global rejection thresholds.

    Parameters
    ----------
    epochs : mne.Epochs object
        The epochs from which to estimate the epochs dictionary
    decim : int
        The decimation factor.
    random_state : int seed, RandomState instance, or None (default)
        The seed of the pseudo random number generator to use.

    Returns
    -------
    reject : dict
        The rejection dictionary with keys 'mag', 'grad', 'eeg', 'eog'
        and 'ecg'.

    Note
    ----
    Sensors marked as bad by user will be excluded when estimating the
    rejection dictionary.
    """
    reject = dict()
    if decim > 1:
        epochs = epochs.copy()
        epochs.decimate(decim=decim)
    for ch_type in ['mag', 'grad', 'eeg', 'eog']:
        if ch_type not in epochs:
            continue

        if ch_type == 'mag':
            picks = pick_types(epochs.info, meg='mag', eeg=False)
        elif ch_type == 'eeg':
            picks = pick_types(epochs.info, meg=False, eeg=True)
        elif ch_type == 'eog':
            picks = pick_types(epochs.info, meg=False, eog=True)
        elif ch_type == 'grad':
            picks = pick_types(epochs.info, meg='grad', eeg=False)

        X = epochs.get_data()[:, picks, :]
        n_epochs, n_channels, n_times = X.shape
        deltas = np.array([np.ptp(d, axis=1) for d in X])
        all_threshes = np.sort(deltas.max(axis=1))

        print('Estimating rejection dictionary for %s' % ch_type)
        cache = dict()
        est = GlobalAutoReject(n_channels=n_channels, n_times=n_times)
        cv = KFold(n_epochs, n_folds=5, random_state=random_state)

        def func(thresh):
            idx = np.where(thresh - all_threshes >= 0)[0][-1]
            thresh = all_threshes[idx]
            if thresh not in cache:
                est.set_params(thresh=thresh)
                obj = -np.mean(cross_val_score(est, X, cv=cv))
                cache.update({thresh: obj})
            return cache[thresh]

        n_epochs = all_threshes.shape[0]
        idx = np.concatenate((
            np.linspace(0, n_epochs, 5, endpoint=False, dtype=int),
            [n_epochs - 1]))  # ensure last point is in init
        idx = np.unique(idx)  # linspace may be non-unique if n_epochs < 5
        initial_x = all_threshes[idx]
        best_thresh, _ = bayes_opt(func, initial_x,
                                   all_threshes,
                                   expected_improvement,
                                   max_iter=10, debug=False,
                                   random_state=random_state)
        reject[ch_type] = best_thresh

    return reject


class _ChannelAutoReject(BaseAutoReject):
    """docstring for AutoReject."""

    def __init__(self, thresh=40e-6):
        self.thresh = thresh

    def fit(self, X, y=None):
        """Fit it.

        Parameters
        ----------
        X : array, shape (n_epochs, n_times)
            The data for one channel.
        y : None
            Redundant. Necessary to be compatible with sklearn
            API.
        """
        deltas = np.ptp(X, axis=1)
        self.deltas_ = deltas
        keep = deltas <= self.thresh
        # XXX: actually go over all the folds before setting the min
        # in skopt. Otherwise, may confuse skopt.
        if self.thresh < np.min(np.ptp(X, axis=1)):
            assert np.sum(keep) == 0
            keep = deltas <= np.min(np.ptp(X, axis=1))
        self.mean_ = _slicemean(X, keep, axis=0)
        return self


def _pick_exclusive_channels(info, ch_type):
    """Pick one and only one type."""
    if ch_type == 'eeg':
        picks = mne.pick_types(info, meg=False, eeg=True)
    elif ch_type == 'eog':
        picks = mne.pick_types(info, meg=False, eog=True)
    elif ch_type == 'meg':
        picks = mne.pick_types(info, meg=True)
    elif ch_type == 'grad' or ch_type == 'mag':
        picks = mne.pick_types(info, meg=ch_type)
    return picks


def _compute_thresh(this_data, method='bayesian_optimization',
                    cv=10, random_state=None):
    """Compute the rejection threshold for one channel.

    Parameters
    ----------
    this_data: array (n_epochs, n_times)
        Data for one channel.
    method : str
        'bayesian_optimization' or 'random_search'
    cv : iterator
        Iterator for cross-validation.
    random_state : int seed, RandomState instance, or None (default)
        The seed of the pseudo random number generator to use.

    Returns
    -------
    best_thresh : float
        The best threshold.

    Notes
    -----
    For method='random_search', the random_state parameter gives deterministic
    results only for scipy versions >= 0.16. This is why we recommend using
    autoreject with scipy version 0.16 or greater.
    """
    est = _ChannelAutoReject()
    all_threshes = np.sort(np.ptp(this_data, axis=1))

    if method == 'random_search':
        param_dist = dict(thresh=uniform(all_threshes[0],
                                         all_threshes[-1]))
        rs = RandomizedSearchCV(est,
                                param_distributions=param_dist,
                                n_iter=20, cv=cv,
                                random_state=random_state)
        rs.fit(this_data)
        best_thresh = rs.best_estimator_.thresh
    elif method == 'bayesian_optimization':
        from sklearn.cross_validation import cross_val_score
        cache = dict()

        def func(thresh):
            idx = np.where(thresh - all_threshes >= 0)[0][-1]
            thresh = all_threshes[idx]
            if thresh not in cache:
                est.set_params(thresh=thresh)
                obj = -np.mean(cross_val_score(est, this_data, cv=cv))
                cache.update({thresh: obj})
            return cache[thresh]

        n_epochs = all_threshes.shape[0]
        idx = np.concatenate((
            np.linspace(0, n_epochs, 40, endpoint=False, dtype=int),
            [n_epochs - 1]))  # ensure last point is in init
        idx = np.unique(idx)  # linspace may be non-unique if n_epochs < 40
        initial_x = all_threshes[idx]
        best_thresh, _ = bayes_opt(func, initial_x,
                                   all_threshes,
                                   expected_improvement,
                                   max_iter=10, debug=False,
                                   random_state=random_state)

    return best_thresh


def compute_thresholds(epochs, method='bayesian_optimization',
                       random_state=None, picks=None, augment=True,
                       verbose='progressbar', n_jobs=1):
    """Compute thresholds for each channel.

    Parameters
    ----------
    epochs : instance of mne.Epochs
        The epochs objects whose thresholds must be computed.
    method : str
        'bayesian_optimization' or 'random_search'
    random_state : int seed, RandomState instance, or None (default)
        The seed of the pseudo random number generator to use
    picks : ndarray, shape(n_channels,) | None
        The channels to be considered for autoreject. If None, defaults
        to data channels {'meg', 'eeg'}.
    augment : boolean
        Whether to augment the data or not. By default it is True, but
        set it to False, if the channel locations are not available.
    verbose : 'tqdm', 'tqdm_notebook', 'progressbar' or False
        The verbosity of progress messages.
        If `'progressbar'`, use `mne.utils.ProgressBar`.
        If `'tqdm'`, use `tqdm.tqdm`.
        If `'tqdm_notebook'`, use `tqdm.tqdm_notebook`.
        If False, suppress all output messages.
    n_jobs : int
        The number of jobs.

    Examples
    --------
    For example, we can compute the channel-level thresholds for all the
    EEG sensors this way:
        >>> compute_thresholds(epochs)
    """
    if method not in ['bayesian_optimization', 'random_search']:
        raise ValueError('`method` param not recognized')
    picks = _handle_picks(epochs.info, picks)
    _check_data(epochs, picks, verbose=verbose,
                ch_constraint='data_channels')
    sub_picks = _check_sub_picks(picks=picks, info=epochs.info)
    if sub_picks is not False:
        threshes = dict()
        for ch_type, this_picks in sub_picks:
            threshes.update(compute_thresholds(
                epochs=epochs, method=method, random_state=random_state,
                picks=this_picks, augment=augment, verbose=verbose,
                n_jobs=n_jobs))
    else:
        n_epochs = len(epochs)
        data, y = epochs.get_data(), np.ones((n_epochs, ))
        if augment:
            epochs_interp = clean_by_interp(epochs, picks=picks,
                                            verbose=verbose)
            # non-data channels will be duplicate
            data = np.concatenate((epochs.get_data(),
                                   epochs_interp.get_data()), axis=0)
            y = np.r_[np.zeros((n_epochs, )), np.ones((n_epochs, ))]
        cv = StratifiedShuffleSplit(y, n_iter=10, test_size=0.2,
                                    random_state=random_state)

        ch_names = epochs.ch_names

        my_thresh = delayed(_compute_thresh)
        verbose = 51 if verbose is not False else 0  # send output to stdout
        threshes = Parallel(n_jobs=n_jobs, verbose=verbose)(
            my_thresh(data[:, pick], cv=cv, method=method,
                      random_state=random_state) for pick in picks)
        threshes = {ch_names[p]: thresh for p, thresh in zip(picks, threshes)}
    return threshes


class LocalAutoReject(BaseAutoReject):
    r"""Automatically reject bad epochs and repair bad trials.

    Parameters
    ----------
    epochs : instance of mne.Epochs
        The epochs object
    consensus_perc : float (0 to 1.0)
        Percentage of channels that must agree as a fraction of
        the total number of channels. This sets :math:`\\kappa/Q`.
    n_interpolate : int (default 0)
        Number of channels for which to interpolate. This is :math:`\\rho`.
    thresh_func : callable | None
        Function which returns the channel-level thresholds. If None,
        defaults to :func:`autoreject.compute_thresholds`.
    picks : ndarray, shape(n_channels,) | None
        The channels to be considered for autoreject. If None, defaults
        to data channels {'meg', 'eeg'}.
    verbose : 'tqdm', 'tqdm_notebook', 'progressbar' or False
        The verbosity of progress messages.
        If `'progressbar'`, use `mne.utils.ProgressBar`.
        If `'tqdm'`, use `tqdm.tqdm`.
        If `'tqdm_notebook'`, use `tqdm.tqdm_notebook`.
        If False, suppress all output messages.

    Attributes
    -----------
    bad_segments : array, shape (n_epochs, n_channels)
        A boolean matrix where 1 denotes a bad data segment
        according to the sensor thresholds.
    fix_log : array, shape (n_epochs, n_channels)
        Similar to bad_segments, but with entries 0, 1, and 2.
            0 : good data segment
            1 : bad data segment not interpolated
            2 : bad data segment interpolated
    bad_epochs_idx : array
        The indices of bad epochs.
    threshes_ : dict
        The sensor-level thresholds with channel names as keys
        and the peak-to-peak thresholds as the values.
    """

    def __init__(self, consensus_perc=0.1,
                 n_interpolate=0, thresh_func=None,
                 method='bayesian_optimization',
                 picks=None,
                 verbose='progressbar'):
        """Init it."""
        if thresh_func is None:
            thresh_func = compute_thresholds
        if not (0 <= consensus_perc <= 1):
            raise ValueError('"consensus_perc" must be between 0 and 1. '
                             'You gave me %s.' % consensus_perc)
        self.consensus_perc = {
            ch: consensus_perc for ch in ('mag', 'grad', 'eeg')}
        self.n_interpolate = {
            ch: n_interpolate for ch in ('mag', 'grad', 'eeg')}
        self.thresh_func = thresh_func
        self.picks = picks
        self.verbose = verbose

    @property
    def bad_segments(self):
        return self.drop_log_

    @property
    def bad_epochs_idx(self):
        return self.bad_epochs_idx_

    def _vote_bad_epochs(self, epochs):
        """Each channel votes for an epoch as good or bad.

        Parameters
        ----------
        epochs : instance of mne.Epochs
            The epochs object for which bad epochs must be found.
        """
        n_epochs = len(epochs)
        picks = _handle_picks(info=epochs.info, picks=self.picks)

        drop_log = np.zeros((n_epochs, len(epochs.ch_names)))
        bad_sensor_counts = np.zeros((len(epochs), ))

        ch_names = [epochs.ch_names[p] for p in picks]
        deltas = np.ptp(epochs.get_data()[:, picks], axis=-1).T
        threshes = [self.threshes_[ch_name] for ch_name in ch_names]
        for ch_idx, (delta, thresh) in enumerate(zip(deltas, threshes)):
            bad_epochs_idx = np.where(delta > thresh)[0]
            # TODO: combine for different ch types
            bad_sensor_counts[bad_epochs_idx] += 1
            drop_log[bad_epochs_idx, picks[ch_idx]] = 1
        return drop_log, bad_sensor_counts

    def _get_epochs_interpolation(self, epochs, drop_log,
                                  ch_type, verbose='progressbar'):
        """Interpolate the bad epochs."""
        # 1: bad segment, # 2: interpolated
        fix_log = drop_log.copy()
        ch_names = epochs.ch_names
        non_picks = np.setdiff1d(range(epochs.info['nchan']), self.picks)
        interp_channels = list()
        n_interpolate = self.n_interpolate[ch_type]
        for epoch_idx in range(len(epochs)):
            n_bads = drop_log[epoch_idx, self.picks].sum()
            if n_bads == 0:
                continue
            else:
                if n_bads <= n_interpolate:
                    interp_chs_mask = drop_log[epoch_idx] == 1
                else:
                    # get peak-to-peak for channels in that epoch
                    data = epochs[epoch_idx].get_data()[0]
                    peaks = np.ptp(data, axis=-1)
                    peaks[non_picks] = -np.inf
                    # find channels which are bad by rejection threshold
                    interp_chs_mask = drop_log[epoch_idx] == 1
                    # ignore good channels
                    peaks[~interp_chs_mask] = -np.inf
                    # find the ordering of channels amongst the bad channels
                    sorted_ch_idx_picks = np.argsort(peaks)[::-1]
                    # then select only the worst n_interpolate channels
                    interp_chs_mask[
                        sorted_ch_idx_picks[n_interpolate:]] = False

            fix_log[epoch_idx][interp_chs_mask] = 2
            interp_chs = np.where(interp_chs_mask)[0]
            interp_chs = [ch_name for idx, ch_name in enumerate(ch_names)
                          if idx in interp_chs]
            interp_channels.append(interp_chs)
        return interp_channels, fix_log

    def _get_bad_epochs(self, bad_sensor_counts, ch_type):
        """Get the indices of bad epochs."""
        sorted_epoch_idx = np.argsort(bad_sensor_counts)[::-1]
        bad_sensor_counts = np.sort(bad_sensor_counts)[::-1]
        n_channels = len(self.picks)
        n_consensus = self.consensus_perc[ch_type] * n_channels
        if np.max(bad_sensor_counts) >= n_consensus:
            n_epochs_drop = np.sum(bad_sensor_counts >=
                                   n_consensus)
            bad_epochs_idx = sorted_epoch_idx[:n_epochs_drop]
        else:
            n_epochs_drop = 0
            bad_epochs_idx = []

        return bad_epochs_idx, sorted_epoch_idx, n_epochs_drop

    def _annotate_epochs(self, threshes, epochs):
        """Get essential annotations for epochs given thresholds."""
        ch_type = _get_ch_type_from_picks(self.picks, epochs.info)[0]

        drop_log, bad_sensor_counts = self._vote_bad_epochs(epochs)

        interp_channels, fix_log = self._get_epochs_interpolation(
            epochs, drop_log=drop_log, ch_type=ch_type)

        (bad_epochs_idx, sorted_epoch_idx,
         n_epochs_drop) = self._get_bad_epochs(
             bad_sensor_counts, ch_type=ch_type)

        bad_epochs_idx = np.sort(bad_epochs_idx)
        good_epochs_idx = np.setdiff1d(np.arange(len(epochs)),
                                       bad_epochs_idx)

        return (drop_log, bad_sensor_counts, interp_channels, fix_log,
                bad_epochs_idx, good_epochs_idx)

    def fit(self, epochs):
        """Compute the thresholds.

        Parameters
        ----------
        epochs : instance of mne.Epochs
            The epochs object from which the channel-level thresholds are
            estimated.

        Returns
        -------
        self : instance of LocalAutoReject
            The instance.
        """
        self.picks = _handle_picks(info=epochs.info, picks=self.picks)
        _check_data(epochs, picks=self.picks, verbose=self.verbose,
                    ch_constraint='single_channel_type')
        self.threshes_ = self.thresh_func(
            epochs.copy(), picks=self.picks, verbose=self.verbose)

        (drop_log, bad_sensor_counts, interp_channels, fix_log,
         bad_epochs_idx, good_epochs_idx) = self._annotate_epochs(
             threshes=self.threshes_, epochs=epochs)

        self.drop_log_ = drop_log
        self.fix_log_ = fix_log
        self.bad_sensor_counts_ = bad_sensor_counts
        self.interp_channels_ = interp_channels
        self.bad_epochs_idx_ = bad_epochs_idx
        self.good_epochs_idx_ = good_epochs_idx

        epochs_copy = epochs.copy()
        self._interpolate_bad_epochs(
            epochs_copy, interp_channels=interp_channels, verbose=self.verbose)
        self.mean_ = _slicemean(epochs_copy.get_data(),
                                good_epochs_idx, axis=0)
        del epochs_copy  # I can't wait for garbage collection.
        return self

    def transform(self, epochs):
        """Fix and find the bad epochs.

        .. note::
           LocalAutoReject partially supports multiple channels.
           While fitting, at this point requires selection of channel types,
           the transform can handle multiple channel types, if `.threshes_`
           parameter contains all necessary channels and `.consensus_perc`
           and `n_interpolate` have meaningful channel type specific
           settings. These are commonly obtained from
           :func:`autoreject.LocalAutoRejectCV`.

        Parameters
        ----------
        epochs : instance of mne.Epochs
            The epochs object for which bad epochs must be found.
        """
        _check_data(epochs, picks=self.picks, verbose=self.verbose,
                    ch_constraint='data_channels')
        if not all(epochs.ch_names[pp] in self.threshes_ for pp in self.picks):
            raise ValueError('You are passing channels which were not present '
                             'at fit-time. Please fit it again, this time '
                             'correctly.')
        epochs_out = epochs.copy()
        sub_picks = _check_sub_picks(picks=self.picks, info=epochs_out.info)
        if sub_picks is not False:
            bad_epochs_idx = list()
            interp_channels_list = list()
            old_picks = self.picks
            for ii, (ch_type, this_picks) in enumerate(sub_picks):
                self.picks = this_picks
                out = self._annotate_epochs(self.threshes_, epochs)
                interp_channels_list.append(out[2])
                bad_epochs_idx_ = out[4]
                bad_epochs_idx = np.union1d(bad_epochs_idx, bad_epochs_idx_)
                bad_epochs_idx = bad_epochs_idx.astype(np.int)
            good_epochs_idx = np.setdiff1d(np.arange(len(epochs)),
                                           bad_epochs_idx)
            if len(good_epochs_idx) == 0:
                raise ValueError('All epochs are bad. Sorry.')

            for ii, (ch_type, this_picks) in enumerate(sub_picks):
                self.picks = this_picks
                self._interpolate_bad_epochs(
                    epochs_out, interp_channels=interp_channels_list[ii],
                    verbose=self.verbose)
            self.picks = old_picks

        else:
            (_, _, interp_channels, _,
             bad_epochs_idx, good_epochs_idx) = self._annotate_epochs(
                 threshes=self.threshes_, epochs=epochs)
            if len(good_epochs_idx) == 0:
                raise ValueError('All epochs are bad. Sorry.')

            self._interpolate_bad_epochs(
                epochs_out, interp_channels=interp_channels,
                verbose=self.verbose)
        if np.any(bad_epochs_idx):
            epochs_out.drop(bad_epochs_idx, reason='AUTOREJECT')
        else:
            warnings.warn(
                "No bad epochs were found for your data. Returning "
                "a copy of the data you wanted to clean. Interpolation "
                "may have been done.")
        return epochs_out

    def _interpolate_bad_epochs(
            self, epochs, interp_channels, verbose='progressbar'):
        """Actually do the interpolation."""
        pos = 4 if hasattr(self, '_leave') else 2
        for epoch_idx, interp_chs in _pbar(
                list(enumerate(interp_channels)),
                desc='Repairing epochs',
                position=pos, leave=True, verbose=verbose):
            epoch = epochs[epoch_idx]
            epoch.info['bads'] = interp_chs
            interpolate_bads(epoch, picks=self.picks, reset_bads=True)
            epochs._data[epoch_idx] = epoch._data


class LocalAutoRejectCV(object):
    r"""Efficiently find n_interp and n_consensus.

    .. note::
       LocalAutoRejectCV by design supports multiple channels.
       If no picks are passed separate solutions will be computed for each
       channel type and internally combines. This then readily supports
       cleaning unseen epochs from the different channel types used during fit.

    Parameters
    ----------
    consensus_percs : array | None
        The values to try for percentage of channels that must agree as a
        fraction of the total number of channels. This sets :math:`\\kappa/Q`.
        If None, defaults to `np.linspace(0, 1.0, 11)`
    n_interpolates : array | None
        The values to try for the number of channels for which to interpolate.
        This is :math:`\\rho`.If None, defaults to
        np.array([1, 4, 32])
    thresh_func : callable | None
        Function which returns the channel-level thresholds. If None,
        defaults to :func:`autoreject.compute_thresholds`.
    cv : a scikit-learn cross-validation object
        Defaults to cv=10
    picks : ndarray, shape(n_channels) | None
        The channels to be considered for autoreject. If None, defaults
        to data channels {'meg', 'eeg'}, which will lead fitting and combining
        autoreject solutions across these channel types.
    verbose : 'tqdm', 'tqdm_notebook', 'progressbar' or False
        The verbosity of progress messages.
        If `'progressbar'`, use `mne.utils.ProgressBar`.
        If `'tqdm'`, use `tqdm.tqdm`.
        If `'tqdm_notebook'`, use `tqdm.tqdm_notebook`.
        If False, suppress all output messages.

    Attributes
    -----------
    bad_segments : array, shape (n_epochs, n_channels)
        A boolean matrix where 1 denotes a bad data segment
        according to the sensor thresholds.
    fix_log : array, shape (n_epochs, n_channels)
        Similar to bad_segments, but with entries 0, 1, and 2.
            0 : good data segment
            1 : bad data segment not interpolated
            2 : bad data segment interpolated
    bad_epochs_idx : array
        The indices of bad epochs.
    threshes_ : dict
        The sensor-level thresholds with channel names as keys
        and the peak-to-peak thresholds as the values.
    loss : array, shape (len(n_interpolates), len(consensus_percs))
        The cross validation error for different parameter values.
    consensus_perc_ : float
        The estimated consensus_perc.
    n_interpolate_ : int
        The estimated n_interpolated.
    """

    def __init__(self, n_interpolates=None, consensus_percs=None,
                 thresh_func=None, cv=None, picks=None,
                 verbose='progressbar'):
        """Init it."""
        self.n_interpolates = n_interpolates
        self.consensus_percs = consensus_percs
        self.thresh_func = thresh_func
        self.cv = cv
        self.verbose = verbose
        self.picks = picks
        self.loss_ = dict()
        self.consensus_perc_ = dict()
        self.n_interpolate_ = dict()

    @property
    def bad_segments(self):
        return self.local_reject_.drop_log_

    @property
    def fix_log(self):
        return self.local_reject_.fix_log_

    @property
    def bad_epochs_idx(self):
        return self.local_reject_.bad_epochs_idx_

    def fit(self, epochs):
        """Fit the epochs on the LocalAutoReject object.

        Parameters
        ----------
        epochs : instance of mne.Epochs
            The epochs object to be fit.

        Returns
        -------
        self : instance of LocalAutoRejectCV
            The instance.
        """
        self.picks = _handle_picks(info=epochs.info, picks=self.picks)
        _check_data(epochs, picks=self.picks, verbose=self.verbose)
        if self.cv is None:
            self.cv = 10
        if isinstance(self.cv, int):
            self.cv = KFold(len(epochs), n_folds=self.cv)
        if self.consensus_percs is None:
            self.consensus_percs = np.linspace(0, 1.0, 11)
        if self.n_interpolates is None:
            if len(self.picks) < 4:
                raise ValueError('Too few channels. autoreject is unlikely'
                                 ' to be effective')
            # XXX: dont interpolate all channels
            max_interp = min(len(self.picks) - 1, 32)
            self.n_interpolates = np.array([1, 4, max_interp])

        # Start recursion here if multiple channel types are present.
        sub_picks = _check_sub_picks(info=epochs.info, picks=self.picks)
        if sub_picks is not False:
            # store accumulation stuff here
            threshes = dict()  # update
            bad_segments = 0.0  # numpy broadcast + sum
            fix_log = 0.0  # ...
            bad_epochs_idx = list()
            consensus_perc = dict()
            n_interpolate = dict()
            for ch_type, this_picks in sub_picks:
                sub_ar = LocalAutoRejectCV(
                    n_interpolates=self.n_interpolates,
                    consensus_percs=self.consensus_percs,
                    thresh_func=self.thresh_func, cv=self.cv,
                    verbose=self.verbose)
                sub_ar.picks = this_picks
                sub_ar.fit(epochs)
                threshes.update(sub_ar.threshes_)
                bad_segments += sub_ar.bad_segments
                fix_log += sub_ar.fix_log
                bad_epochs_idx = np.union1d(
                    sub_ar.local_reject_.bad_epochs_idx_,
                    bad_epochs_idx).astype(int)

                consensus_perc[ch_type] = sub_ar.consensus_perc_[ch_type]
                n_interpolate[ch_type] = sub_ar.n_interpolate_[ch_type]

            good_epochs_idx = np.setdiff1d(np.arange(len(epochs)),
                                           bad_epochs_idx).astype(int)
            # assemble stuff, update and return self
            self.threshes_ = threshes
            self.local_reject_ = sub_ar.local_reject_
            self.local_reject_.threshes_ = threshes
            self.local_reject_.fix_log_ = fix_log
            self.local_reject_.drop_log_ = bad_segments
            self.local_reject_.bad_epochs_idx_ = bad_epochs_idx
            self.local_reject_.good_epochs_idx_ = good_epochs_idx

            self.n_interpolate_ = n_interpolate
            self.local_reject_.n_interpolate = n_interpolate
            self.consensus_perc_ = consensus_perc
            self.local_reject_.consensus_perc = consensus_perc
            return self

        # Continue here if only one channel type is present.
        n_folds = len(self.cv)
        loss = np.zeros((len(self.consensus_percs), len(self.n_interpolates),
                         n_folds))

        local_reject = LocalAutoReject(thresh_func=self.thresh_func,
                                       verbose=self.verbose,
                                       picks=self.picks)
        ch_type = _get_ch_type_from_picks(
            picks=self.picks, info=epochs.info)[0]

        # The thresholds must be learnt from the entire data
        local_reject.fit(epochs)
        self.threshes_ = local_reject.threshes_

        drop_log, bad_sensor_counts = local_reject._vote_bad_epochs(epochs)
        desc = 'n_interp'

        for jdx, n_interp in enumerate(_pbar(self.n_interpolates, desc=desc,
                                       position=1, verbose=self.verbose)):
            # we can interpolate before doing cross-valida(tion
            # because interpolation is independent across trials.
            local_reject.n_interpolate[ch_type] = n_interp
            interp_channels, fix_log = local_reject._get_epochs_interpolation(
                epochs, drop_log=drop_log, ch_type=ch_type)
            local_reject.interp_channels_ = interp_channels

            epochs_interp = epochs.copy()
            local_reject._interpolate_bad_epochs(
                epochs_interp, interp_channels=interp_channels,
                verbose=self.verbose)

            for fold, (train, test) in enumerate(_pbar(self.cv, desc='Fold',
                                                 position=3,
                                                 verbose=self.verbose)):
                for idx, consensus_perc in enumerate(self.consensus_percs):
                    # \kappa must be greater than \rho
                    n_channels = len(self.picks)
                    if consensus_perc * n_channels <= n_interp:
                        loss[idx, jdx, fold] = np.inf
                        continue

                    local_reject.consensus_perc[ch_type] = consensus_perc
                    local_reject.bad_sensor_counts = bad_sensor_counts[train]

                    bad_epochs_idx, _, _ = local_reject._get_bad_epochs(
                        bad_sensor_counts, ch_type=ch_type)
                    local_reject.bad_epochs_idx_ = np.sort(bad_epochs_idx)
                    n_train = len(epochs[train])
                    good_epochs_idx = np.setdiff1d(np.arange(n_train),
                                                   bad_epochs_idx)
                    local_reject.mean_ = _slicemean(
                        epochs_interp[train].get_data()[:, self.picks],
                        good_epochs_idx, axis=0)
                    X = epochs[test].get_data()[:, self.picks]
                    loss[idx, jdx, fold] = -local_reject.score(X)

        self.loss_[ch_type] = loss
        best_idx, best_jdx = np.unravel_index(loss.mean(axis=-1).argmin(),
                                              loss.shape[:2])
        consensus_perc = self.consensus_percs[best_idx]
        n_interpolate = self.n_interpolates[best_jdx]
        self.n_interpolate_[ch_type] = n_interpolate
        self.consensus_perc_[ch_type] = consensus_perc
        if self.verbose is not False:
            print('Estimated consensus_perc=%0.2f and n_interpolate=%d'
                  % (consensus_perc, n_interpolate))
        local_reject.consensus_perc[ch_type] = consensus_perc
        local_reject.n_interpolate[ch_type] = n_interpolate
        local_reject._leave = False
        out = local_reject._annotate_epochs(
            threshes=local_reject.threshes_, epochs=epochs)
        local_reject.fix_log_ = out[3]
        local_reject.bad_epochs_idx_ = out[4]
        local_reject.good_epochs_idx_ = out[5]
        self.local_reject_ = local_reject
        return self

    def transform(self, epochs):
        """Remove bad epochs, repairs sensors and returns clean epochs.

        Parameters
        ----------
        epochs : instance of mne.Epochs
            The epochs object which must be cleaned.

        Returns
        -------
        epochs_clean : instance of mne.Epochs
            The cleaned epochs
        """
        if len(self.n_interpolate_) == 0:
            raise ValueError('Please run autoreject.fit() method first')

        _check_data(epochs, picks=self.picks, verbose=self.verbose)
        old_picks = self.local_reject_.picks
        self.local_reject_.picks = self.picks

        epochs_clean = self.local_reject_.transform(epochs)

        self.local_reject_.picks = old_picks
        return epochs_clean

    def fit_transform(self, epochs):
        """Estimate the rejection params and finds bad epochs.

        Parameters
        ----------
        epochs : instance of mne.Epochs
            The epochs object which must be cleaned.

        Returns
        -------
        epochs_clean : instance of mne.Epochs
            The cleaned epochs
        """
        return self.fit(epochs).transform(epochs)
