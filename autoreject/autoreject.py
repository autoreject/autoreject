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
                    _handle_picks, _check_data,
                    _get_picks_by_type)
from .bayesopt import expected_improvement, bayes_opt
from .viz import plot_epochs

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

    data_picks = _handle_picks(info=epochs.info, picks=None)
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

    # def fit_transform(self, epochs):
    #     """Estimate rejection params and find bad epochs.

    #     Parameters
    #     ----------
    #     epochs : instance of mne.Epochs
    #         The epochs object which must be cleaned.
    #     """
    #     return self.fit(epochs).transform(epochs)


# XXX : make it private !
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
    picks = _handle_picks(info=epochs.info, picks=picks)
    _check_data(epochs, picks, verbose=verbose,
                ch_constraint='data_channels')
    picks_by_type = _get_picks_by_type(picks=picks, info=epochs.info)
    picks_by_type = None if len(picks_by_type) == 1 else picks_by_type  # XXX
    if picks_by_type is not None:
        threshes = dict()
        for ch_type, this_picks in picks_by_type:
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
    consensus : float (0 to 1.0)
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
    labels : array, shape (n_epochs, n_channels)
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

    def __init__(self, consensus=0.1,
                 n_interpolate=0, thresh_func=None,
                 method='bayesian_optimization',
                 picks=None,
                 verbose='progressbar'):
        """Init it."""
        if thresh_func is None:
            thresh_func = compute_thresholds
        if not (0 <= consensus <= 1):
            raise ValueError('"consensus" must be between 0 and 1. '
                             'You gave me %s.' % consensus)
        self.consensus = consensus
        self.n_interpolate = n_interpolate
        self.consensus_ = dict()
        self.n_interpolate_ = dict()  # XXX : this will break ...
        self.thresh_func = thresh_func
        self.picks = picks
        self.verbose = verbose

    def _vote_bad_epochs(self, epochs, picks):
        """Each channel votes for an epoch as good or bad.

        Parameters
        ----------
        epochs : instance of mne.Epochs
            The epochs object for which bad epochs must be found.
        """
        n_epochs = len(epochs)
        picks = _handle_picks(info=epochs.info, picks=picks)

        ch_names = [cc for cc in epochs.ch_names]
        drop_log = np.zeros((n_epochs, len(ch_names)))
        bad_sensor_counts = np.zeros((len(epochs), ))

        this_ch_names = [epochs.ch_names[p] for p in picks]
        deltas = np.ptp(epochs.get_data()[:, picks], axis=-1).T
        threshes = [self.threshes_[ch_name] for ch_name in this_ch_names]
        for ch_idx, (delta, thresh) in enumerate(zip(deltas, threshes)):
            bad_epochs_idx = np.where(delta > thresh)[0]
            # TODO: combine for different ch types
            bad_sensor_counts[bad_epochs_idx] += 1
            drop_log[bad_epochs_idx, picks[ch_idx]] = 1
        return drop_log, bad_sensor_counts, ch_names

    def _get_epochs_interpolation(self, epochs, drop_log,
                                  ch_type, picks, n_interpolate,
                                  verbose='progressbar'):
        """Interpolate the bad epochs."""
        # 1: bad segment, # 2: interpolated
        labels = drop_log.copy()
        non_picks = np.setdiff1d(range(epochs.info['nchan']), picks)
        for epoch_idx in range(len(epochs)):
            n_bads = drop_log[epoch_idx, picks].sum()
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

            labels[epoch_idx][interp_chs_mask] = 2
        return labels

    def _get_bad_epochs(self, bad_sensor_counts, ch_type, picks):
        """Get the indices of bad epochs."""
        sorted_epoch_idx = np.argsort(bad_sensor_counts)[::-1]
        bad_sensor_counts = np.sort(bad_sensor_counts)[::-1]
        n_channels = len(picks)
        n_consensus = self.consensus_[ch_type] * n_channels
        bad_epochs = np.zeros(len(bad_sensor_counts), dtype=np.bool)
        if np.max(bad_sensor_counts) >= n_consensus:
            n_epochs_drop = np.sum(bad_sensor_counts >=
                                   n_consensus)
            bad_epochs_idx = sorted_epoch_idx[:n_epochs_drop]
            bad_epochs[bad_epochs_idx] = True

        return bad_epochs

    def _get_reject_log(self, epochs, picks):
        """Get essential annotations for epochs given thresholds."""
        ch_type = _get_picks_by_type(
            picks=picks, info=epochs.info)[0][0]

        drop_log, bad_sensor_counts, ch_names = self._vote_bad_epochs(
            epochs, picks=picks)
        labels = self._get_epochs_interpolation(
            epochs, drop_log=drop_log, ch_type=ch_type, picks=picks,
            n_interpolate=self.n_interpolate_[ch_type])

        assert len(labels) == len(epochs)

        bad_epochs = self._get_bad_epochs(
            bad_sensor_counts, ch_type=ch_type, picks=picks)

        return (drop_log, bad_sensor_counts, labels, ch_names, bad_epochs)

    def get_reject_log(self, epochs, threshes=None, picks=None):
        """Annotate epochs.

        .. note::
           If multiple channel types are present, reject_log.bad_epochs
           reflects the union of bad epochs across channel types.

        Parameters
        ----------
        epochs : instance of mne.Epochs
            The epoched data to be annotated.
        picks : np.ndarray, shape(n_channels, ) | list | None
            The channel indices to be used. If None, the .picks attribute
            will be used.

        Returns
        -------
        reject_log : instance of autoreject.RejectLog
            The rejection log.
        """
        picks = (self.picks_ if picks is None else picks)
        picks_by_type = _get_picks_by_type(picks=picks, info=epochs.info)
        assert len(picks_by_type) == 1
        ch_type, this_picks = picks_by_type[0]
        (_, _, labels, ch_names, bad_epochs) = self._get_reject_log(
            epochs, picks=picks)
        reject_log = RejectLog(labels=labels, bad_epochs=bad_epochs,
                               ch_names=ch_names)
        return reject_log

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
        self.picks_ = _handle_picks(info=epochs.info, picks=self.picks)
        _check_data(epochs, picks=self.picks_, verbose=self.verbose,
                    ch_constraint='single_channel_type')
        ch_type, picks_ = _get_picks_by_type(
            picks=self.picks_, info=epochs.info)[0]
        self.n_interpolate_[ch_type] = self.n_interpolate
        self.consensus_[ch_type] = self.consensus

        self.threshes_ = self.thresh_func(
            epochs.copy(), picks=self.picks_, verbose=self.verbose)

        reject_log = self.get_reject_log(epochs=epochs, picks=self.picks_)

        epochs_copy = epochs.copy()
        interp_channels = _get_interp_chs(
            reject_log.labels, reject_log.ch_names, picks_)
        self._interpolate_bad_epochs(
            epochs_copy, interp_channels=interp_channels,
            picks=self.picks_, verbose=self.verbose)
        self.mean_ = _slicemean(epochs_copy.get_data(),
                                np.nonzero(np.invert(reject_log.bad_epochs)),
                                axis=0)
        del epochs_copy  # I can't wait for garbage collection.ba
        return self

    def transform(self, epochs, return_log=False):
        """Fix and find the bad epochs.

        .. note::
           LocalAutoReject partially supports multiple channels.
           While fitting, at this point requires selection of channel types,
           the transform can handle multiple channel types, if `.threshes_`
           parameter contains all necessary channels and `.consensus`
           and `n_interpolate` have meaningful channel type specific
           settings. These are commonly obtained from
           :func:`autoreject.LocalAutoRejectCV`.

        Parameters
        ----------
        epochs : instance of mne.Epochs
            The epochs object for which bad epochs must be found.

        return_log : bool
            If true the rejection log is also returned.

        Returns
        -------
        epochs_clean : instance of mne.Epochs
            The cleaned epochs.

        reject_log : instance of autoreject.RejectLog
            The rejection log. Returned only of return_log is True.
        """
        _check_data(epochs, picks=self.picks, verbose=self.verbose,
                    ch_constraint='data_channels')

        reject_log = self.get_reject_log(epochs, picks=None)
        if np.all(reject_log.bad_epochs):
            raise ValueError('All epochs are bad. Sorry.')

        epochs_clean = epochs.copy()
        picks_by_type = _get_picks_by_type(picks=self.picks_, info=epochs.info)
        for ch_type, this_picks in picks_by_type:
            _apply_interp(reject_log, self, epochs_clean, self.threshes_,
                          this_picks, self.verbose)

        _apply_drop(reject_log, self, epochs_clean, self.threshes_,
                    self.picks_, self.verbose)

        if return_log:
            return epochs_clean, reject_log
        else:
            return epochs_clean

    def _interpolate_bad_epochs(
            self, epochs, interp_channels, picks, verbose='progressbar'):
        """Actually do the interpolation."""
        pos = 4 if hasattr(self, '_leave') else 2
        assert len(epochs) == len(interp_channels)

        for epoch_idx, interp_chs in _pbar(
                list(enumerate(interp_channels)),
                desc='Repairing epochs',
                position=pos, leave=True, verbose=verbose):
            epoch = epochs[epoch_idx]
            epoch.info['bads'] = interp_chs
            interpolate_bads(epoch, picks=picks, reset_bads=True)
            epochs._data[epoch_idx] = epoch._data


def _run_local_reject_cv(epochs, thresh_func, picks_, n_interpolates, cv,
                         consensus, verbose):

    n_folds = len(cv)
    loss = np.zeros((len(consensus), len(n_interpolates),
                    n_folds))

    # The thresholds must be learnt from the entire data
    local_reject = LocalAutoReject(thresh_func=thresh_func,
                                   verbose=verbose,
                                   picks=picks_)
    local_reject.fit(epochs)
    assert len(local_reject.consensus_) == 1  # works with one ch_type
    ch_type = next(iter(local_reject.consensus_))

    drop_log, bad_sensor_counts, ch_names = local_reject._vote_bad_epochs(
        epochs, picks=picks_)
    desc = 'n_interp'

    for jdx, n_interp in enumerate(_pbar(n_interpolates, desc=desc,
                                   position=1, verbose=verbose)):
        # we can interpolate before doing cross-valida(tion
        # because interpolation is independent across trials.
        # local_reject.n_interpolate_[ch_type] = n_interp
        labels = local_reject._get_epochs_interpolation(
            epochs, drop_log=drop_log, ch_type=ch_type, picks=picks_,
            n_interpolate=n_interp)

        interp_channels = _get_interp_chs(labels, ch_names, picks_)
        epochs_interp = epochs.copy()
        local_reject._interpolate_bad_epochs(
            epochs_interp, interp_channels=interp_channels,
            picks=picks_, verbose=verbose)

        for fold, (train, test) in enumerate(_pbar(cv, desc='Fold',
                                             position=3,
                                             verbose=verbose)):
            for idx, this_consensus in enumerate(consensus):
                # \kappa must be greater than \rho
                n_channels = len(picks_)
                if this_consensus * n_channels <= n_interp:
                    loss[idx, jdx, fold] = np.inf
                    continue

                local_reject.consensus_[ch_type] = this_consensus

                bad_epochs = local_reject._get_bad_epochs(
                    bad_sensor_counts, picks=picks_, ch_type=ch_type)

                good_epochs_idx = np.nonzero(
                    np.invert(bad_epochs[train]))[0]
                local_reject.mean_ = _slicemean(
                    epochs_interp[train].get_data()[:, picks_],
                    good_epochs_idx, axis=0)
                X = epochs[test].get_data()[:, picks_]
                loss[idx, jdx, fold] = -local_reject.score(X)

    return local_reject, loss


class LocalAutoRejectCV(object):
    r"""Efficiently find n_interp and n_consensus.

    .. note::
       LocalAutoRejectCV by design supports multiple channels.
       If no picks are passed separate solutions will be computed for each
       channel type and internally combined. This then readily supports
       cleaning unseen epochs from the different channel types used during fit.

    Parameters
    ----------
    consensus : array | None
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
    local_reject_ : list
        The instances of LocalAutoReject for each channel type.
    threshes_ : dict
        The sensor-level thresholds with channel names as keys
        and the peak-to-peak thresholds as the values.
    loss_ : dict of array, shape (len(n_interpolates), len(consensus))
        The cross validation error for different parameter values.
    consensus_ : dict
        The estimated consensus per channel type.
    n_interpolate_ : dict
        The estimated n_interpolate per channel type.
    """

    def __init__(self, n_interpolates=None, consensus=None,
                 thresh_func=None, cv=10, picks=None,
                 verbose='progressbar'):
        """Init it."""
        self.n_interpolates = n_interpolates
        self.consensus = consensus
        self.thresh_func = thresh_func
        self.cv = cv
        self.verbose = verbose
        self.picks = picks  # XXX : should maybe be ch_types?

        if self.consensus is None:
            self.consensus = np.linspace(0, 1.0, 11)

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
        self.picks_ = _handle_picks(picks=self.picks, info=epochs.info)
        _check_data(epochs, picks=self.picks_, verbose=self.verbose)
        cv = self.cv
        if isinstance(cv, int):
            # XXX: don't change cv !
            cv = KFold(len(epochs), n_folds=cv)

        if self.n_interpolates is None:
            if len(self.picks_) < 4:
                raise ValueError('Too few channels. autoreject is unlikely'
                                 ' to be effective')
            # XXX: dont interpolate all channels
            max_interp = min(len(self.picks_) - 1, 32)
            # XXX : self.n_interpolates -> self.n_interpolates_
            # why self.n_interpolates and self.n_interpolate_ ???
            self.n_interpolates = np.array([1, 4, max_interp])

        # XXX : maybe use an mne function in pick.py ?
        picks_by_type = _get_picks_by_type(info=epochs.info, picks=self.picks_)

        self.n_interpolate_ = dict()  # rho
        self.consensus_ = dict()  # kappa
        self.threshes_ = dict()  # update
        self.loss_ = dict()
        self.local_reject_ = dict()

        for ch_type, this_picks in picks_by_type:
            this_local_reject, this_loss = \
                _run_local_reject_cv(epochs, self.thresh_func, this_picks,
                                     self.n_interpolates, cv,
                                     self.consensus, self.verbose)
            self.threshes_.update(this_local_reject.threshes_)
            self.consensus_[ch_type] = this_local_reject.consensus_[ch_type]
            self.n_interpolate_[ch_type] = \
                this_local_reject.n_interpolate_[ch_type]
            self.loss_[ch_type] = this_loss

            best_idx, best_jdx = \
                np.unravel_index(this_loss.mean(axis=-1).argmin(),
                                 this_loss.shape[:2])
            self.n_interpolate_[ch_type] = self.n_interpolates[best_jdx]
            self.consensus_[ch_type] = self.consensus[best_idx]
            self.local_reject_[ch_type] = this_local_reject
            if self.verbose is not False:
                print('Estimated consensus=%0.2f and n_interpolate=%d'
                      % (self.consensus_[ch_type],
                         self.n_interpolate_[ch_type]))

        return self

    def get_reject_log(self, epochs, picks=None):
        """Annotate epochs.

        .. note::
           If multiple channel types are present, annot['bad_epochs_idx']
           reflects the union of bad trials across channel types.

        Parameters
        ----------
        epochs : instance of mne.Epochs
            The epoched data to be annotated.
        picks : np.ndarray, shape(n_channels, ) | list | None
            The channel indices to be used. If None, the .picks attribute
            will be used.

        Returns
        -------
        reject_log : instance of autoreject.RejectLog
            The reject log.
        """
        # XXX gut feeling that there is a bad condition that we miss
        ch_names = [cc for cc in epochs.ch_names]
        labels = np.ones((len(epochs), len(ch_names)))
        labels.fill(np.nan)
        reject_log = RejectLog(
            labels=labels,
            bad_epochs=np.zeros(len(epochs), dtype=np.bool),
            ch_names=ch_names)

        picks_by_type = _get_picks_by_type(info=epochs.info, picks=self.picks_)
        for ch_type, this_picks in picks_by_type:
            this_reject_log = self.local_reject_[ch_type].get_reject_log(
                epochs, threshes=self.threshes_, picks=this_picks)
            reject_log.labels[:, this_picks] = \
                this_reject_log.labels[:, this_picks]
            reject_log.bad_epochs = np.logical_xor(
                reject_log.bad_epochs, this_reject_log.bad_epochs)
        return reject_log

    def transform(self, epochs, return_log=False):
        """Remove bad epochs, repairs sensors and returns clean epochs.

        Parameters
        ----------
        epochs : instance of mne.Epochs
            The epochs object which must be cleaned.

        return_log : bool
            If true the rejection log is also returned.

        Returns
        -------
        epochs_clean : instance of mne.Epochs
            The cleaned epochs

        reject_log : instance of autoreject.RejectLog
            The rejection log. Returned only of return_log is True.
        """
        # XXX : should be a check_fitted method
        if not hasattr(self, 'n_interpolate_'):
            raise ValueError('Please run autoreject.fit() method first')

        _check_data(epochs, picks=self.picks_, verbose=self.verbose)

        reject_log = self.get_reject_log(epochs)
        epochs_clean = epochs.copy()
        picks_by_type = dict(
            _get_picks_by_type(info=epochs.info, picks=self.picks_))
        for ch_type, lr in self.local_reject_.items():
            this_picks = picks_by_type[ch_type]
            _apply_interp(reject_log, lr, epochs_clean, self.threshes_,
                          this_picks, self.verbose)

        _apply_drop(reject_log, lr, epochs_clean, self.threshes_, self.picks_,
                    self.verbose)

        if return_log:
            return epochs_clean, reject_log
        else:
            return epochs_clean

    def fit_transform(self, epochs, return_log=False):
        """Estimate the rejection params and finds bad epochs.

        Parameters
        ----------
        epochs : instance of mne.Epochs
            The epochs object which must be cleaned.

        return_log : bool
            If true the rejection log is also returned.

        Returns
        -------
        epochs_clean : instance of mne.Epochs
            The cleaned epochs.

        reject_log : instance of autoreject.RejectLog
            The rejection log. Returned only of return_log is True.
        """
        return self.fit(epochs).transform(epochs, return_log=return_log)


def _check_fit(epochs, threshes_, picks_):
    if not all(epochs.ch_names[pp] in threshes_
       for pp in picks_):
            raise ValueError(
                'You are passing channels which were not present '
                'at fit-time. Please fit it again, this time '
                'correctly.')


def _apply_interp(reject_log, local_reject, epochs, threshes_, picks_,
                  verbose):
    _check_fit(epochs, threshes_, picks_)
    interp_channels = _get_interp_chs(
        reject_log.labels, reject_log.ch_names, picks_)
    local_reject._interpolate_bad_epochs(
        epochs, interp_channels=interp_channels,
        picks=picks_, verbose=verbose)


def _apply_drop(reject_log, local_reject, epochs, threshes_, picks_,
                verbose):
    _check_fit(epochs, threshes_, picks_)
    if np.any(reject_log.bad_epochs):
        epochs.drop(np.nonzero(reject_log.bad_epochs)[0],
                    reason='AUTOREJECT')
    else:
        warnings.warn(
            "No bad epochs were found for your data. Returning "
            "a copy of the data you wanted to clean. Interpolation "
            "may have been done.")


def _get_interp_chs(labels, ch_names, picks):
    """Convert labels to channel names."""
    interp_channels = list()
    for this_labels in labels:
        interp = (this_labels == 2)
        interp_channels.append(
            [ch_names[ii] for ii in np.nonzero(interp)[0]
             if ii in picks])
    return interp_channels


class RejectLog(object):
    """The Rejection Log.

    Parameters
    ----------
    bad_epochs : array-like, shape (n_epochs,)
        The boolean array with entries True for epochs that
        are marked as bad.
    labels : array, shape (n_channels, n_epochs)
        It contains integers that encode if a channel in a given
        epoch is good (value 0), bad (1), or bad and interpolated (2).
    ch_names : list of str
        The list of channels corresponding to the rows of the labels.
    """

    def __init__(self, bad_epochs, labels, ch_names):
        self.bad_epochs = bad_epochs
        self.labels = labels
        self.ch_names = ch_names

    def plot(self, ch_type):
        import matplotllib.pyplots as plt
        # set_matplotlib_defaults(plt)  # XXX : don't hard code this!

        plt.figure(figsize=(12, 6))
        plt.imshow(self.labels, cmap='Reds',
                   interpolation='nearest')
        # XXX to be fixed
        ch_names_ = self.ch_names[7::10]
        ax = plt.gca()
        ax.grid(False)
        ax.set_xlabel('Channels')
        ax.set_ylabel('Epochs')
        plt.setp(ax, xticks=range(7, len(picks), 10),
                 xticklabels=ch_names_)
        plt.setp(ax.get_yticklabels(), rotation=0)
        plt.setp(ax.get_xticklabels(), rotation=90)
        ax.tick_params(axis=u'both', which=u'both', length=0)
        plt.tight_layout(rect=[None, None, None, 1.1])
        plt.show()

    def plot_epochs(self, epochs, scalings, title=''):
        """Plot interpolated and dropped epochs."""
        return plot_epochs(
            epochs=epochs,
            bad_epochs_idx=np.where(self.bad_epochs)[0],
            log_labels=self.labels, scalings=scalings,
            title='')
