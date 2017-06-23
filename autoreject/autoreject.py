"""Automated rejection and repair of trials in M/EEG."""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Denis A. Engemann <denis.engemann@gmail.com>

import numpy as np
from scipy.stats.distributions import uniform

import mne
from mne.io.pick import channel_indices_by_type
from mne.utils import logger

from sklearn.base import BaseEstimator
from sklearn.model_selection import RandomizedSearchCV
from sklearn.cross_validation import KFold, StratifiedShuffleSplit

from sklearn.externals.joblib import Memory, Parallel, delayed

from .utils import (clean_by_interp, interpolate_bads, _get_epochs_type, _pbar,
                    _handle_picks)

from .bayesopt import expected_improvement, bayes_opt

mem = Memory(cachedir='cachedir')
mem.clear(warn=False)


def _check_data(epochs, picks, verbose='progressbar'):
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

    if sum(ch in ch_types_picked for ch in ('mag', 'grad', 'eeg')) > 1:
        raise ValueError('AutoReject handles only one channel type for now')

    if n_bads > 0:
        if verbose is not False:
            logger.info(
                '%i channels are marked as bad. These will be ignored.'
                'If you want them to be considered by autoreject please '
                'remove them from epochs.info["bads"].' % n_bads)


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
        if hasattr(self, 'n_channels'):
            X = X.reshape(-1, self.n_channels, self.n_times)
        if np.any(np.isnan(self.mean_)):
            return -np.inf
        else:
            return -np.sqrt(np.mean((np.median(X, axis=0) - self.mean_) ** 2))

    def fit_transform(self, epochs):
        """Estimates the rejection params and finds bad epochs.

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
        self.thresh = thresh
        self.n_channels = n_channels
        self.n_times = n_times

    def fit(self, X, y=None):
        if self.n_channels is None or self.n_times is None:
            raise ValueError('Cannot fit without knowing n_channels'
                             ' and n_times')
        X = X.reshape(-1, self.n_channels, self.n_times)
        deltas = np.array([np.ptp(d, axis=1) for d in X])
        epoch_deltas = deltas.max(axis=1)
        keep = epoch_deltas <= self.thresh
        self.mean_ = _slicemean(X, keep, axis=0)
        return self


def get_rejection_threshold(epochs):
    """Compute global rejection thresholds.

    Parameters
    ----------
    epochs : mne.Epochs object
        The epochs from which to estimate the epochs dictionary

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
    X = epochs.get_data()
    picks = channel_indices_by_type(epochs.info)
    for ch_type in ['mag', 'grad', 'eeg', 'eog', 'ecg']:
        if ch_type not in epochs:
            continue
        if ch_type == 'ecg' and 'mag' not in epochs:
            continue
        if ch_type == 'eog' and not \
                ('mag' in epochs or 'grad' in epochs or 'eeg' in epochs):
            continue

        this_picks = [p for p in picks[ch_type] if epochs.info['ch_names'][p]
                      not in epochs.info['bads']]
        deltas = np.array([np.ptp(d, axis=1) for d in X[:, this_picks, :]])
        param_range = deltas.max(axis=1)
        print('Estimating rejection dictionary for %s with %d candidate'
              ' thresholds' % (ch_type, param_range.shape[0]))

        if ch_type == 'mag' or ch_type == 'ecg':
            this_epoch = epochs.copy().pick_types(meg='mag', eeg=False)
        elif ch_type == 'eeg':
            this_epoch = epochs.copy().pick_types(meg=False, eeg=True)
        elif ch_type == 'eog':
            # Cannot mix channel types in cv score
            if 'eeg' in epochs:
                this_epoch = epochs.copy().pick_types(meg=False, eeg=True)
            elif 'grad' in epochs:
                this_epoch = epochs.copy().pick_types(meg='grad', eeg=False)
            elif 'mag' in epochs:
                this_epoch = epochs.copy().pick_types(meg='mag', eeg=False)
        elif ch_type == 'grad':
            this_epoch = epochs.copy().pick_types(meg='grad', eeg=False)

        _, test_scores = validation_curve(
            GlobalAutoReject(), this_epoch, y=None,
            param_name="thresh", param_range=param_range, cv=5)

        test_scores = -test_scores.mean(axis=1)
        reject[ch_type] = param_range[np.argmin(test_scores)]
    return reject


class _ChannelAutoReject(BaseAutoReject):
    """docstring for AutoReject"""

    def __init__(self, thresh=40e-6):
        self.thresh = thresh

    def fit(self, X, y=None):
        """
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
    """pick one and only one type."""
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
    """ Compute the rejection threshold for one channel.

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
        bounds = [(all_threshes[0], all_threshes[-1])]
        best_thresh, _ = bayes_opt(func, initial_x, bounds,
                                   expected_improvement,
                                   max_iter=10, debug=False,
                                   random_state=random_state)

    return best_thresh


def compute_thresholds(epochs, method='bayesian_optimization',
                       random_state=None, picks=None, verbose='progressbar',
                       n_jobs=1):
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
    _check_data(epochs, picks, verbose=verbose)
    n_epochs = len(epochs)
    picks = _handle_picks(info=epochs.info, picks=picks)
    epochs_interp = clean_by_interp(epochs, picks=picks, verbose=verbose)
    data = np.concatenate((epochs.get_data(), epochs_interp.get_data()),
                          axis=0)  # non-data channels will be duplicate
    y = np.r_[np.zeros((n_epochs, )), np.ones((n_epochs, ))]
    cv = StratifiedShuffleSplit(y, n_iter=10, test_size=0.2,
                                random_state=random_state)

    threshes = dict()
    ch_names = epochs_interp.ch_names

    my_thresh = delayed(_compute_thresh)
    verbose = 51 if verbose is not False else 0  # send output to stdout
    threshes = Parallel(n_jobs=n_jobs, verbose=verbose)(
        my_thresh(data[:, pick], cv=cv, method=method,
                  random_state=random_state) for pick in picks)
    threshes = {ch_names[p]: thresh for p, thresh in zip(picks, threshes)}
    return threshes


class LocalAutoReject(BaseAutoReject):
    """Class to automatically reject bad epochs and repair bad trials.

    Parameters
    ----------
    epochs : instance of mne.Epochs
        The epochs object
    thresh_func : callable | None
        Function which returns the channel-level thresholds. If None,
        defaults to :func:`autoreject.compute_thresholds`.
    consensus_perc : float (0 to 1.0)
        percentage of channels that must agree as a fraction of
        the total number of channels.
    n_interpolate : int (default 0)
        Number of channels for which to interpolate
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
    def __init__(self, thresh_func=None, consensus_perc=0.1,
                 n_interpolate=0, method='bayesian_optimization',
                 picks=None,
                 verbose='progressbar'):
        if thresh_func is None:
            thresh_func = compute_thresholds
        if not (0 <= consensus_perc <= 1):
            raise ValueError('"consensus_perc" must be between 0 and 1. '
                             'You gave me %s.' % consensus_perc)
        self.consensus_perc = consensus_perc
        self.n_interpolate = n_interpolate
        self.thresh_func = thresh_func
        self.picks = picks
        self.verbose = verbose

    @property
    def bad_segments(self):
        return self._drop_log

    @property
    def bad_epochs_idx(self):
        return self._bad_epochs_idx

    def fit(self, epochs):
        """Compute the thresholds.

        Parameters
        ----------
        epochs : instance of mne.Epochs
            The epochs object from which the channel-level thresholds are
            estimated.
        """
        self.picks = _handle_picks(info=epochs.info, picks=self.picks)
        self.threshes_ = self.thresh_func(
            epochs.copy(), picks=self.picks, verbose=self.verbose)
        return self

    def transform(self, epochs):
        """Fixes and finds the bad epochs.

        Parameters
        ----------
        epochs : instance of mne.Epochs
            The epochs object for which bad epochs must be found.
        """
        _check_data(epochs, picks=self.picks, verbose=self.verbose)
        self._vote_epochs(epochs)
        self._interpolate_bad_epochs(epochs, verbose=self.verbose)

        bad_epochs_idx = self._get_bad_epochs()
        self._bad_epochs_idx = np.sort(bad_epochs_idx)
        self.good_epochs_idx = np.setdiff1d(np.arange(len(epochs)),
                                            bad_epochs_idx)
        self.mean_ = _slicemean(epochs.get_data(),
                                self.good_epochs_idx, axis=0)
        epochs.drop(bad_epochs_idx, reason='AUTOREJECT')
        return epochs

    def _vote_epochs(self, epochs):
        """Each channel votes for an epoch as good or bad

        Parameters
        ----------
        epochs : instance of mne.Epochs
            The epochs object for which bad epochs must be found.
        """
        n_epochs = len(epochs)
        picks = _handle_picks(info=epochs.info, picks=self.picks)

        self._drop_log = np.zeros((n_epochs, len(epochs.ch_names)))
        self.bad_epoch_counts = np.zeros((len(epochs), ))

        ch_names = [epochs.ch_names[p] for p in picks]

        deltas = np.ptp(epochs.get_data()[:, picks], axis=-1).T
        threshes = [self.threshes_[ch_name] for ch_name in ch_names]
        for ch_idx, (delta, thresh) in enumerate(zip(deltas, threshes)):
            bad_epochs_idx = np.where(delta > thresh)[0]
            # TODO: combine for different ch types
            self.bad_epoch_counts[bad_epochs_idx] += 1
            self._drop_log[bad_epochs_idx, picks[ch_idx]] = 1

    def _get_bad_epochs(self):
        """Get the indices of bad epochs.
        """
        # TODO: this must be done separately for each channel type?
        self.sorted_epoch_idx = np.argsort(self.bad_epoch_counts)[::-1]
        bad_epoch_counts = np.sort(self.bad_epoch_counts)[::-1]
        n_channels = self._drop_log.shape[1]
        n_consensus = self.consensus_perc * n_channels
        if np.max(bad_epoch_counts) >= n_consensus:
            self.n_epochs_drop = np.sum(self.bad_epoch_counts >=
                                        n_consensus) + 1
            bad_epochs_idx = self.sorted_epoch_idx[:self.n_epochs_drop]
        else:
            self.n_epochs_drop = 0
            bad_epochs_idx = []

        return bad_epochs_idx

    def _interpolate_bad_epochs(self, epochs, verbose='progressbar'):
        """interpolate the bad epochs.

        Parameters
        ----------
        epochs : instance of mne.Epochs
            The epochs object which must be fixed.
        """
        drop_log = self._drop_log
        # 1: bad segment, # 2: interpolated
        self.fix_log = self._drop_log.copy()
        ch_names = [epochs.ch_names[pp] for pp in self.picks]
        non_picks = np.setdiff1d(range(epochs.info['nchan']), self.picks)
        # TODO: raise error if preload is not True
        pos = 4 if hasattr(self, '_leave') else 2
        for epoch_idx in _pbar(range(len(epochs)), desc='Repairing epochs',
                               position=pos, leave=True, verbose=verbose):
            n_bads = drop_log[epoch_idx].sum()
            if n_bads == 0:
                continue
            else:
                if n_bads <= self.n_interpolate:
                    bad_chs_mask = drop_log[epoch_idx] == 1
                else:
                    # get peak-to-peak for channels in that epoch
                    data = epochs[epoch_idx].get_data()[0]
                    peaks = np.ptp(data, axis=-1)
                    peaks[non_picks] = -np.inf
                    # find channels which are bad by rejection threshold
                    bad_chs_mask = drop_log[epoch_idx] == 1
                    # find the ordering of channels amongst the bad channels
                    sorted_ch_idx_picks = np.argsort(peaks)[::-1]
                    # then select only the worst n_interpolate channels
                    bad_chs_mask[
                        sorted_ch_idx_picks[self.n_interpolate:]] = False

            self.fix_log[epoch_idx][bad_chs_mask] = 2
            bad_chs = np.where(bad_chs_mask)[0]
            bad_chs = [ch_name for idx, ch_name in enumerate(ch_names)
                       if idx in bad_chs]
            epoch = epochs[epoch_idx]
            epoch.info['bads'] = bad_chs
            interpolate_bads(epoch, reset_bads=True)
            epochs._data[epoch_idx] = epoch._data


class LocalAutoRejectCV(object):
    """Efficiently find n_interp and n_consensus.

    Parameters
    ----------
    n_interpolates : array | None
        The values of :math:`\\rho` to try. If None, defaults to
        np.array([1, 4, 32])
    consensus_percs : array | None
        The values of :math:`\\kappa/Q` to try. If None, defaults
        to `np.linspace(0, 1.0, 11)`
    thresh_func : callable | None
        Function which returns the channel-level thresholds. If None,
        defaults to :func:`autoreject.compute_thresholds`.
    cv : a scikit-learn cross-validation object
        Defaults to cv=10
    picks : ndarray, shape(n_channels) | None
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
    loss : array, shape (len(n_interpolates), len(consensus_percs))
        The cross validation error for different parameter values.
    consensus_percs_ : float
        The estimated consensus_perc.
    n_interpolates_ : int
        The estimated n_interpolated.
    """

    def __init__(self, n_interpolates=None, consensus_percs=None,
                 thresh_func=None, method='bayesian_optimization', cv=None,
                 picks=None,
                 verbose='progressbar'):
        self.n_interpolates = n_interpolates
        self.consensus_percs = consensus_percs
        self.thresh_func = thresh_func
        self.cv = cv
        self.verbose = verbose
        self.picks = picks

    @property
    def bad_segments(self):
        return self._local_reject._drop_log

    @property
    def fix_log(self):
        return self._local_reject.fix_log

    @property
    def bad_epochs_idx(self):
        return self._local_reject._bad_epochs_idx

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

        n_folds = len(self.cv)
        loss = np.zeros((len(self.consensus_percs), len(self.n_interpolates),
                         n_folds))

        local_reject = LocalAutoReject(thresh_func=self.thresh_func,
                                       verbose=self.verbose,
                                       picks=self.picks)

        # The thresholds must be learnt from the entire data
        local_reject.fit(epochs)
        self.threshes_ = local_reject.threshes_

        local_reject._vote_epochs(epochs)
        bad_epoch_counts = local_reject.bad_epoch_counts.copy()
        desc = 'n_interp'
        for jdx, n_interp in enumerate(_pbar(self.n_interpolates, desc=desc,
                                       position=1, verbose=self.verbose)):
            # we can interpolate before doing cross-validation
            # because interpolation is independent across trials.
            local_reject.n_interpolate = n_interp
            epochs_interp = epochs.copy()

            local_reject._interpolate_bad_epochs(epochs_interp,
                                                 verbose=self.verbose)
            for fold, (train, test) in enumerate(_pbar(self.cv, desc='Fold',
                                                 position=3,
                                                 verbose=self.verbose)):
                for idx, consensus_perc in enumerate(self.consensus_percs):
                    # \kappa must be greater than \rho
                    n_channels = local_reject._drop_log.shape[1]
                    if consensus_perc * n_channels <= n_interp:
                        loss[idx, jdx, fold] = np.inf
                        continue
                    local_reject.consensus_perc = consensus_perc
                    local_reject.bad_epoch_counts = bad_epoch_counts[train]

                    bad_epochs_idx = local_reject._get_bad_epochs()
                    local_reject._bad_epochs_idx = np.sort(bad_epochs_idx)
                    n_train = len(epochs[train])
                    good_epochs_idx = np.setdiff1d(np.arange(n_train),
                                                   bad_epochs_idx)
                    local_reject.mean_ = _slicemean(
                        epochs_interp[train].get_data()[:, self.picks],
                        good_epochs_idx, axis=0)
                    X = epochs[test].get_data()[:, self.picks]
                    loss[idx, jdx, fold] = -local_reject.score(X)

        self.loss = loss
        best_idx, best_jdx = np.unravel_index(loss.mean(axis=-1).argmin(),
                                              loss.shape[:2])
        consensus_perc = self.consensus_percs[best_idx]
        n_interpolate = self.n_interpolates[best_jdx]
        self.consensus_perc_ = consensus_perc
        self.n_interpolate_ = n_interpolate
        if self.verbose is not False:
            print('Estimated consensus_perc=%0.2f and n_interpolate=%d'
                  % (consensus_perc, n_interpolate))
        local_reject.consensus_perc = consensus_perc
        local_reject.n_interpolate = n_interpolate
        local_reject._leave = False
        self._local_reject = local_reject
        return self

    def transform(self, epochs):
        """Removes bad epochs, repairs sensors and returns clean epochs.

        Parameters
        ----------
        epochs : instance of mne.Epochs
            The epochs object which must be cleaned.
        """
        if not hasattr(self, 'n_interpolate_'):
            raise ValueError('Please run autoreject.fit() method first')

        _check_data(epochs, picks=self.picks, verbose=self.verbose)
        return self._local_reject.transform(epochs.copy())

    def fit_transform(self, epochs):
        """Estimates the rejection params and finds bad epochs.

        Parameters
        ----------
        epochs : instance of mne.Epochs
            The epochs object which must be cleaned.
        """
        return self.fit(epochs).transform(epochs)
