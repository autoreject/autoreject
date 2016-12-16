"""Automated rejection and repair of trials in M/EEG."""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Denis A. Engemann <denis.engemann@gmail.com>

import numpy as np
from scipy.stats.distributions import uniform

import mne

from sklearn.base import BaseEstimator
from sklearn.model_selection import RandomizedSearchCV
from sklearn.cross_validation import KFold, StratifiedShuffleSplit

from joblib import Memory
from pandas import DataFrame

from .utils import clean_by_interp, interpolate_bads, _get_epochs_type, _pbar

mem = Memory(cachedir='cachedir')
mem.clear(warn=False)


def _check_data(epochs):
    BaseEpochs = _get_epochs_type()
    if not isinstance(epochs, BaseEpochs):
        raise ValueError('Only accepts MNE epochs objects.')

    # needed for len
    try:
        epochs.drop_bad()
    except AttributeError:
        epochs.drop_bad_epochs()
    if any(len(drop) > 0 and drop != ['IGNORED']
            for drop in epochs.drop_log):
        msg = ('Some epochs are being dropped (maybe due to '
               'incomplete data). Please check that no epoch '
               'is dropped when you call epochs.drop_bad_epochs().')
        raise RuntimeError(msg)


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
    """
    from sklearn.model_selection import validation_curve
    if not isinstance(estimator, GlobalAutoReject):
        msg = 'No guarantee that it will work on this estimator.'
        raise NotImplementedError(msg)

    BaseEpochs = _get_epochs_type()
    if not isinstance(epochs, BaseEpochs):
        raise ValueError('Only accepts MNE epochs objects.')

    X = epochs.get_data()
    n_epochs, n_channels, n_times = X.shape

    estimator.n_channels = n_channels
    estimator.n_times = n_times

    train_scores, test_scores = \
        validation_curve(estimator, X.reshape(n_epochs, -1), y=y,
                         param_name="thresh", param_range=param_range,
                         cv=cv, n_jobs=1, verbose=1)

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
    if ch_type == 'eeg':
        picks = mne.pick_types(info, meg=False, eeg=True)
    elif ch_type == 'eog':
        picks = mne.pick_types(info, meg=False, eog=True)
    elif ch_type == 'meg':
        picks = mne.pick_types(info, meg=True)
    elif ch_type == 'grad' or ch_type == 'mag':
        picks = mne.pick_types(info, meg=ch_type)
    return picks


def _compute_thresh(this_data, thresh_range, method='bayesian_optimization',
                    cv=10, random_state=None):
    """ Compute the rejection threshold for one channel.

    Parameters
    ----------
    this_data: array (n_epochs, n_times)
        Data for one channel.
    thresh_range : tuple
        The range (low, high) of thresholds over which to optimize.
    method : str
        'bayesian_optimization' or 'random_search'
    cv : iterator
        Iterator for cross-validation.
    random_state : int seed, RandomState instance, or None (default)
        The seed of the pseudo random number generator to use.

    Returns
    -------
    rs : instance of RandomizedSearchCV
        The RandomizedSearchCV object.

    Notes
    -----
    For method='random_search', the random_state parameter gives deterministic
    results only for scipy versions >= 0.16. This is why we recommend using
    autoreject with scipy version 0.16 or greater.
    """
    est = _ChannelAutoReject()

    if method == 'random_search':
        param_dist = dict(thresh=uniform(thresh_range[0],
                                         thresh_range[1]))
        rs = RandomizedSearchCV(est,
                                param_distributions=param_dist,
                                n_iter=20, cv=cv,
                                random_state=random_state)
        rs.fit(this_data)
    elif method == 'bayesian_optimization':
        from skopt import gp_minimize
        from sklearn.cross_validation import cross_val_score

        def objective(thresh):
            est.set_params(thresh=thresh)
            return -np.mean(cross_val_score(est, this_data, cv=cv))
        space = [(thresh_range[0], thresh_range[1])]
        rs = gp_minimize(objective, space, n_calls=50,
                         random_state=random_state)

    return rs


def compute_thresholds(epochs, method='bayesian_optimization',
                       random_state=None, verbose='progressbar'):
    """Compute thresholds for each channel.

    Parameters
    ----------
    epochs : instance of mne.Epochs
        The epochs objects whose thresholds must be computed.
    method : str
        'bayesian_optimization' or 'random_search'
    random_state : int seed, RandomState instance, or None (default)
        The seed of the pseudo random number generator to use
    verbose : 'tqdm', 'tqdm_notebook', 'progressbar' or False
        The verbosity of progress messages.
        If `'progressbar'`, use `mne.utils.ProgressBar`.
        If `'tqdm'`, use `tqdm.tqdm`.
        If `'tqdm_notebook'`, use `tqdm.tqdm_notebook`.
        If False, suppress all output messages.

    Examples
    --------
    For example, we can compute the channel-level thresholds for all the
    EEG sensors this way:
        >>> compute_thresholds(epochs)
    """
    if method not in ['bayesian_optimization', 'random_search']:
        raise ValueError('`method` param not recognized')
    ch_types = [ch_type for ch_type in ('eeg', 'meg')
                if ch_type in epochs]
    n_epochs = len(epochs)
    epochs_interp = clean_by_interp(epochs, verbose=verbose)
    data = np.concatenate((epochs.get_data(), epochs_interp.get_data()),
                          axis=0)
    y = np.r_[np.zeros((n_epochs, )), np.ones((n_epochs, ))]
    cv = StratifiedShuffleSplit(y, n_iter=10, test_size=0.2,
                                random_state=random_state)

    threshes = dict()
    for ch_type in ch_types:
        picks = _pick_exclusive_channels(epochs.info, ch_type)
        threshes[ch_type] = []
        for ii, pick in enumerate(_pbar(picks, desc='Computing thresholds',
                                  verbose=verbose)):
            # lower bound must be minimum ptp, otherwise random search
            # screws up.
            thresh_low = np.min(np.ptp(data[:, pick], axis=1))
            thresh_high = np.max(np.ptp(data[:, pick], axis=1))
            rs = _compute_thresh(data[:, pick], cv=cv, method=method,
                                 thresh_range=(thresh_low, thresh_high),
                                 random_state=random_state)
            if method == 'random_search':
                thresh = rs.best_estimator_.thresh
            elif method == 'bayesian_optimization':
                thresh = rs.x[0]
            threshes[ch_type].append(thresh)
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
    verbose : 'tqdm', 'tqdm_notebook', 'progressbar' or False
        The verbosity of progress messages.
        If `'progressbar'`, use `mne.utils.ProgressBar`.
        If `'tqdm'`, use `tqdm.tqdm`.
        If `'tqdm_notebook'`, use `tqdm.tqdm_notebook`.
        If False, suppress all output messages.
    """
    def __init__(self, thresh_func=None, consensus_perc=0.1,
                 n_interpolate=0, method='bayesian_optimization',
                 verbose='progressbar'):
        if thresh_func is None:
            thresh_func = compute_thresholds
        if not (0 <= consensus_perc <= 1):
            raise ValueError('"consensus_perc" must be between 0 and 1. '
                             'You gave me %s.' % consensus_perc)
        self.consensus_perc = consensus_perc
        self.n_interpolate = n_interpolate
        self.thresh_func = thresh_func
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
        self.threshes_ = self.thresh_func(epochs, verbose=self.verbose)
        return self

    def transform(self, epochs):
        """Fixes and finds the bad epochs.

        Parameters
        ----------
        epochs : instance of mne.Epochs
            The epochs object for which bad epochs must be found.
        """
        epochs = epochs.copy()
        _check_data(epochs)

        self._vote_epochs(epochs)
        ch_types = [ch_type for ch_type in ('eeg', 'meg') if ch_type in epochs]
        for ch_type in ch_types:
            self._interpolate_bad_epochs(epochs, ch_type=ch_type,
                                         verbose=self.verbose)

        bad_epochs_idx = self._get_bad_epochs()
        self._bad_epochs_idx = np.sort(bad_epochs_idx)
        self.good_epochs_idx = np.setdiff1d(np.arange(len(epochs)),
                                            bad_epochs_idx)
        self.mean_ = _slicemean(epochs.get_data(),
                                self.good_epochs_idx, axis=0)
        return epochs[self.good_epochs_idx]

    def _vote_epochs(self, epochs):
        """Each channel votes for an epoch as good or bad

        Parameters
        ----------
        epochs : instance of mne.Epochs
            The epochs object for which bad epochs must be found.
        """
        n_epochs = len(epochs)
        picks = mne.pick_types(epochs.info, meg=True, eeg=True, eog=True)
        self._drop_log = DataFrame(np.zeros((n_epochs, len(picks)), dtype=int),
                                   columns=epochs.info['ch_names'])
        self.bad_epoch_counts = np.zeros((len(epochs), ))
        ch_types = [ch_type for ch_type in ('eeg', 'meg')
                    if ch_type in epochs]
        for ch_type in ch_types:
            picks = _pick_exclusive_channels(epochs.info, ch_type)
            ch_names = [epochs.info['ch_names'][p] for p in picks]
            deltas = np.ptp(epochs.get_data()[:, picks], axis=-1).T
            threshes = self.threshes_[ch_type]
            for delta, thresh, ch_name in zip(deltas, threshes, ch_names):
                bad_epochs_idx = np.where(delta > thresh)[0]
                # TODO: combine for different ch types
                self.bad_epoch_counts[bad_epochs_idx] += 1
                self._drop_log.ix[bad_epochs_idx, ch_name] = 1

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

    def _interpolate_bad_epochs(self, epochs, ch_type, verbose='progressbar'):
        """interpolate the bad epochs.

        Parameters
        ----------
        epochs : instance of mne.Epochs
            The epochs object which must be fixed.
        """
        drop_log = self._drop_log
        # 1: bad segment, # 2: interpolated, # 3: dropped
        self.fix_log = self._drop_log.copy()
        ch_names = drop_log.columns.values
        n_consensus = self.consensus_perc * len(ch_names)
        # TODO: raise error if preload is not True
        pos = 4 if hasattr(self, '_leave') else 2
        for epoch_idx in _pbar(range(len(epochs)), desc='Repairing epochs',
                               position=pos, leave=True, verbose=verbose):
            n_bads = drop_log.ix[epoch_idx].sum()
            if n_bads == 0 or n_bads > n_consensus:
                continue
            else:
                if n_bads <= self.n_interpolate:
                    bad_chs = drop_log.ix[epoch_idx].values == 1
                else:
                    # get peak-to-peak for channels in that epoch
                    data = epochs[epoch_idx].get_data()[0, :, :]
                    peaks = np.ptp(data, axis=-1)
                    # find channels which are bad by rejection threshold
                    bad_chs = np.where(drop_log.ix[epoch_idx].values == 1)[0]
                    # find the ordering of channels amongst the bad channels
                    sorted_ch_idx = np.argsort(peaks[bad_chs])[::-1]
                    # then select only the worst n_interpolate channels
                    bad_chs = bad_chs[sorted_ch_idx[:self.n_interpolate]]

            self.fix_log.ix[epoch_idx][bad_chs] = 2
            bad_chs = ch_names[bad_chs].tolist()
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
    verbose : 'tqdm', 'tqdm_notebook', 'progressbar' or False
        The verbosity of progress messages.
        If `'progressbar'`, use `mne.utils.ProgressBar`.
        If `'tqdm'`, use `tqdm.tqdm`.
        If `'tqdm_notebook'`, use `tqdm.tqdm_notebook`.
        If False, suppress all output messages.

    Returns
    -------
    local_ar : instance of LocalAutoReject
        The fitted LocalAutoReject object.
    """

    def __init__(self, n_interpolates=None, consensus_percs=None,
                 thresh_func=None, method='bayesian_optimization', cv=None,
                 verbose='progressbar'):
        self.n_interpolates = n_interpolates
        self.consensus_percs = consensus_percs
        self.thresh_func = thresh_func
        self.cv = cv
        self.verbose = verbose

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
        """
        _check_data(epochs)
        if self.cv is None:
            self.cv = KFold(len(epochs), n_folds=10)
        if self.consensus_percs is None:
            self.consensus_percs = np.linspace(0, 1.0, 11)
        if self.n_interpolates is None:
            if epochs.info['nchan'] < 4:
                raise ValueError('Too few channels. Auto reject is unlikely'
                                 ' to be effective')
            max_interp = min(epochs.info['nchan'], 32)
            self.n_interpolates = np.array([1, 4, max_interp])

        n_folds = len(self.cv)
        loss = np.zeros((len(self.consensus_percs), len(self.n_interpolates),
                         n_folds))

        local_reject = LocalAutoReject(thresh_func=self.thresh_func,
                                       verbose=self.verbose)

        # The thresholds must be learnt from the entire data
        local_reject.fit(epochs)

        local_reject._vote_epochs(epochs)
        bad_epoch_counts = local_reject.bad_epoch_counts.copy()
        desc = 'n_interp'
        for jdx, n_interp in enumerate(_pbar(self.n_interpolates, desc=desc,
                                       position=1, verbose=self.verbose)):
            # we can interpolate before doing cross-validation
            # because interpolation is independent across trials.
            local_reject.n_interpolate = n_interp
            ch_types = [ch_type for ch_type in ('eeg', 'meg') if
                        ch_type in epochs]
            epochs_interp = epochs.copy()
            for ch_type in ch_types:
                local_reject._interpolate_bad_epochs(epochs_interp,
                                                     ch_type=ch_type,
                                                     verbose=self.verbose)
            for fold, (train, test) in enumerate(_pbar(self.cv, desc='Fold',
                                                 position=3,
                                                 verbose=self.verbose)):
                for idx, consensus_perc in enumerate(self.consensus_percs):
                    local_reject.consensus_perc = consensus_perc
                    local_reject.bad_epoch_counts = bad_epoch_counts[train]

                    bad_epochs_idx = local_reject._get_bad_epochs()
                    local_reject._bad_epochs_idx = np.sort(bad_epochs_idx)
                    n_train = len(epochs[train])
                    good_epochs_idx = np.setdiff1d(np.arange(n_train),
                                                   bad_epochs_idx)
                    local_reject.mean_ = _slicemean(
                        epochs_interp[train].get_data(),
                        good_epochs_idx, axis=0)
                    X = epochs[test].get_data()
                    loss[idx, jdx, fold] = -local_reject.score(X)

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
        return self._local_reject.transform(epochs)

    def fit_transform(self, epochs):
        """Estimates the rejection params and finds bad epochs.

        Parameters
        ----------
        epochs : instance of mne.Epochs
            The epochs object which must be cleaned.
        """
        return self.fit(epochs).transform(epochs)
