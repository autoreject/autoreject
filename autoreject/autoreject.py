"""Automated rejection and repair of trials in M/EEG."""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Denis A. Engemann <denis.engemann@gmail.com>

import os.path as op
from functools import partial

import numpy as np
from scipy.stats.distributions import uniform

from joblib import Parallel, delayed

try:  # for mne < 1.0
    from mne.externals.h5io import read_hdf5, write_hdf5
except ImportError:
    from h5io import read_hdf5, write_hdf5

import mne
from mne import pick_types
from mne.viz import plot_epochs as plot_mne_epochs

from sklearn.base import BaseEstimator
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold, StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score, check_cv

from .utils import (_clean_by_interp, interpolate_bads, _get_epochs_type,
                    _pbar, _handle_picks, _check_data, _compute_dots,
                    _get_picks_by_type, _pprint)
from .bayesopt import expected_improvement, bayes_opt

_INIT_PARAMS = ('consensus', 'n_interpolate', 'picks',
                'verbose', 'n_jobs', 'cv', 'random_state',
                'thresh_method')

_FIT_PARAMS = ('threshes_', 'n_interpolate_', 'consensus_',
               'dots', 'picks_', 'loss_')


def _slicemean(obj, this_slice, axis):
    mean = np.nan
    if len(obj[this_slice]) > 0:
        mean = np.mean(obj[this_slice], axis=axis)
    return mean


def validation_curve(epochs, y=None, param_name="thresh", param_range=None,
                     cv=None, return_param_range=False, n_jobs=1):
    """Validation curve on epochs for global autoreject.

    Parameters
    ----------
    epochs : instance of mne.Epochs.
        The epochs.
    y : array | None
        The labels.
    param_name : str
        Name of the parameter that will be varied.
        Defaults to 'thresh'.
    param_range : array | None
        The values of the parameter that will be evaluated.
        If None, 15 values between the min and the max threshold
        will be tested.
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation strategy.
    return_param_range : bool
        If True the used param_range is returned.
        Defaults to False.
    n_jobs : int
        The number of thresholds to compute in parallel.

    Returns
    -------
    train_scores : array
        The scores in the training set
    test_scores : array
        The scores in the test set
    param_range : array
        The thresholds used to build the validation curve.
        Only returned if `return_param_range` is True.
    """
    from sklearn.model_selection import validation_curve
    estimator = _GlobalAutoReject()

    BaseEpochs = _get_epochs_type()
    if not isinstance(epochs, BaseEpochs):
        raise ValueError('Only accepts MNE epochs objects.')

    data_picks = _handle_picks(info=epochs.info, picks=None)
    X = epochs.get_data()[:, data_picks, :]
    n_epochs, n_channels, n_times = X.shape

    if param_range is None:
        ptps = np.ptp(X, axis=2)
        param_range = np.linspace(ptps.min(), ptps.max(), 15)

    estimator.n_channels = n_channels
    estimator.n_times = n_times

    train_scores, test_scores = \
        validation_curve(estimator, X.reshape(n_epochs, -1), y=y,
                         param_name="thresh", param_range=param_range,
                         cv=cv, n_jobs=n_jobs, verbose=0)

    out = (train_scores, test_scores)
    if return_param_range:
        out += (param_range,)

    return out


def read_auto_reject(fname):
    """Read AutoReject object.

    Parameters
    ----------
    fname : str
        The filename where the AutoReject object is saved.

    Returns
    -------
    ar : instance of autoreject.AutoReject
    """
    state = read_hdf5(fname, title='autoreject')
    ar = AutoReject()
    ar.__setstate__(state)
    return ar


class BaseAutoReject(BaseEstimator):
    """Base class for rejection."""

    def score(self, X, y=None):
        """Score it."""
        if hasattr(self, 'n_channels'):
            X = X.reshape(-1, self.n_channels, self.n_times)
        if np.any(np.isnan(self.mean_)):
            return -np.inf
        else:
            return -np.sqrt(np.mean((np.median(X, axis=0) - self.mean_) ** 2))


class _GlobalAutoReject(BaseAutoReject):
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


def get_rejection_threshold(epochs, decim=1, random_state=None,
                            ch_types=None, cv=5, verbose=True):
    """Compute global rejection thresholds.

    Parameters
    ----------
    epochs : mne.Epochs object
        The epochs from which to estimate the epochs dictionary
    decim : int
        The decimation factor: Increment for selecting every nth time slice.
    random_state : int seed, RandomState instance, or None (default)
        The seed of the pseudo random number generator to use.
    ch_types : str | list of str | None
        The channel types for which to find the rejection dictionary.
        e.g., ['mag', 'grad']. If None, the rejection dictionary
        will have keys ['mag', 'grad', 'eeg', 'eog', 'hbo', 'hbr'].
    cv : a scikit-learn cross-validation object
        Defaults to cv=5
    verbose : boolean
        The verbosity of progress messages.
        If False, suppress all output messages.

    Returns
    -------
    reject : dict
        The rejection dictionary with keys as specified by ch_types.

    Notes
    -----
    Sensors marked as bad by user will be excluded when estimating the
    rejection dictionary.
    """
    reject = dict()

    if ch_types is not None and not isinstance(ch_types, (list, str)):
        raise ValueError('ch_types must be of type None, list,'
                         'or str. Got %s' % type(ch_types))

    if ch_types is None:
        ch_types = ['mag', 'grad', 'eeg', 'eog', 'hbo', 'hbr']
    elif isinstance(ch_types, str):
        ch_types = [ch_types]

    if decim > 1:
        epochs = epochs.copy()
        epochs.decimate(decim=decim)

    cv = check_cv(cv)

    for ch_type in ch_types:
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
        elif ch_type in ['hbo', 'hbr']:
            picks = pick_types(epochs.info, meg=False, fnirs=ch_type)

        X = epochs.get_data()[:, picks, :]
        n_epochs, n_channels, n_times = X.shape
        deltas = np.array([np.ptp(d, axis=1) for d in X])
        all_threshes = np.sort(deltas.max(axis=1))

        if verbose:
            print('Estimating rejection dictionary for %s' % ch_type)
        cache = dict()
        est = _GlobalAutoReject(n_channels=n_channels, n_times=n_times)

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


def _compute_thresh(this_data, method='bayesian_optimization',
                    cv=10, y=None, random_state=None):
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
        rs.fit(this_data, y)
        best_thresh = rs.best_estimator_.thresh
    elif method == 'bayesian_optimization':
        cache = dict()

        def func(thresh):
            idx = np.where(thresh - all_threshes >= 0)[0][-1]
            thresh = all_threshes[idx]
            if thresh not in cache:
                est.set_params(thresh=thresh)
                obj = -np.mean(cross_val_score(est, this_data, y=y, cv=cv))
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
                       verbose=True, n_jobs=1):
    """Compute thresholds for each channel.

    Parameters
    ----------
    epochs : instance of mne.Epochs
        The epochs objects whose thresholds must be computed.
    method : str
        'bayesian_optimization' or 'random_search'
    random_state : int seed, RandomState instance, or None (default)
        The seed of the pseudo random number generator to use
    picks : str | list | slice | None
        Channels to include. Slices and lists of integers will be interpreted
        as channel indices. In lists, channel *type* strings (e.g.,
        ``['meg', 'eeg']``) will pick channels of those types, channel *name*
        strings (e.g., ``['MEG0111', 'MEG2623']`` will pick the given channels.
        Can also be the string values ``'all'`` to pick all channels, or
        ``'data'`` to pick data channels. None (default) will pick data
        channels {'meg', 'eeg'}. Note that channels in ``info['bads']`` *will
        be included* if their names or indices are explicitly provided.
    augment : boolean
        Whether to augment the data or not. By default it is True, but
        set it to False, if the channel locations are not available.
    verbose : boolean
        The verbosity of progress messages.
        If False, suppress all output messages.
    n_jobs : int
        Number of jobs to run in parallel

    Returns
    -------
    threshes : dict
        The channel-level rejection thresholds

    Examples
    --------
    For example, we can compute the channel-level thresholds for all the
    EEG sensors this way:

    >>> compute_thresholds(epochs)
    """
    return _compute_thresholds(epochs, method=method,
                               random_state=random_state, picks=picks,
                               augment=augment, verbose=verbose, n_jobs=n_jobs)


def _compute_thresholds(epochs, method='bayesian_optimization',
                        random_state=None, picks=None, augment=True,
                        dots=None, verbose=True, n_jobs=1):
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
            threshes.update(_compute_thresholds(
                epochs=epochs, method=method, random_state=random_state,
                picks=this_picks, augment=augment, dots=dots,
                verbose=verbose, n_jobs=n_jobs))
    else:
        n_epochs = len(epochs)
        data, y = epochs.get_data(), np.ones((n_epochs, ))
        if augment:
            epochs_interp = _clean_by_interp(epochs, picks=picks,
                                             dots=dots, verbose=verbose)
            # non-data channels will be duplicate
            data = np.concatenate((epochs.get_data(),
                                   epochs_interp.get_data()), axis=0)
            y = np.r_[np.zeros((n_epochs, )), np.ones((n_epochs, ))]
        cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2,
                                    random_state=random_state)

        ch_names = epochs.ch_names

        my_thresh = delayed(_compute_thresh)
        parallel = Parallel(n_jobs=n_jobs, verbose=0)
        desc = 'Computing thresholds ...'
        threshes = parallel(
            my_thresh(data[:, pick], cv=cv, method=method, y=y,
                      random_state=random_state)
            for pick in _pbar(picks, desc=desc, verbose=verbose))
        threshes = {ch_names[p]: thresh for p, thresh in zip(picks, threshes)}
    return threshes


class _AutoReject(BaseAutoReject):
    r"""Automatically reject bad epochs and repair bad trials.

    Parameters
    ----------
    n_interpolate : int (default 0)
        Number of channels for which to interpolate. This is :math:`\\rho`.
    consensus : float (0 to 1.0)
        Percentage of channels that must agree as a fraction of
        the total number of channels. This sets :math:`\\kappa/Q`.
    thresh_func : callable | None
        Function which returns the channel-level thresholds. If None,
        defaults to :func:`autoreject.compute_thresholds`.
    picks : str | list | slice | None
        Channels to include. Slices and lists of integers will be interpreted
        as channel indices. In lists, channel *type* strings (e.g.,
        ``['meg', 'eeg']``) will pick channels of those types, channel *name*
        strings (e.g., ``['MEG0111', 'MEG2623']`` will pick the given channels.
        Can also be the string values ``'all'`` to pick all channels, or
        ``'data'`` to pick data channels. None (default) will pick data
        channels {'meg', 'eeg'}. Note that channels in ``info['bads']`` *will
        be included* if their names or indices are explicitly provided.
    thresh_method : str
        'bayesian_optimization' or 'random_search'.
    dots : tuple
        2-length tuple returned by utils._compute_dots.
    verbose : boolean
        The verbosity of progress messages.
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

    def __init__(self, n_interpolate=0, consensus=0.1, thresh_func=None,
                 picks=None, thresh_method='bayesian_optimization',  dots=None,
                 verbose=True):
        """Init it."""
        if thresh_func is None:
            thresh_func = _compute_thresholds
        if not (0 <= consensus <= 1):
            raise ValueError('"consensus" must be between 0 and 1. '
                             'You gave me %s.' % consensus)
        self.consensus = consensus
        self.n_interpolate = n_interpolate
        self.thresh_func = thresh_func
        self.picks = picks
        self.verbose = verbose
        self.dots = dots

    def __repr__(self):
        """repr."""
        class_name = self.__class__.__name__
        params = dict(n_interpolate=self.n_interpolate,
                      consensus=self.consensus,
                      verbose=self.verbose, picks=self.picks)
        return '%s(%s)' % (class_name, _pprint(params,
                                               offset=len(class_name),),)

    def _vote_bad_epochs(self, epochs, picks):
        """Each channel votes for an epoch as good or bad.

        Parameters
        ----------
        epochs : instance of mne.Epochs
            The epochs object for which bad epochs must be found.
        picks : array-like
            The indices of the channels to consider.
        """
        labels = np.zeros((len(epochs), len(epochs.ch_names)))
        labels.fill(np.nan)
        bad_sensor_counts = np.zeros((len(epochs),))

        this_ch_names = [epochs.ch_names[p] for p in picks]
        deltas = np.ptp(epochs.get_data()[:, picks], axis=-1).T
        threshes = [self.threshes_[ch_name] for ch_name in this_ch_names]
        for ch_idx, (delta, thresh) in enumerate(zip(deltas, threshes)):
            bad_epochs_idx = np.where(delta > thresh)[0]
            labels[:, picks[ch_idx]] = 0
            labels[bad_epochs_idx, picks[ch_idx]] = 1

        bad_sensor_counts = np.sum(labels == 1, axis=1)
        return labels, bad_sensor_counts

    def _get_epochs_interpolation(self, epochs, labels,
                                  picks, n_interpolate,
                                  verbose=True):
        """Interpolate the bad epochs."""
        # 1: bad segment, # 2: interpolated
        assert labels.shape[0] == len(epochs)
        assert labels.shape[1] == len(epochs.ch_names)
        labels = labels.copy()
        non_picks = np.setdiff1d(range(epochs.info['nchan']), picks)
        for epoch_idx in range(len(epochs)):
            n_bads = labels[epoch_idx, picks].sum()
            if n_bads == 0:
                continue
            else:
                if n_bads <= n_interpolate:
                    interp_chs_mask = labels[epoch_idx] == 1
                else:
                    # get peak-to-peak for channels in that epoch
                    data = epochs[epoch_idx].get_data()[0]
                    peaks = np.ptp(data, axis=-1)
                    peaks[non_picks] = -np.inf
                    # find channels which are bad by rejection threshold
                    interp_chs_mask = labels[epoch_idx] == 1
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
        """Get the mask of bad epochs."""
        # XXX : avoid sorting twice
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

    def get_reject_log(self, epochs, threshes=None, picks=None):
        """Get rejection logs from epochs.

        .. note::
           If multiple channel types are present, reject_log.bad_epochs
           reflects the union of bad epochs across channel types.

        Parameters
        ----------
        epochs : instance of mne.Epochs
            The epochs from which to get the drop logs.
        picks : str | list | slice | None
            Channels to include. Slices and lists of integers will be
            interpreted as channel indices. In lists, channel *type* strings
            (e.g., ``['meg', 'eeg']``) will pick channels of those types,
            channel *name* strings (e.g., ``['MEG0111', 'MEG2623']`` will pick
            the given channels. Can also be the string values ``'all'`` to pick
            all channels, or ``'data'`` to pick data channels. None (default)
            will use the .picks attribute. Note that channels in
            ``info['bads']`` *will be included* if their names or indices are
            explicitly provided.

        Returns
        -------
        reject_log : instance of autoreject.RejectLog
            The rejection log.
        """
        picks = (self.picks_ if picks is None else
                 _handle_picks(epochs.info, picks))
        picks_by_type = _get_picks_by_type(picks=picks, info=epochs.info)
        assert len(picks_by_type) == 1
        ch_type, this_picks = picks_by_type[0]
        del picks

        labels, bad_sensor_counts = self._vote_bad_epochs(
            epochs, picks=this_picks)

        labels = self._get_epochs_interpolation(
            epochs, labels=labels, picks=this_picks,
            n_interpolate=self.n_interpolate_[ch_type])

        assert len(labels) == len(epochs)

        bad_epochs = self._get_bad_epochs(
            bad_sensor_counts, ch_type=ch_type, picks=this_picks)

        reject_log = RejectLog(labels=labels, bad_epochs=bad_epochs,
                               ch_names=epochs.ch_names)
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
        self : instance of _AutoReject
            The instance.
        """
        self.picks_ = _handle_picks(info=epochs.info, picks=self.picks)
        _check_data(epochs, picks=self.picks_, verbose=self.verbose,
                    ch_constraint='single_channel_type')

        picks_by_type = _get_picks_by_type(picks=self.picks_, info=epochs.info)
        assert len(picks_by_type) == 1
        ch_type, this_picks = picks_by_type[0]

        self.consensus_ = dict()
        self.n_interpolate_ = dict()
        self.n_interpolate_[ch_type] = self.n_interpolate
        self.consensus_[ch_type] = self.consensus

        self.threshes_ = self.thresh_func(
            epochs.copy(), dots=self.dots, picks=self.picks_,
            verbose=self.verbose)

        reject_log = self.get_reject_log(epochs=epochs, picks=self.picks_)

        epochs_copy = epochs.copy()
        interp_channels = _get_interp_chs(
            reject_log.labels, reject_log.ch_names, this_picks)

        # interpolate copy to compute the clean .mean_
        _interpolate_bad_epochs(
            epochs_copy, interp_channels=interp_channels,
            picks=self.picks_, verbose=self.verbose)
        self.mean_ = _slicemean(
            epochs_copy.get_data(),
            np.nonzero(np.invert(reject_log.bad_epochs))[0], axis=0)
        del epochs_copy  # I can't wait for garbage collection.
        return self

    def transform(self, epochs, return_log=False):
        """Fix and find the bad epochs.

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
        # this one knows how to handle picks.
        _apply_interp(reject_log, self, epochs_clean, self.threshes_,
                      self.picks_, self.dots, self.verbose)

        _apply_drop(reject_log, self, epochs_clean, self.threshes_,
                    self.picks_, self.verbose)

        if return_log:
            return epochs_clean, reject_log
        else:
            return epochs_clean


def _interpolate_bad_epochs(
        epochs, interp_channels, picks, dots=None, verbose=True):
    """Actually do the interpolation."""
    assert len(epochs) == len(interp_channels)
    pos = 2

    for epoch_idx, interp_chs in _pbar(
            list(enumerate(interp_channels)),
            desc='Repairing epochs',
            position=pos, leave=True, verbose=verbose):
        epoch = epochs[epoch_idx]
        epoch.info['bads'] = interp_chs
        interpolate_bads(epoch, dots=dots, picks=picks, reset_bads=True)
        epochs._data[epoch_idx] = epoch._data


def _run_local_reject_cv(epochs, thresh_func, picks_, n_interpolate, cv,
                         consensus, dots, verbose):
    n_folds = cv.get_n_splits()
    loss = np.zeros((len(consensus), len(n_interpolate),
                     n_folds))

    # The thresholds must be learnt from the entire data
    local_reject = _AutoReject(thresh_func=thresh_func,
                               verbose=verbose, picks=picks_,
                               dots=dots)
    local_reject.fit(epochs)

    assert len(local_reject.consensus_) == 1  # works with one ch_type
    ch_type = next(iter(local_reject.consensus_))

    labels, bad_sensor_counts = \
        local_reject._vote_bad_epochs(epochs, picks=picks_)
    desc = 'n_interp'

    for jdx, n_interp in enumerate(_pbar(n_interpolate, desc=desc,
                                         position=1, verbose=verbose)):
        # we can interpolate before doing cross-valida(tion
        # because interpolation is independent across trials.
        local_reject.n_interpolate_[ch_type] = n_interp
        labels = local_reject._get_epochs_interpolation(
            epochs, labels=labels, picks=picks_, n_interpolate=n_interp)

        interp_channels = _get_interp_chs(labels, epochs.ch_names, picks_)
        epochs_interp = epochs.copy()
        # for learning we need to go by channnel type, even for meg
        _interpolate_bad_epochs(
            epochs_interp, interp_channels=interp_channels,
            picks=picks_, dots=dots, verbose=verbose)

        # Hack to allow len(self.cv_.split(X)) as ProgressBar
        # assumes an iterable whereas self.cv_.split(X) is a
        # generator
        class CVSplits(object):
            def __init__(self, gen, length):
                self.gen = gen
                self.length = length

            def __len__(self):
                return self.length

            def __iter__(self):
                return self.gen

        X = epochs.get_data()[:, picks_]
        cv_splits = CVSplits(cv.split(X), n_folds)
        pbar = _pbar(cv_splits, desc='Fold',
                     position=3, verbose=verbose)

        for fold, (train, test) in enumerate(pbar):
            for idx, this_consensus in enumerate(consensus):
                # \kappa must be greater than \rho
                n_channels = len(picks_)
                if this_consensus * n_channels <= n_interp:
                    loss[idx, jdx, fold] = np.inf
                    continue

                local_reject.consensus_[ch_type] = this_consensus
                bad_epochs = local_reject._get_bad_epochs(
                    bad_sensor_counts[train], picks=picks_, ch_type=ch_type)

                good_epochs_idx = np.nonzero(np.invert(bad_epochs))[0]

                local_reject.mean_ = _slicemean(
                    epochs_interp[train].get_data()[:, picks_],
                    good_epochs_idx, axis=0)
                loss[idx, jdx, fold] = -local_reject.score(X[test])

    return local_reject, loss


class AutoReject(object):
    r"""Efficiently find n_interpolate and consensus.

    .. note::
       AutoReject by design supports multiple channels.
       If no picks are passed, separate solutions will be computed for each
       channel type and internally combined. This then readily supports
       cleaning unseen epochs from the different channel types used during fit.

    Parameters
    ----------
    n_interpolate : array | None
        The values to try for the number of channels for which to interpolate.
        This is :math:`\\rho`. If None, defaults to np.array([1, 4, 32])
    consensus : array | None
        The values to try for percentage of channels that must agree as a
        fraction of the total number of channels. This sets :math:`\\kappa/Q`.
        If None, defaults to `np.linspace(0, 1.0, 11)`
    cv : a scikit-learn cross-validation object
        Defaults to cv=10
    picks : str | list | slice | None
        Channels to include. Slices and lists of integers will be interpreted
        as channel indices. In lists, channel *type* strings (e.g.,
        ``['meg', 'eeg']``) will pick channels of those types, channel *name*
        strings (e.g., ``['MEG0111', 'MEG2623']`` will pick the given channels.
        Can also be the string values ``'all'`` to pick all channels, or
        ``'data'`` to pick data channels. None (default) will pick data
        channels {'meg', 'eeg'}, which will lead fitting and combining
        autoreject solutions across these channel types. Note that channels in
        ``info['bads']`` *will be included* if their names or indices are
        explicitly provided.
    thresh_method : str
        'bayesian_optimization' or 'random_search'
    n_jobs : int
        The number of jobs.
    random_state : int seed, RandomState instance, or None (default)
        The seed of the pseudo random number generator to use.
    verbose : boolean
        The verbosity of progress messages.
        If False, suppress all output messages.

    Attributes
    -----------
    local_reject_ : list
        The instances of _AutoReject for each channel type.
    threshes_ : dict
        The sensor-level thresholds with channel names as keys
        and the peak-to-peak thresholds as the values.
    loss_ : dict of array, shape (len(n_interpolate), len(consensus))
        The cross validation error for different parameter values.
    consensus_ : dict
        The estimated consensus per channel type.
    n_interpolate_ : dict
        The estimated n_interpolate per channel type.
    picks_ : array-like, shape (n_data_channels,)
        The data channels considered by autoreject. By default
        only data channels, not already marked as bads are considered.
    """

    def __init__(self, n_interpolate=None, consensus=None,
                 thresh_func=None, cv=10, picks=None,
                 thresh_method='bayesian_optimization',
                 n_jobs=1, random_state=None, verbose=True):
        """Init it."""
        self.n_interpolate = n_interpolate
        self.consensus = consensus
        self.thresh_method = thresh_method
        self.cv = cv
        self.verbose = verbose
        self.picks = picks
        self.n_jobs = n_jobs
        self.random_state = random_state

        if self.consensus is None:
            self.consensus = np.linspace(0, 1.0, 11)

    def __repr__(self):
        """repr."""
        class_name = self.__class__.__name__
        params = dict(n_interpolate=self.n_interpolate,
                      consensus=self.consensus,
                      cv=self.cv, verbose=self.verbose, picks=self.picks,
                      thresh_method=self.thresh_method,
                      random_state=self.random_state, n_jobs=self.n_jobs)
        return '%s(%s)' % (class_name, _pprint(params,
                                               offset=len(class_name),),)

    def __getstate__(self):
        """Get the state of autoreject as a dictionary."""
        state = dict()

        for param in _INIT_PARAMS:
            state[param] = getattr(self, param)
        for param in _FIT_PARAMS:
            if hasattr(self, param):
                state[param] = getattr(self, param)

        if hasattr(self, 'local_reject_'):
            state['local_reject_'] = dict()
            for ch_type in self.local_reject_:
                state['local_reject_'][ch_type] = dict()
                for param in _INIT_PARAMS[:4] + _FIT_PARAMS[:4]:
                    state['local_reject_'][ch_type][param] = \
                        getattr(self.local_reject_[ch_type], param)
        return state

    def __setstate__(self, state):
        """Set the state of autoreject."""
        for param in state.keys():
            if param == 'local_reject_':
                local_reject_ = dict()
                for ch_type in state['local_reject_']:
                    init_kwargs = {
                        key: state['local_reject_'][ch_type][key]
                        for key in _INIT_PARAMS[:4]
                    }
                    local_reject_[ch_type] = _AutoReject(**init_kwargs)
                    for key in _FIT_PARAMS[:4]:
                        setattr(local_reject_[ch_type], key,
                                state['local_reject_'][ch_type][key])
                self.local_reject_ = local_reject_
            elif param in _INIT_PARAMS + _FIT_PARAMS:
                setattr(self, param, state[param])

    def fit(self, epochs):
        """Fit the epochs on the AutoReject object.

        Parameters
        ----------
        epochs : instance of mne.Epochs
            The epochs object to be fit.

        Returns
        -------
        self : instance of AutoReject
            The instance.
        """
        self.picks_ = _handle_picks(info=epochs.info, picks=self.picks)
        _check_data(epochs, picks=self.picks_, verbose=self.verbose)
        self.cv_ = self.cv
        if isinstance(self.cv_, int):
            self.cv_ = KFold(n_splits=self.cv_)

        # XXX : maybe use an mne function in pick.py ?
        picks_by_type = _get_picks_by_type(info=epochs.info, picks=self.picks_)
        ch_types = [ch_type for ch_type, _ in picks_by_type]
        self.dots = None
        if 'mag' in ch_types or 'grad' in ch_types:
            meg_picks = pick_types(epochs.info, meg=True,
                                   eeg=False, exclude=[])
            this_info = mne.pick_info(epochs.info, meg_picks, copy=True)
            self.dots = _compute_dots(this_info)

        thresh_func = partial(_compute_thresholds, n_jobs=self.n_jobs,
                              method=self.thresh_method,
                              random_state=self.random_state,
                              dots=self.dots)

        if self.n_interpolate is None:
            if len(self.picks_) < 4:
                raise ValueError('Too few channels. autoreject is unlikely'
                                 ' to be effective')
            # XXX: dont interpolate all channels
            max_interp = min(len(self.picks_) - 1, 32)
            self.n_interpolate = np.array([1, 4, max_interp])

        self.n_interpolate_ = dict()  # rho
        self.consensus_ = dict()  # kappa
        self.threshes_ = dict()  # update
        self.loss_ = dict()
        self.local_reject_ = dict()

        for ch_type, this_picks in picks_by_type:
            if self.verbose is not False:
                print('Running autoreject on ch_type=%s' % ch_type)
            this_local_reject, this_loss = \
                _run_local_reject_cv(epochs, thresh_func, this_picks,
                                     self.n_interpolate, self.cv_,
                                     self.consensus, self.dots,
                                     self.verbose)
            self.threshes_.update(this_local_reject.threshes_)

            best_idx, best_jdx = \
                np.unravel_index(this_loss.mean(axis=-1).argmin(),
                                 this_loss.shape[:2])

            self.consensus_[ch_type] = self.consensus[best_idx]
            self.n_interpolate_[ch_type] = self.n_interpolate[best_jdx]
            self.loss_[ch_type] = this_loss

            # update local reject with best and store it
            this_local_reject.consensus_[ch_type] = self.consensus_[ch_type]
            this_local_reject.n_interpolate_[ch_type] = \
                self.n_interpolate_[ch_type]

            # needed for generating reject logs by channel
            self.local_reject_[ch_type] = this_local_reject

            if self.verbose is not False:
                print('\n\n\n\nEstimated consensus=%0.2f and n_interpolate=%d'
                      % (self.consensus_[ch_type],
                         self.n_interpolate_[ch_type]))
        return self

    def get_reject_log(self, epochs, picks=None):
        """Get rejection logs of epochs.

        .. note::
           If multiple channel types are present, reject_log['bad_epochs_idx']
           reflects the union of bad trials across channel types.

        Parameters
        ----------
        epochs : instance of mne.Epochs
            The epoched data for which the reject log is computed.
        picks : str | list | slice | None
            Channels to include. Slices and lists of integers will be
            interpreted as channel indices. In lists, channel *type* strings
            (e.g., ``['meg', 'eeg']``) will pick channels of those types,
            channel *name* strings (e.g., ``['MEG0111', 'MEG2623']`` will pick
            the given channels. Can also be the string values ``'all'`` to pick
            all channels, or ``'data'`` to pick data channels. None (default)
            will use the .picks attribute. Note that channels in
            ``info['bads']`` *will be included* if their names or indices are
            explicitly provided.

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
            reject_log.bad_epochs = np.logical_or(
                reject_log.bad_epochs, this_reject_log.bad_epochs)
            reject_log.ch_names = this_reject_log.ch_names
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
            The rejection log. Returned only if return_log is True.
        """
        # XXX : should be a check_fitted method
        if not hasattr(self, 'n_interpolate_'):
            raise ValueError('Please run autoreject.fit() method first')

        _check_data(epochs, picks=self.picks_, verbose=self.verbose)

        reject_log = self.get_reject_log(epochs)
        epochs_clean = epochs.copy()
        _apply_interp(reject_log, epochs_clean, self.threshes_,
                      self.picks_, self.dots, self.verbose)

        _apply_drop(reject_log, epochs_clean, self.threshes_, self.picks_,
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

    def save(self, fname, overwrite=False):
        """Save autoreject object with the HDF5 format.

        Parameters
        ----------
        fname : str
            The filename to save to. The filename must end
            in '.h5' or '.hdf5'.
        overwrite : bool
            If True, overwrite file if it already exists. Defaults to False.
        """
        fname = op.realpath(fname)
        if not overwrite and op.isfile(fname):
            raise ValueError('%s already exists. Please make overwrite=True'
                             'if you want to overwrite this file' % fname)

        write_hdf5(fname, self.__getstate__(), overwrite=overwrite,
                   title='autoreject')


def _check_fit(epochs, threshes_, picks_):
    msg = ('You are passing channels which were not present '
           'at fit-time. Please fit it again, this time '
           'correctly.')
    if not all(epochs.ch_names[pp] in threshes_
               for pp in picks_):
        raise ValueError(msg)


def _apply_interp(reject_log, epochs, threshes_, picks_, dots,
                  verbose):
    _check_fit(epochs, threshes_, picks_)
    interp_channels = _get_interp_chs(
        reject_log.labels, reject_log.ch_names, picks_)
    _interpolate_bad_epochs(
        epochs, interp_channels=interp_channels,
        picks=picks_, dots=dots, verbose=verbose)


def _apply_drop(reject_log, epochs, threshes_, picks_,
                verbose):
    _check_fit(epochs, threshes_, picks_)
    if np.any(reject_log.bad_epochs):
        epochs.drop(np.nonzero(reject_log.bad_epochs)[0],
                    reason='AUTOREJECT')
    elif verbose:
        print("No bad epochs were found for your data. Returning "
              "a copy of the data you wanted to clean. Interpolation "
              "may have been done.")


def _get_interp_chs(labels, ch_names, picks):
    """Convert labels to channel names.
    It returns a list of length n_epochs. Each entry contains
    the names of the channels to interpolate.

    labels is of shape n_epochs x n_channels
    and picks is the sublist of channels to consider.
    """
    interp_channels = list()
    assert labels.shape[1] == len(ch_names)
    assert labels.shape[1] > np.max(picks)
    idx_nan_in_row = np.where(np.any(~np.isnan(labels), axis=0))[0]
    np.testing.assert_array_equal(picks, idx_nan_in_row)
    for this_labels in labels:
        interp_idx = np.where(this_labels == 2)[0]
        interp_channels.append([ch_names[ii] for ii in interp_idx])
    return interp_channels


class RejectLog(object):
    """The Rejection Log.

    Parameters
    ----------
    bad_epochs : array-like, shape (n_epochs,)
        The boolean array with entries True for epochs that
        are marked as bad.
    labels : array, shape (n_epochs, n_channels)
        It contains integers that encode if a channel in a given
        epoch is good (value 0), bad (1), or bad and interpolated (2).
    ch_names : list of str
        The list of channels corresponding to the rows of the labels.
    """

    def __init__(self, bad_epochs, labels, ch_names):
        self.bad_epochs = bad_epochs
        self.labels = labels
        self.ch_names = ch_names
        assert len(bad_epochs) == labels.shape[0]
        assert len(ch_names) == labels.shape[1]

    def plot(self, orientation='vertical', show_names='auto', show=True,
             ax=None):
        """Plot an image of good, bad and interpolated channels for each epoch.

        Parameters
        ----------
        orientation : 'vertical' or 'horizontal'
            If `'vertical'` (default), will plot sensors on x-axis and epochs
            on y-axis. If `'horizontal'`, will plot epochs on x-axis and
            sensors on y-axis.
        show_names : 'auto' | int
            If 'auto' (default), show all channel names if fewer than 25
            entries. Otherwise it shows every 5 entries. If int, show every
            show_names entries.
        show : bool
            If True (default), display the figure immediately.
        ax : matplotlib.axes.Axes | None
            The axes to plot to. In ``None`` (default), create a new
            figure and axes.

        Returns
        -------
        figure : Instance of matplotlib.figure.Figure
            The figure object containing the plot.
        """
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        if show_names == 'auto':
            show_names = 1 if len(self.ch_names) < 25 else 5

        if ax is None:
            figure, ax = plt.subplots(figsize=(12, 6))
        else:
            figure = ax.get_figure()
        ax.grid(False)
        ch_names_ = self.ch_names[::show_names]

        image = self.labels.copy()
        image[image == 2] = 0.5  # move interp to 0.5
        # good, interp, bad
        legend_label = {0: 'good', 0.5: 'interpolated', 1: 'bad'}
        cmap = mpl.colors.ListedColormap(['lightgreen', 'blue', 'red'])
        if orientation == 'horizontal':
            img = ax.imshow(image.T, cmap=cmap,
                            vmin=0, vmax=1, interpolation='nearest')
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Channels')
            plt.setp(ax, yticks=range(0, self.labels.shape[1], show_names),
                     yticklabels=ch_names_)
            plt.setp(ax.get_yticklabels(), fontsize=8)
            # add red box around rejected epochs
            for idx in np.where(self.bad_epochs)[0]:
                ax.add_patch(patches.Rectangle(
                    (idx - 0.5, -0.5), 1, len(self.ch_names), linewidth=1,
                    edgecolor='r', facecolor='none'))

            # add legend
            handles = [patches.Patch(color=img.cmap(img.norm(i)), label=label)
                       for i, label in legend_label.items()]
            ax.legend(handles=handles, bbox_to_anchor=(0.7, 1.2), ncol=3,
                      borderaxespad=0.)

        elif orientation == 'vertical':
            img = ax.imshow(image, cmap=cmap,
                            vmin=0, vmax=1, interpolation='nearest')
            ax.set_xlabel('Channels')
            ax.set_ylabel('Epochs')
            plt.setp(ax, xticks=range(0, self.labels.shape[1], show_names),
                     xticklabels=ch_names_)
            plt.setp(ax.get_xticklabels(), fontsize=8, rotation='vertical')
            # add red box around rejected epochs
            for idx in np.where(self.bad_epochs)[0]:
                ax.add_patch(patches.Rectangle(
                    (-0.5, idx - 0.5), len(self.ch_names), 1, linewidth=1,
                    edgecolor='r', facecolor='none'))

            # add legend
            handles = [patches.Patch(color=img.cmap(img.norm(i)), label=label)
                       for i, label in legend_label.items()]
            ax.legend(handles=handles, bbox_to_anchor=(0.7, 1.2), ncol=3,
                      borderaxespad=0.)

        else:
            msg = """orientation can be only \
                  'horizontal' or 'vertical'. Got %s""" % orientation
            raise ValueError(msg)

        # XXX to be fixed
        plt.setp(ax.get_yticklabels(), rotation=0)
        plt.setp(ax.get_xticklabels(), rotation=90)
        ax.tick_params(axis=u'both', which=u'both', length=0)
        figure.tight_layout()
        if show:
            plt.show()
        return figure

    def plot_epochs(self, epochs, scalings=None, title=''):
        """Plot interpolated and dropped epochs.

        Parameters
        ----------
        epochs : instance of Epochs
            The epochs.
        scalings : dict | None
            Scaling factors for the traces. If None, defaults to::

                dict(mag=1e-12, grad=4e-11, eeg=20e-6, eog=150e-6, ecg=5e-4,
                     emg=1e-3, ref_meg=1e-12, misc=1e-3, stim=1,
                     resp=1, chpi=1e-4, whitened=1e2)
        title : str
            The title to display.

        Returns
        -------
        fig : Instance of matplotlib.figure.Figure
            Epochs traces.
        """
        labels = self.labels
        n_epochs, n_channels = labels.shape

        if not labels.shape[0] == len(epochs.events):
            raise ValueError('The number of epochs should match the number of'
                             'epochs *before* autoreject. Please provide'
                             'the epochs object before running autoreject')
        if not labels.shape[1] == len(epochs.ch_names):
            raise ValueError('The number of channels should match the number'
                             ' of channels before running autoreject.')
        bad_epochs_idx = np.where(self.bad_epochs)[0]
        if len(bad_epochs_idx) > 0 and \
                bad_epochs_idx.max() > len(epochs.events):
            raise ValueError('You had a bad_epoch with index'
                             '%d but there are only %d epochs. Make sure'
                             ' to provide the epochs *before* running'
                             'autoreject.'
                             % (bad_epochs_idx.max(),
                                len(epochs.events)))

        color_map = {0: 'k', 1: 'r', 2: (0.6, 0.6, 0.6, 1.0)}
        epoch_colors = list()
        for epoch_idx, label_epoch in enumerate(labels):
            if self.bad_epochs[epoch_idx]:
                epoch_color = ['r'] * n_channels
                epoch_colors.append(epoch_color)
                continue
            epoch_color = list()
            for this_label in label_epoch:
                if not np.isnan(this_label):
                    epoch_color.append(color_map[this_label])
                else:
                    epoch_color.append(None)
            epoch_colors.append(epoch_color)

        return plot_mne_epochs(
            epochs=epochs,
            epoch_colors=epoch_colors, scalings=scalings,
            title='')
