"""Automated rejection and repair of trials in M/EEG."""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Denis A. Engemann <denis.engemann@gmail.com>

import numpy as np
from scipy.stats.distributions import uniform

import mne
from mne.utils import ProgressBar

from sklearn.base import BaseEstimator
from sklearn.grid_search import RandomizedSearchCV
from sklearn.cross_validation import KFold

from joblib import Memory
from pandas import DataFrame

from .utils import clean_by_interp, interpolate_bads

mem = Memory(cachedir='cachedir')
mem.clear()


def _check_data(epochs):
    if not isinstance(epochs, mne.epochs._BaseEpochs):
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


def validation_curve(estimator, epochs, y, param_name, param_range, cv=None,
                     n_jobs=1):
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
    """
    from sklearn.learning_curve import validation_curve
    if not isinstance(estimator, GlobalAutoReject):
        msg = 'No guarantee that it will work on this estimator.'
        raise NotImplementedError(msg)
    if not isinstance(epochs, mne.epochs._BaseEpochs):
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
        self.mean_ = np.mean(X[keep], axis=0)
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
        self.mean_ = np.mean(X[keep], axis=0)
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


def _compute_thresh(this_data, thresh_range, cv=10):
    """ Compute the rejection threshold for one channel.

    Parameters
    ----------
    this_data: array (n_epochs, n_times)
        Data for one channel.
    cv : iterator
        Iterator for cross-validation.
    """
    est = _ChannelAutoReject()

    param_dist = dict(thresh=uniform(thresh_range[0],
                                     thresh_range[1]))
    rs = RandomizedSearchCV(est,  # XXX : is random really better than grid?
                            param_distributions=param_dist,
                            n_iter=20, cv=cv)
    rs.fit(this_data)
    best_thresh = rs.best_estimator_.thresh

    return best_thresh


def compute_thresholds(epochs, thresh_range=None):
    """Compute thresholds for each channel.

    Parameters
    ----------
    epochs : instance of mne.Epochs
        The epochs objects whose thresholds must be computed.
    thresh_range : dict
        Possible keys are 'eeg', 'grad' and 'mag'. Each entry is a tuple
        of the form (low, high) which specifies the range to try.

    Examples
    --------
    For example, we can compute the channel-level thresholds for all the
    EEG sensors this way:
        >>> compute_thresholds(epochs, range=dict(eeg=(20e-7, 400e-6)))
    """
    if thresh_range is None:
        thresh_range = dict(eeg=(20e-7, 400e-6),
                            grad=(400e-13, 20000e-13),
                            mag=(400e-15, 20000e-15))
    if not all(key in ['eeg', 'grad', 'mag'] for key in thresh_range.keys()):
        raise KeyError('Invalid key provided to thresh_range')

    ch_types = [ch_type for ch_type in ('eeg', 'meg')
                if ch_type in epochs]
    epochs_interp = clean_by_interp(epochs)
    data = np.concatenate((epochs.get_data(), epochs_interp.get_data()),
                          axis=0)
    threshes = dict()
    picks_grad, picks_mag = list(), list()
    for ch_type in ch_types:
        print('Compute optimal thresholds for %s' % ch_type)
        picks = _pick_exclusive_channels(epochs.info, ch_type)
        if ch_type == 'meg':
            picks_grad = _pick_exclusive_channels(epochs.info, 'grad')
            picks_mag = _pick_exclusive_channels(epochs.info, 'mag')
        np.random.seed(42)  # has no effect unless shuffle=True is used
        cv = KFold(data.shape[0], 10, random_state=42)
        threshes[ch_type] = []
        for ii, pick in enumerate(picks):
            if pick in picks_grad:
                thresh_type = 'grad'
            elif pick in picks_mag:
                thresh_type = 'mag'
            else:
                thresh_type = 'eeg'
            thresh = _compute_thresh(data[:, pick], cv=cv,
                                     thresh_range=thresh_range[thresh_type])
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
    """
    def __init__(self, thresh_func=None, consensus_perc=0.1, n_interpolate=0):

        # TODO: must be able to try different consensus percs
        # with pretrained thresh
        if thresh_func is None:
            thresh_func = compute_thresholds
        if not (0 <= consensus_perc <= 1):
            raise ValueError('"consensus_perc" must be between 0 and 1. '
                             'You gave me %s.' % consensus_perc)
        self.consensus_perc = consensus_perc
        self.n_interpolate = n_interpolate
        self.thresh_func = mem.cache(thresh_func)

    @property
    def bad_segments(self):
        return self._drop_log

    def fit(self, epochs):
        """Compute the thresholds.

        Parameters
        ----------
        epochs : instance of mne.Epochs
            The epochs object from which the channel-level thresholds are
            estimated.
        """
        self.threshes_ = self.thresh_func(epochs)
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
            self._interpolate_bad_epochs(epochs, ch_type=ch_type)

        bad_epochs_idx = self._get_bad_epochs()
        self.bad_epochs_idx = np.sort(bad_epochs_idx)
        self.good_epochs_idx = np.setdiff1d(np.arange(len(epochs)),
                                            bad_epochs_idx)
        self.mean_ = epochs[self.good_epochs_idx].get_data().mean(axis=0)
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
            print('No bad epochs dropped by consensus.')

        return bad_epochs_idx

    def _interpolate_bad_epochs(self, epochs, ch_type):
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
        pbar = ProgressBar(len(epochs) - 1, mesg='Repairing epochs: ',
                           spinner=True)
        # TODO: raise error if preload is not True
        for epoch_idx in range(len(epochs)):
            pbar.update(epoch_idx + 1)
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
        The values of :math:`\\rho` to try. If None, defaults
        to `np.linspace(0, 1.0, 11)`
    consensus_percs : array | None
        The values of :math:`\kappa` to try. If None, defaults to
        np.array([1, 4, 32])
    thresh_func : callable | None
        Function which returns the channel-level thresholds. If None,
        defaults to :func:`autoreject.compute_thresholds`.
    cv : a scikit-learn cross-validation object
        Defaults to cv=10

    Returns
    -------
    local_ar : instance of LocalAutoReject
        The fitted LocalAutoReject object.
    """

    def __init__(self, n_interpolates=None, consensus_percs=None,
                 thresh_func=None, cv=None):
        self.n_interpolates = n_interpolates
        self.consensus_percs = consensus_percs
        self.thresh_func = thresh_func
        self.cv = cv

    @property
    def bad_segments(self):
        return self._local_reject._drop_log

    def fit(self, epochs):
        """Fit the epochs on the LocalAutoReject object.

        Parameters
        ----------
        epochs : instance of mne.Epochs
            The epochs object to be fit.
        """
        _check_data(epochs)
        if self.cv is None:
            self.cv = KFold(len(epochs), n_folds=10, random_state=42)
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

        local_reject = LocalAutoReject(thresh_func=self.thresh_func)

        # The thresholds must be learnt from the entire data
        local_reject.fit(epochs)

        local_reject._vote_epochs(epochs)
        bad_epoch_counts = local_reject.bad_epoch_counts.copy()
        for jdx, n_interp in enumerate(self.n_interpolates):
            # we can interpolate before doing cross-validation
            # because interpolation is independent across trials.
            local_reject.n_interpolate = n_interp
            ch_types = [ch_type for ch_type in ('eeg', 'meg') if
                        ch_type in epochs]
            epochs_interp = epochs.copy()
            for ch_type in ch_types:
                local_reject._interpolate_bad_epochs(epochs_interp,
                                                     ch_type=ch_type)
            for fold, (train, test) in enumerate(self.cv):
                for idx, consensus_perc in enumerate(self.consensus_percs):
                    print('[Val fold %d] Trying consensus perc %0.2f,'
                          'n_interp %d' % (fold + 1, consensus_perc, n_interp))
                    local_reject.consensus_perc = consensus_perc
                    local_reject.bad_epoch_counts = bad_epoch_counts[train]

                    bad_epochs_idx = local_reject._get_bad_epochs()
                    local_reject.bad_epochs_idx = np.sort(bad_epochs_idx)
                    n_train = len(epochs[train])
                    good_epochs_idx = np.setdiff1d(np.arange(n_train),
                                                   bad_epochs_idx)
                    epochs_train = epochs_interp[train][good_epochs_idx]
                    local_reject.mean_ = epochs_train.get_data().mean(axis=0)
                    X = epochs[test].get_data()
                    loss[idx, jdx, fold] = -local_reject.score(X)

        best_idx, best_jdx = np.unravel_index(loss.mean(axis=-1).argmin(),
                                              loss.shape[:2])
        consensus_perc = self.consensus_percs[best_idx]
        n_interpolate = self.n_interpolates[best_jdx]
        self.consensus_perc_ = consensus_perc
        self.n_interpolate_ = n_interpolate
        local_reject = LocalAutoReject(compute_thresholds, consensus_perc,
                                       n_interpolate=n_interpolate)
        local_reject.fit(epochs)
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
