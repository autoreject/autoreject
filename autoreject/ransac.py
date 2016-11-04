"""RANSAC code

The code is adopted from the PREP pipeline written in MATLAB:
https://github.com/VisLab/EEG-Clean-Tools. This implementation
also works for MEG data.

"""

import numpy as np
from sklearn.externals.joblib import Parallel, delayed

from mne.channels.interpolation import _make_interpolation_matrix
from mne.parallel import check_n_jobs

from .utils import _pbar


def _randsample(X, num):
    """Generate random subsamples (as done in PREP code)"""
    Y = []
    for k in range(num):
        pick = int(round(1 + (len(X) - 1) * np.random.random())) - 1
        Y.append(X[pick])
        del X[pick]
    return Y


def _iterate_epochs(ransac, epochs, idxs, verbose):
    n_channels = len(epochs.info['ch_names'])
    corrs = np.zeros((len(idxs), n_channels))
    for idx, _ in enumerate(_pbar(idxs, desc='Iterating epochs',
                            verbose=verbose)):
        ransac.fit(epochs[idx]).predict(epochs[idx])
        corrs[idx, :] = ransac.corr_
    return corrs


class Ransac(object):

    def __init__(self, n_resample=50, min_channels=0.25, min_corr=0.75,
                 unbroken_time=0.4, ch_type='eeg', n_jobs=1,
                 verbose='progressbar'):
        """Implements RAndom SAmple Consensus (RANSAC) method to detect bad sensors.

        Parameters
        ----------
        n_resample : int
            Number of times the sensors are resampled.
        min_channels : float
            Fraction of sensors for robust reconstruction.
        min_corr : float
            Cut-off correlation for abnormal wrt neighbours.
        unbroken_time : float
            Cut-off fraction of time sensor can have poor RANSAC
            predictability.
        ch_type : str
            'meg' | 'eeg'
        n_jobs : int
            Number of parallel jobs.
        verbose : 'tqdm', 'tqdm_notebook', 'progressbar' or False
            The verbosity of progress messages.
            If `'progressbar'`, use `mne.utils.ProgressBar`.
            If `'tqdm'`, use `tqdm.tqdm`.
            If `'tqdm_notebook'`, use `tqdm.tqdm_notebook`.
            If False, suppress all output messages.

        Notes
        -----
        The window_size is automatically set to the epoch length.

        References
        ----------
        [1] Bigdely-Shamlo, Nima, et al.
            "The PREP pipeline: standardized preprocessing for large-scale EEG
            analysis." Frontiers in neuroinformatics 9 (2015).
        [2] Mainak Jas, Denis Engemann, Yousra Bekhti, Federico Raimondo, and
            Alexandre Gramfort, "Autoreject: Automated artifact rejection for
            MEG and EEG." arXiv preprint arXiv:1612.08194, 2016.
        """
        self.n_resample = n_resample
        self.min_channels = min_channels
        self.min_corr = min_corr
        self.unbroken_time = unbroken_time
        self.ch_type = ch_type
        self.n_jobs = n_jobs
        self.verbose = verbose

    def _get_random_subsets(self, info):
        """ Get random channels"""
        # have to set the seed here
        np.random.seed(435656)
        n_channels = len(info['ch_names'])

        # number of channels to interpolate from
        n_samples = int(np.round(self.min_channels * n_channels))

        # get picks for resamples
        picks = []
        for idx in range(self.n_resample):
            pick = _randsample(range(n_channels), n_samples)
            picks.append(pick)

        # get channel subsets as lists
        ch_subsets = []
        for pick in picks:
            ch_subsets.append([info['ch_names'][p] for p in pick])

        return ch_subsets

    def _get_mappings(self, inst):
        from .utils import _fast_map_meg_channels

        ch_subsets = self.ch_subsets_
        pos = np.array([ch['loc'][:3] for ch in inst.info['chs']])
        ch_names = inst.info['ch_names']
        n_channels = len(ch_names)
        pick_to = range(n_channels)
        mappings = []
        # Try different channel subsets
        for idx in range(len(ch_subsets)):
            # don't do the following as it will sort the channels!
            # pick_from = pick_channels(ch_names, ch_subsets[idx])
            pick_from = np.array([ch_names.index(name)
                                 for name in ch_subsets[idx]])
            mapping = np.zeros((n_channels, n_channels))
            if self.ch_type == 'meg':
                mapping[:, pick_from] = _fast_map_meg_channels(inst, pick_from,
                                                               pick_to)
            elif self.ch_type == 'eeg':
                mapping[:, pick_from] = _make_interpolation_matrix(
                    pos[pick_from], pos[pick_to], alpha=1e-5)
            mappings.append(mapping)
        mappings = np.concatenate(mappings)
        return mappings

    def _compute_correlations(self, inst):
        """Compute correlation between prediction and real data."""
        mappings = self.mappings_
        n_epochs, n_channels, n_times = inst.get_data().shape

        # get the predictions
        y_pred = inst.get_data()[0].T.dot(mappings.T)
        y_pred = y_pred.reshape((n_times, n_channels,
                                 self.n_resample), order='F')
        # pool them using median
        # XXX: weird that original implementation sorts and takes middle value.
        # Isn't really the median if n_resample even
        y_pred = np.median(y_pred, axis=-1)
        # compute correlation
        num = np.sum(inst.get_data()[0].T * y_pred, axis=0)
        denom = (np.sqrt(np.sum(inst.get_data()[0].T ** 2, axis=0)) *
                 np.sqrt(np.sum(y_pred ** 2, axis=0)))
        corr = num / denom
        return corr

    def fit(self, epochs):
        self.ch_subsets_ = self._get_random_subsets(epochs.info)
        self.mappings_ = self._get_mappings(epochs)
        return self

    def predict(self, epochs):
        self.corr_ = self._compute_correlations(epochs)

    def fit_predict(self, epochs):
        n_epochs = len(epochs)
        n_jobs = check_n_jobs(self.n_jobs)
        parallel = Parallel(n_jobs, verbose=10)
        my_iterator = delayed(_iterate_epochs)
        if self.verbose is not False and self.n_jobs > 1:
            print('Iterating epochs ...')
        verbose = False if self.n_jobs > 1 else self.verbose
        corrs = parallel(my_iterator(self, epochs, idxs, verbose)
                         for idxs in np.array_split(np.arange(n_epochs),
                         n_jobs))
        self.corr_ = np.concatenate(corrs)
        if self.verbose is not False and self.n_jobs > 1:
            print('[Done]')
        # compute how many windows is a sensor RANSAC-bad
        self.bad_log = np.zeros_like(self.corr_)
        self.bad_log[self.corr_ < self.min_corr] = 1
        bad_log = self.bad_log.sum(axis=0)

        bad_idx = np.where(bad_log > self.unbroken_time * n_epochs)[0]
        if len(bad_idx) > 0:
            self.bad_chs_ = [epochs.info['ch_names'][p] for p in bad_idx]
        else:
            self.bad_chs_ = []
