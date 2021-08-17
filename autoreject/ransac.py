"""RANSAC code

The code is adopted from the PREP pipeline written in MATLAB:
https://github.com/VisLab/EEG-Clean-Tools. This implementation
also works for MEG data.
"""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>

import numpy as np
from joblib import Parallel, delayed

import mne
from mne.channels.interpolation import _make_interpolation_matrix
from mne.parallel import check_n_jobs
from mne.utils import check_random_state

from .utils import _pbar, _handle_picks
from .utils import _check_data


def _iterate_epochs(ransac, epochs, idxs, ch_subset, verbose):
    ransac.mappings_ = ransac._get_mappings(epochs, ch_subset)
    n_channels = len(ransac.picks)
    corrs = np.zeros((len(idxs), n_channels))
    for idx, _ in enumerate(_pbar(idxs, desc='Iterating epochs',
                                  verbose=verbose)):
        ransac.corr_ = ransac._compute_correlations(
            epochs.get_data()[idx, ransac.picks])
        corrs[idx, :] = ransac.corr_
    return corrs


def _get_channel_type(epochs, picks):
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


class Ransac(object):
    """RANSAC algorithm to find bad sensors and repair them."""

    def __init__(self, n_resample=50, min_channels=0.25, min_corr=0.75,
                 unbroken_time=0.4, n_jobs=1,
                 random_state=435656, picks=None,
                 verbose=True):
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
        n_jobs : int
            Number of parallel jobs.
        random_state : None | int | np.random.RandomState
            The seed of the pseudo random number generator to use.
        picks : str | list | slice | None
            Channels to include. Slices and lists of integers will be
            interpreted as channel indices. In lists, channel *name* strings
            (e.g., ``['MEG0111', 'MEG2623']``) will pick the given channels.
            None (default) will pick data channels {'meg', 'eeg'}. Note that
            channels in ``info['bads']`` *will be included* if their names or
            indices are explicitly provided.
        verbose : boolean
            The verbosity of progress messages.
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
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.picks = picks

    def _get_random_subsets(self, info, random_state):
        """ Get random channels"""
        # have to set the seed here
        rng = check_random_state(random_state)
        picked_info = mne.io.pick.pick_info(info, self.picks)
        n_channels = len(picked_info['ch_names'])

        # number of channels to interpolate from
        n_samples = int(np.round(self.min_channels * n_channels))

        # get picks for resamples
        picks = []
        for idx in range(self.n_resample):
            pick = rng.permutation(n_channels)[:n_samples].copy()
            picks.append(pick)

        # get channel subsets as lists
        ch_subsets = []
        for pick in picks:
            ch_subsets.append([picked_info['ch_names'][p] for p in pick])

        return ch_subsets

    def _get_mappings(self, inst, ch_subsets):
        from .utils import _fast_map_meg_channels

        picked_info = mne.io.pick.pick_info(inst.info, self.picks)
        pos = np.array([ch['loc'][:3] for ch in picked_info['chs']])
        ch_names = picked_info['ch_names']
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
                mapping[:, pick_from] = _fast_map_meg_channels(
                    picked_info.copy(), pick_from, pick_to)
            elif self.ch_type == 'eeg':
                mapping[:, pick_from] = _make_interpolation_matrix(
                    pos[pick_from], pos[pick_to], alpha=1e-5)
            mappings.append(mapping)
        mappings = np.concatenate(mappings)
        return mappings

    def _compute_correlations(self, data):
        """Compute correlation between prediction and real data."""
        mappings = self.mappings_
        n_channels, n_times = data.shape

        # get the predictions
        y_pred = data.T.dot(mappings.T)
        y_pred = y_pred.reshape((n_times, len(self.picks),
                                 self.n_resample), order='F')
        # pool them using median
        # XXX: weird that original implementation sorts and takes middle value.
        # Isn't really the median if n_resample even
        y_pred = np.median(y_pred, axis=-1)
        # compute correlation
        num = np.sum(data.T * y_pred, axis=0)
        denom = (np.sqrt(np.sum(data.T ** 2, axis=0)) *
                 np.sqrt(np.sum(y_pred ** 2, axis=0)))

        corr = num / denom
        return corr

    def fit(self, epochs):
        self.picks = _handle_picks(info=epochs.info, picks=self.picks)
        _check_data(epochs, picks=self.picks,
                    ch_constraint='single_channel_type', verbose=self.verbose)
        self.ch_type = _get_channel_type(epochs, self.picks)
        n_epochs = len(epochs)

        n_jobs = check_n_jobs(self.n_jobs)
        parallel = Parallel(n_jobs, verbose=10)
        my_iterator = delayed(_iterate_epochs)
        if self.verbose is not False and self.n_jobs > 1:
            print('Iterating epochs ...')
        verbose = False if self.n_jobs > 1 else self.verbose
        rng = check_random_state(self.random_state)
        base_random_state = rng.randint(np.iinfo(np.int16).max)
        self.ch_subsets_ = [self._get_random_subsets(
                            epochs.info, base_random_state + random_state)
                            for random_state in np.arange(0, n_epochs, n_jobs)]
        epoch_idxs = np.array_split(np.arange(n_epochs), n_jobs)
        corrs = parallel(my_iterator(self, epochs, idxs, chs, verbose)
                         for idxs, chs in zip(epoch_idxs, self.ch_subsets_))
        self.corr_ = np.concatenate(corrs)
        if self.verbose is not False and self.n_jobs > 1:
            print('[Done]')

        # compute how many windows is a sensor RANSAC-bad
        self.bad_log = np.zeros_like(self.corr_)
        self.bad_log[self.corr_ < self.min_corr] = 1
        bad_log = self.bad_log.sum(axis=0)

        bad_idx = np.where(bad_log > self.unbroken_time * n_epochs)[0]
        if len(bad_idx) > 0:
            self.bad_chs_ = [
                epochs.info['ch_names'][self.picks[p]] for p in bad_idx]
        else:
            self.bad_chs_ = []
        return self

    def transform(self, epochs):
        epochs = epochs.copy()
        _check_data(epochs, picks=self.picks,
                    ch_constraint='single_channel_type', verbose=self.verbose)
        epochs.info['bads'] = self.bad_chs_
        epochs.interpolate_bads(reset_bads=True)
        return epochs

    def fit_transform(self, epochs):
        return self.fit(epochs).transform(epochs)
