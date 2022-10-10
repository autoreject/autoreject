"""RANSAC code

The code is adopted from the PREP pipeline written in MATLAB:
https://github.com/VisLab/EEG-Clean-Tools. This implementation
also works for MEG data.
"""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Simon Kern

import numpy as np
from joblib import Parallel, delayed

import mne
from mne.channels.interpolation import _make_interpolation_matrix
from mne.parallel import parallel_func
from mne.utils import check_random_state

from .utils import _pbar, _handle_picks
from .utils import _check_data, _get_channel_type


class Ransac(object):
    """RANSAC algorithm to find bad sensors and repair them.

    Implements RAndom SAmple Consensus (RANSAC) method to detect bad sensors.

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
    random_state : int | np.random.RandomState | None
        The seed of the pseudo random number generator to use.
        Defaults to 435656.
    picks : str | list | slice | None
        Channels to include. Slices and lists of integers will be
        interpreted as channel indices. In lists, channel *name* strings
        (e.g., ``['MEG0111', 'MEG2623']``) will pick the given channels.
        None (default) will pick data channels {'meg', 'eeg'}. Note that
        channels in ``info['bads']`` *will be included* if their names or
        indices are explicitly provided.
    verbose : bool
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

    def __init__(self, n_resample=50, min_channels=0.25, min_corr=0.75,
                 unbroken_time=0.4, n_jobs=1,
                 random_state=435656, picks=None,
                 verbose=True):
        """Initialize Ransac object."""
        self.n_resample = n_resample
        self.min_channels = min_channels
        self.min_corr = min_corr
        self.unbroken_time = unbroken_time
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.picks = picks

    def _iterate_epochs(self, epochs, idxs):
        n_channels = len(self.picks)
        corrs = np.zeros((len(idxs), n_channels))
        for i, idx in enumerate(_pbar(idxs, desc='Iterating epochs',
                                      verbose=self.verbose)):
            data = epochs.get_data()[idx, self.picks]
            corrs[i, :] = self._compute_correlations(data)
        return corrs

    def _get_random_subsets(self, info):
        """ Get random channels"""
        # have to set the seed here, as here the only part with randomization
        # occurs. However, all subsets are precomputed outside of Parallel,
        # therefore, we can simply compute them once
        rng = check_random_state(self.random_state)
        picked_info = mne.io.pick.pick_info(info, self.picks)
        n_channels = len(picked_info['ch_names'])

        # number of channels to interpolate from
        n_samples = int(np.round(self.min_channels * n_channels))

        # get picks for resamples, but ignore channels marked as bad
        bad_chs = info['bads']
        ch_list = [ch for ch in picked_info['ch_names'] if ch not in bad_chs]
        assert len(ch_list) >= n_samples, 'too many channels marked as bad,'\
                                          'cannot perform interpolation with'\
                                          f'min_channels={self.min_channels}'

        # randomly sample subsets of good channels
        ch_subsets = list()
        for idx in range(self.n_resample):
            picks = rng.choice(ch_list, size=n_samples, replace=False)
            picks = [str(p) for p in picks]  # convert from str-array to string
            ch_subsets.append(picks)

        return ch_subsets

    def _get_mappings(self, inst, ch_subsets):
        from .utils import _fast_map_meg_channels

        picked_info = mne.io.pick.pick_info(inst.info, self.picks)
        pos = np.array([ch['loc'][:3] for ch in picked_info['chs']])
        ch_names = picked_info['ch_names']
        n_channels = len(ch_names)
        pick_to = range(n_channels)
        mappings = list()
        # Try different channel subsets
        for subset in _pbar(ch_subsets, desc='interpolating channels',
                            verbose=self.verbose):
            # don't do the following as it will sort the channels!
            # pick_from = pick_channels(ch_names, ch_subsets[idx])
            pick_from = np.array([ch_names.index(name)
                                  for name in subset])
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
        """Perform RANSAC on the given epochs.

        Steps:

        #. Interpolate all channels from a subset of channels
           (fraction denoted as `min_channels`), repeat `n_resample` times.
        #. See if correlation of interpolated channels to original channel
           is above 75% per epoch (`min_corr`)
        #. If more than `unbroken_time` fraction of epochs have a lower
           correlation than that, add channel to ``self.bad_chs_``

        Parameters
        ----------
        epochs : mne.Epochs
            An Epochs object with data to perform RANSAC on

        Returns
        -------
        self : Ransac
            The updated instance with the list of bad channels accessible by
            ``self.bad_chs_``
        """
        self.picks = _handle_picks(info=epochs.info, picks=self.picks)
        _check_data(epochs, picks=self.picks,
                    ch_constraint='single_channel_type', verbose=self.verbose)
        self.ch_type = _get_channel_type(epochs, self.picks)
        n_epochs = len(epochs)

        _, _, n_jobs = parallel_func(self._get_mappings, self.n_jobs)
        parallel = Parallel(n_jobs, verbose=10 if self.verbose else 0)

        # create `n_resample` different random subsamples of channels,
        # with each subsample set containing `min_channels` amount of
        # random channels from the list of all channels.
        self.ch_subsets_ = self._get_random_subsets(epochs.info)

        # compute mappings with possibility of parallelization
        # max n_resample splits possible
        n_splits = min(n_jobs, self.n_resample)
        ch_subsets_split = np.array_split(self.ch_subsets_, n_splits)
        delayed_func = delayed(self._get_mappings)
        # no random seed needs to be supplied to get_mappings, as there is
        # no random subsampling happening here
        mappings = parallel(delayed_func(epochs, ch_subset) for ch_subset
                            in ch_subsets_split)
        self.mappings_ = np.concatenate(mappings)

        # compute correlations with possibility of parallelization
        delayed_func = delayed(self._iterate_epochs)
        n_splits = min(n_jobs, n_epochs)  # max n_epochs splits possible
        epoch_idxs_splits = np.array_split(np.arange(n_epochs), n_splits)
        corrs = parallel(delayed_func(epochs, idxs) for idxs
                         in epoch_idxs_splits)
        self.corr_ = np.concatenate(corrs)

        if self.verbose is not False:
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
            self.bad_chs_ = list()
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
