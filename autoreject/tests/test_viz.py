# Author: Mainak Jas <mainak.jas@telecom-paristech.fr>
# License: BSD (3-clause)

import numpy as np

import mne
from mne.datasets import sample
from mne import io

from autoreject.viz import plot_epochs

from nose.tools import assert_raises

import matplotlib
matplotlib.use('Agg')

data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
raw = io.read_raw_fif(raw_fname, preload=False)
raw.crop(0, 15)
raw.info['projs'] = list()


def test_viz():
    """Test viz."""
    import matplotlib.pyplot as plt

    events = mne.find_events(raw)
    picks = mne.pick_channels(raw.info['ch_names'],
                              ['MEG 2443', 'MEG 2442', 'MEG 2441'])
    epochs = mne.Epochs(raw, events, picks=picks, baseline=(None, 0),
                        reject=None, preload=True)
    bad_epochs_idx = [0, 1, 3]
    n_epochs, n_channels, _ = epochs.get_data().shape
    fix_log = np.zeros((n_epochs, n_channels))

    plot_epochs(epochs, bad_epochs_idx=bad_epochs_idx, fix_log=fix_log)
    plot_epochs(epochs, bad_epochs_idx=bad_epochs_idx)
    plot_epochs(epochs, fix_log=fix_log)
    assert_raises(ValueError, plot_epochs, epochs[:2],
                  bad_epochs_idx=bad_epochs_idx, fix_log=fix_log)
    plt.close('all')
