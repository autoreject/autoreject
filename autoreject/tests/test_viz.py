# Author: Mainak Jas <mainak.jas@telecom-paristech.fr>
# License: BSD-3-Clause

import numpy as np
import pytest

import mne
from mne.datasets import testing
from mne import io

import autoreject
from autoreject.utils import set_matplotlib_defaults

data_path = testing.data_path(download=False)
raw_fname = data_path / 'MEG' / 'sample' / 'sample_audvis_trunc_raw.fif'


@testing.requires_testing_data
def test_viz(browser_backend):
    """Test viz."""
    import matplotlib.pyplot as plt
    raw = io.read_raw_fif(raw_fname, preload=False)
    raw.crop(0, 15)
    raw.del_proj()

    set_matplotlib_defaults(plt)

    events = mne.find_events(raw)
    picks = mne.pick_channels(raw.info['ch_names'],
                              ['MEG 2443', 'MEG 2442', 'MEG 2441'])
    epochs = mne.Epochs(raw, events, picks=picks, baseline=(None, 0),
                        reject=None, preload=True,
                        event_id={'1': 1, '2': 2, '3': 3, '4': 4})
    bad_epochs_idx = [0, 1, 3]
    n_epochs, n_channels, _ = epochs.get_data().shape
    bad_epochs = np.zeros(n_epochs, dtype=bool)
    bad_epochs[bad_epochs_idx] = True
    assert len(bad_epochs) == 15

    labels = np.zeros((n_epochs, n_channels))
    labels[2, 0] = np.nan  # one good epoch has a nan label
    reject_log = autoreject.RejectLog(bad_epochs, labels, epochs.ch_names)
    reject_log.plot_epochs(epochs)
    reject_log.plot()
    reject_log.plot(orientation='horizontal')
    pytest.raises(ValueError, reject_log.plot_epochs, epochs[:2])
    pytest.raises(ValueError, reject_log.plot, 'down')
    plt.close('all')

    fig_in, ax = plt.subplots()
    fig_out = reject_log.plot(ax=ax)
    assert fig_in == fig_out
