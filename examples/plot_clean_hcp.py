import matplotlib
matplotlib.use('agg')

import os
import getpass
import os.path as op
import numpy as np
from functools import partial

import mne
from mne import write_evokeds, combine_evoked

import hcp
from hcp.preprocessing import interpolate_missing_channels

from library.hcp_utils import compensate_reference, decimate_raw
from autoreject import LocalAutoRejectCV, compute_thresholds

username = getpass.getuser()

mne.utils.set_log_level('warning')

if username == 'mjas':  # telecom workstation
    storage_dir = '/tsi/doctorants/data_gramfort/HCP2'
    hcp_path = op.join(storage_dir, 'HCP')
elif username == 'mainak':  # laptop
    hcp_path = '/home/mainak/Desktop/projects/auto_reject/HCP/'

run_index = 0
data_type = 'task_working_memory'
tmin, tmax = -1.5, 2.5
subjects = np.loadtxt('hcp_good_subjects.txt', dtype=str)
decim = 4


def _sum_evokeds(evokeds, evoked_gt, tmin):
    evoked = combine_evoked(evokeds, weights='nave')
    if evoked_gt.times[0] != evoked.times[0]:
        tmin = tmin + 1 / evoked.info['sfreq']
        evoked.crop(tmin=tmin)
    return evoked

if not os.path.exists('../data/evokeds_hcp'):
    os.mkdir('../data/evokeds_hcp')

methods = ['noreject', 'autoreject', 'sns', 'faster', 'ransac']

for subject in subjects:

    if not op.exists(op.join(hcp_path, subject)):
        continue

    print('Processing subject %s' % subject)
    # preprocessed evoked / epochs for ground-truth
    #
    # epochs_gt = hcp.io.read_epochs_hcp(subject, data_type=data_type,
    #                                    onset='stim', hcp_path=hcp_path)
    try:
        # do this for run_index 0 and 1
        evoked_gt = hcp.io.read_evokeds_hcp(subject, data_type=data_type,
                                            onset='stim', hcp_path=hcp_path)[7]
    except IOError:  # dont process subject if ground truth doesn't exist
        continue

    assert evoked_gt.comment == 'Wrkmem_LM-TIM-face_BT-diff_MODE-mag'
    evoked_gt = interpolate_missing_channels(
        evoked_gt, subject=subject, data_type=data_type, hcp_path=hcp_path,
        run_index=run_index)
    for ch in evoked_gt.info['chs']:
        x, y, z = ch['loc'][:3]
        ch['loc'][:3] = -y, x, z

    # read in raw data and filter it
    #
    evokeds = dict()
    for m in methods:
        evokeds[m] = list()
    for run_index in [0, 1]:

        # File IO
        #
        raw = hcp.io.read_raw_hcp(subject=subject, hcp_path=hcp_path,
                                  data_type=data_type, run_index=run_index)
        annots = hcp.io.read_annot_hcp(subject, data_type, hcp_path=hcp_path,
                                       run_index=run_index)
        trial_info = hcp.io.read_trial_info_hcp(
            subject, data_type, hcp_path=hcp_path, run_index=run_index)

        #
        # create events array
        faces_idx = trial_info['TIM']['codes'][:, 3] == 1
        onsets = np.round((trial_info['TIM']['codes'][faces_idx, 6] - 1) /
                          decim).astype(int)

        n_events = onsets.shape[0]
        events = np.zeros((n_events, 3), dtype=int)
        events[:, 0] = onsets
        events[:, 1] = 0
        events[:, 2] = 1  # let's makes faces = event_id: 1

        decimate_raw(raw, decim)
        compensate_reference(raw)

        # XXX: MNE complains if l_freq = 0.5 Hz
        raw.filter(0.55, 60, method='iir',
                   iir_params=dict(order=4, ftype='butter'),
                   n_jobs=-1)

        # Apply ICA for EOG
        raw.pick_types(stim=True, meg=True, exclude=[])
        ica_mat = hcp.io.read_ica_hcp(subject, hcp_path=hcp_path,
                                      data_type=data_type,
                                      run_index=run_index)
        if ica_mat is not None:
            # exclude = np.array(annots['ica']['ecg_eog_ic']) - 1
            exclude = np.setdiff1d(range(annots['ica']['total_ic_number'][0]),
                                   np.array(annots['ica']['brain_ic_vs']) - 1)
        hcp.preprocessing.apply_ica_hcp(raw, ica_mat=ica_mat, exclude=exclude)

        # coordinate system magic for spatial_colors
        #
        for ch in raw.info['chs']:
            x, y, z = ch['loc'][:3]
            ch['loc'][:3] = -y, x, z

        # extract epochs
        #
        event_id = dict(faces=1)
        epochs_kwargs = dict(events=events, event_id=event_id,
                             tmin=tmin, tmax=tmax, proj=False,
                             reject=None, baseline=None, preload=True)
        epochs = mne.Epochs(raw, **epochs_kwargs)
        del raw

        # also prevent interpolation bug with picks
        epochs.pick_types(stim=False, meg=True)  # remove trigger channel now

        # apply autoreject
        #
        thresh_func = partial(compute_thresholds,
                              method='bayesian_optimization',
                              random_state=42)
        ar = LocalAutoRejectCV(thresh_func=thresh_func)
        epochs_ar = ar.fit_transform(epochs['faces'])
        evokeds['autoreject'].append(epochs_ar.average())

        # compute and plot evokeds
        #
        evokeds['noreject'].append(epochs['faces'].average())
