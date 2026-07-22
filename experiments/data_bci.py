"""
Data loading for BCI Competition IV-2a and Ninapro DB5.
"""
import os
import numpy as np
import torch
import mne
import scipy.io
from config import BCI_2A_DIR, NINAPRO_DB5_DIR, BCI2A_CFG, NINAPRO_CFG


# ═══════════════════════════════════════════════════════════════════════════════
# BCI Competition IV Dataset 2a
# ═══════════════════════════════════════════════════════════════════════════════
def load_bci2a_subject(subject_id, session='T'):
    """
    Load BCI-IV-2a data for one subject.

    Args:
        subject_id: 1-9
        session: 'T' (training) or 'E' (evaluation)

    Returns:
        epochs: (n_trials, n_channels, n_times) numpy array
        labels: (n_trials,) numpy array, values 0-3
    """
    fname = os.path.join(BCI_2A_DIR, f'A{subject_id:02d}{session}.gdf')
    if not os.path.exists(fname):
        raise FileNotFoundError(f"GDF file not found: {fname}")

    raw = mne.io.read_raw_gdf(fname, preload=True, verbose=False)

    # Select only EEG channels (first 22)
    eeg_channels = raw.ch_names[:22]
    raw.pick(eeg_channels)

    # Bandpass filter
    raw.filter(BCI2A_CFG['lowcut'], BCI2A_CFG['highcut'],
               method='iir', verbose=False)

    # Get events
    events, event_id = mne.events_from_annotations(raw, verbose=False)

    # Motor imagery event codes: 769=left hand, 770=right hand, 771=feet, 772=tongue
    mi_event_ids = {}
    class_map = {}
    for key, val in event_id.items():
        if key == '769':
            mi_event_ids['left_hand'] = val
            class_map[val] = 0
        elif key == '770':
            mi_event_ids['right_hand'] = val
            class_map[val] = 1
        elif key == '771':
            mi_event_ids['feet'] = val
            class_map[val] = 2
        elif key == '772':
            mi_event_ids['tongue'] = val
            class_map[val] = 3

    if not mi_event_ids:
        # Evaluation files may use event 783 (cue unknown) — need separate labels
        # Fall back to using event 768 (start of trial) and look for label file
        return _load_bci2a_eval_fallback(subject_id, raw, events, event_id)

    # Extract epochs
    tmin = BCI2A_CFG['epoch_tmin']
    tmax = BCI2A_CFG['epoch_tmax']
    epochs = mne.Epochs(raw, events, mi_event_ids, tmin=tmin, tmax=tmax,
                        baseline=None, preload=True, verbose=False)

    data = epochs.get_data()  # (n_trials, n_channels, n_times)
    labels = np.array([class_map[e] for e in epochs.events[:, 2]])

    return data.astype(np.float32), labels.astype(np.int64)


def _load_bci2a_eval_fallback(subject_id, raw, events, event_id):
    """
    Load evaluation session using true labels file if available,
    otherwise extract epochs around trial-start events (768).
    For eval sessions, we use the training session labels order.
    """
    # Try to find true labels file
    label_file = os.path.join(BCI_2A_DIR,
                              f'A{subject_id:02d}E.mat')
    true_labels = None
    if os.path.exists(label_file):
        mat = scipy.io.loadmat(label_file)
        for key in mat:
            if 'label' in key.lower() or 'class' in key.lower():
                true_labels = mat[key].flatten()
                break

    # Extract epochs around event 768 (start of trial)
    trial_event_id = None
    for key, val in event_id.items():
        if key == '768':
            trial_event_id = val
            break

    if trial_event_id is None:
        raise ValueError(f"Cannot find trial start events for subject {subject_id}")

    trial_events = events[events[:, 2] == trial_event_id]
    tmin = BCI2A_CFG['epoch_tmin']
    tmax = BCI2A_CFG['epoch_tmax']

    epochs_data = []
    for ev in trial_events:
        start = int(ev[0] + tmin * BCI2A_CFG['sfreq'])
        stop = int(ev[0] + tmax * BCI2A_CFG['sfreq'])
        if stop <= raw.n_times:
            data_chunk = raw.get_data(start=start, stop=stop)
            epochs_data.append(data_chunk)

    data = np.array(epochs_data, dtype=np.float32)

    if true_labels is not None:
        labels = true_labels[:len(data)].astype(np.int64)
        # Convert from 1-indexed to 0-indexed if needed
        if labels.min() >= 1:
            labels = labels - 1
    else:
        # Without labels, return None labels (cannot evaluate)
        labels = None

    return data, labels


def load_bci2a_all(session='T'):
    """
    Load all 9 subjects.

    Returns:
        list of (data, labels) tuples
    """
    subjects = []
    for s in range(1, 10):
        try:
            data, labels = load_bci2a_subject(s, session)
            if labels is not None:
                subjects.append((data, labels))
                print(f"  Subject {s}: {data.shape[0]} trials, "
                      f"{data.shape[1]} channels, {data.shape[2]} samples")
        except Exception as e:
            print(f"  Subject {s}: FAILED - {e}")
    return subjects


def bci2a_to_federated(subjects):
    """
    Convert BCI-IV-2a data to federated format with sub-band filtering.

    Each trial is filtered into mu (8-13 Hz) and beta (13-30 Hz) bands
    and concatenated along the channel axis.

    Returns:
        client_data: list of (n_trials, T, d_x) tensors
        client_labels: list of (n_trials,) tensors
    """
    from scipy.signal import butter, sosfiltfilt

    sfreq = BCI2A_CFG['sfreq']
    sub_bands = [(8, 13), (13, 30)]

    client_data = []
    client_labels = []
    for data, labels in subjects:
        # data: (n_trials, n_channels, n_times)
        band_data = []
        for low, high in sub_bands:
            sos = butter(5, [low, high], btype='bandpass',
                         fs=sfreq, output='sos')
            filtered = sosfiltfilt(sos, data, axis=2).astype(np.float32)
            band_data.append(filtered)

        # Concatenate: (n_trials, n_channels*2, n_times)
        data_multi = np.concatenate(band_data, axis=1)

        # Transpose to (n_trials, n_times, d_x)
        X = torch.from_numpy(data_multi.transpose(0, 2, 1))
        mean = X.mean(dim=(0, 1), keepdim=True)
        std = X.std(dim=(0, 1), keepdim=True) + 1e-8
        X = (X - mean) / std
        y = torch.from_numpy(labels)
        client_data.append(X)
        client_labels.append(y)
    return client_data, client_labels


# ═══════════════════════════════════════════════════════════════════════════════
# Ninapro DB5
# ═══════════════════════════════════════════════════════════════════════════════
def load_ninapro_db5_subject(subject_id):
    """
    Load Ninapro DB5 data for one subject (all 3 exercises).

    Returns:
        segments: (n_segments, window_size, n_channels) numpy array
        labels: (n_segments,) numpy array (0-indexed gesture labels)
    """
    subdir = os.path.join(NINAPRO_DB5_DIR, f's{subject_id}')
    all_emg = []
    all_labels = []
    label_offset = 0

    for ex in range(1, 4):
        fname = os.path.join(subdir, f'S{subject_id}_E{ex}_A1.mat')
        if not os.path.exists(fname):
            continue
        mat = scipy.io.loadmat(fname)
        emg = mat['emg'].astype(np.float32)  # (T, 16)
        restimulus = mat['restimulus'].flatten().astype(np.int64)

        # Find active gesture segments (restimulus > 0)
        # Relabel: exercise 1 has gestures 1-12, ex2 has 1-17, ex3 has 1-23
        # Remap to continuous labels
        for gesture_id in np.unique(restimulus):
            if gesture_id == 0:  # rest
                continue
            mask = restimulus == gesture_id
            indices = np.where(mask)[0]
            if len(indices) < NINAPRO_CFG['window_size']:
                continue

            global_label = label_offset + gesture_id - 1  # 0-indexed

            # Sliding window extraction
            start = indices[0]
            end = indices[-1] + 1
            ws = NINAPRO_CFG['window_size']
            step = NINAPRO_CFG['window_step']
            for w_start in range(start, end - ws + 1, step):
                w_end = w_start + ws
                # Only include if majority of window is this gesture
                segment_labels = restimulus[w_start:w_end]
                if (segment_labels == gesture_id).mean() > 0.8:
                    all_emg.append(emg[w_start:w_end])
                    all_labels.append(global_label)

        # Update offset for next exercise
        n_gestures_this_ex = len(np.unique(restimulus)) - 1  # exclude rest
        label_offset += n_gestures_this_ex

    segments = np.array(all_emg, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.int64)
    return segments, labels


def load_ninapro_db5_all():
    """Load all 10 subjects."""
    subjects = []
    for s in range(1, 11):
        try:
            segments, labels = load_ninapro_db5_subject(s)
            n_classes = len(np.unique(labels))
            print(f"  Subject {s}: {segments.shape[0]} segments, "
                  f"{n_classes} classes")
            subjects.append((segments, labels))
        except Exception as e:
            print(f"  Subject {s}: FAILED - {e}")
    return subjects


def ninapro_to_federated(subjects):
    """
    Convert Ninapro data to federated format.

    Returns:
        client_data: list of (n_segments, T, d_x) tensors
        client_labels: list of (n_segments,) tensors
    """
    client_data = []
    client_labels = []
    for segments, labels in subjects:
        X = torch.from_numpy(segments)  # (n_segments, T, d_x)
        # Z-score per subject
        mean = X.mean(dim=(0, 1), keepdim=True)
        std = X.std(dim=(0, 1), keepdim=True) + 1e-8
        X = (X - mean) / std
        y = torch.from_numpy(labels)
        client_data.append(X)
        client_labels.append(y)
    return client_data, client_labels
