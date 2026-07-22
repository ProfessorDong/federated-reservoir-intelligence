"""
Additional BCI data loaders: BCI-IV-2b (2-class, 9 subjects, multi-session)
used for the session-to-session drift experiments.
"""
import os
import numpy as np
import torch
import mne
import scipy.io

BCI_2B_DIR = os.path.join(os.path.dirname(__file__), '..', 'data',
                          'BCICompetition', 'DataSet2', 'extracted')
BCI_4_DIR = os.path.join(os.path.dirname(__file__), '..', 'data',
                         'BCICompetition', 'DataSet4', 'extracted')


# ═══════════════════════════════════════════════════════════════════════════════
# BCI Competition IV Dataset 2b
# 9 subjects, 2-class (left/right hand), 3 bipolar EEG channels
# 5 sessions: 01T, 02T, 03T training; 04E, 05E evaluation
# ═══════════════════════════════════════════════════════════════════════════════
def load_bci2b_subject(subject_id, sessions=None):
    """
    Load BCI-IV-2b data for one subject across specified sessions.

    Args:
        subject_id: 1-9
        sessions: list of session strings like ['01T','02T','03T','04E','05E']
                  Default: all training sessions

    Returns:
        session_data: dict mapping session -> (epochs, labels)
    """
    if sessions is None:
        sessions = ['01T', '02T', '03T']

    session_data = {}
    for sess in sessions:
        fname = os.path.join(BCI_2B_DIR, f'B{subject_id:02d}{sess}.gdf')
        if not os.path.exists(fname):
            continue

        try:
            raw = mne.io.read_raw_gdf(fname, preload=True, verbose=False)
        except Exception as e:
            print(f"    Failed to load {fname}: {e}")
            continue

        # Select EEG channels only (first 3 are bipolar EEG)
        eeg_chs = [ch for ch in raw.ch_names if 'EEG' in ch.upper()
                    or ch.startswith('EEG')]
        if len(eeg_chs) < 3:
            eeg_chs = raw.ch_names[:3]
        raw.pick(eeg_chs[:3])

        # Bandpass
        raw.filter(4.0, 40.0, method='iir', verbose=False)

        events, event_id = mne.events_from_annotations(raw, verbose=False)

        # Motor imagery: 769=left hand, 770=right hand
        mi_events = {}
        class_map = {}
        for key, val in event_id.items():
            if key == '769':
                mi_events['left'] = val
                class_map[val] = 0
            elif key == '770':
                mi_events['right'] = val
                class_map[val] = 1

        if not mi_events:
            continue

        epochs = mne.Epochs(raw, events, mi_events, tmin=0.5, tmax=2.5,
                            baseline=None, preload=True, verbose=False)
        data = epochs.get_data().astype(np.float32)
        labels = np.array([class_map[e] for e in epochs.events[:, 2]], dtype=np.int64)

        session_data[sess] = (data, labels)

    return session_data


def load_bci2b_all():
    """
    Load all BCI-IV-2b subjects.

    Returns:
        subjects: list of dicts, each mapping session -> (data, labels)
    """
    subjects = []
    for s in range(1, 10):
        print(f"  Subject {s}...")
        sess_data = load_bci2b_subject(s, sessions=['01T', '02T', '03T'])
        if sess_data:
            # Merge training sessions
            all_data = []
            all_labels = []
            for sess in sorted(sess_data.keys()):
                data, labels = sess_data[sess]
                all_data.append(data)
                all_labels.append(labels)
                print(f"    {sess}: {data.shape[0]} trials")
            merged_data = np.concatenate(all_data)
            merged_labels = np.concatenate(all_labels)
            subjects.append((merged_data, merged_labels, sess_data))
    return subjects


def bci2b_to_federated(subjects):
    """Convert BCI-IV-2b to federated format."""
    client_data = []
    client_labels = []
    for merged_data, merged_labels, _ in subjects:
        X = torch.from_numpy(merged_data.transpose(0, 2, 1))
        mean = X.mean(dim=(0, 1), keepdim=True)
        std = X.std(dim=(0, 1), keepdim=True) + 1e-8
        X = (X - mean) / std
        y = torch.from_numpy(merged_labels)
        client_data.append(X)
        client_labels.append(y)
    return client_data, client_labels


# ═══════════════════════════════════════════════════════════════════════════════
# BCI Competition IV Dataset 4 (ECoG finger movements)
# 3 subjects, 62/48 ECoG channels, 5 fingers (regression)
# ═══════════════════════════════════════════════════════════════════════════════
def load_bci4_subject(subject_id):
    """
    Load BCI-IV-4 ECoG data.

    Returns:
        train_data: (T_train, n_channels) numpy array
        train_dg: (T_train, 5) data glove values
        test_data: (T_test, n_channels) numpy array
    """
    fname = os.path.join(BCI_4_DIR, f'sub{subject_id}_comp.mat')
    mat = scipy.io.loadmat(fname)
    train_data = mat['train_data'].astype(np.float32)
    test_data = mat['test_data'].astype(np.float32)
    train_dg = mat['train_dg'].astype(np.float32)
    return train_data, train_dg, test_data


def load_bci4_all():
    """Load all 3 ECoG subjects."""
    subjects = []
    for s in range(1, 4):
        train_data, train_dg, test_data = load_bci4_subject(s)
        n_ch = train_data.shape[1]
        print(f"  Subject {s}: {train_data.shape[0]} train samples, "
              f"{test_data.shape[0]} test samples, {n_ch} channels")
        subjects.append((train_data, train_dg, test_data))
    return subjects


def bci4_to_federated(subjects, window_size=500, window_step=100):
    """
    Convert ECoG continuous data to windowed classification task.
    Convert regression to classification: which finger is most active.
    """
    client_data = []
    client_labels = []

    for train_data, train_dg, _ in subjects:
        T, d_x = train_data.shape
        segments = []
        labels = []

        for start in range(0, T - window_size, window_step):
            segment = train_data[start:start + window_size]
            dg_mean = train_dg[start:start + window_size].mean(axis=0)
            # Classify as the most active finger
            label = int(np.argmax(np.abs(dg_mean)))
            if np.max(np.abs(dg_mean)) > 0.1:  # threshold for movement
                segments.append(segment)
                labels.append(label)

        if segments:
            X = torch.from_numpy(np.array(segments, dtype=np.float32))
            # Normalize
            mean = X.mean(dim=(0, 1), keepdim=True)
            std = X.std(dim=(0, 1), keepdim=True) + 1e-8
            X = (X - mean) / std
            y = torch.from_numpy(np.array(labels, dtype=np.int64))
            client_data.append(X)
            client_labels.append(y)
            print(f"    → {len(segments)} segments, {len(np.unique(labels))} classes")

    return client_data, client_labels
