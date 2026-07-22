"""
Data loading for DVS128 Gesture and N-Caltech101.
"""
import os
import struct
import numpy as np
import torch
from config import DVS128_DIR, NCALTECH_DIR, DVS128_CFG, NCALTECH_CFG


# ═══════════════════════════════════════════════════════════════════════════════
# DVS128 Gesture — AEDAT 3.1 parser
# ═══════════════════════════════════════════════════════════════════════════════
def read_aedat31(filepath, max_events=None):
    """
    Read AEDAT 3.1 file (IBM DVS Gesture format).

    Returns:
        events: dict with 'x', 'y', 'polarity', 'timestamp' numpy arrays
    """
    with open(filepath, 'rb') as f:
        # Skip header lines
        line = f.readline()
        while b'END-HEADER' not in line:
            line = f.readline()

        all_x, all_y, all_p, all_t = [], [], [], []
        total_events = 0

        while True:
            # Read packet header (28 bytes)
            header_bytes = f.read(28)
            if len(header_bytes) < 28:
                break

            event_type = struct.unpack('<H', header_bytes[0:2])[0]
            event_size = struct.unpack('<I', header_bytes[4:8])[0]
            event_number = struct.unpack('<I', header_bytes[20:24])[0]

            if event_type != 1:  # Not polarity events
                f.seek(event_size * event_number, 1)
                continue

            # Read all events in this packet
            packet_data = f.read(event_size * event_number)
            if len(packet_data) < event_size * event_number:
                break

            for i in range(event_number):
                offset = i * event_size
                data = struct.unpack('<I', packet_data[offset:offset+4])[0]
                ts = struct.unpack('<i', packet_data[offset+4:offset+8])[0]

                # DVS128 AEDAT 3.1 format (from README):
                # x = (data >> 17) & 0x1FFF
                # y = (data >> 2) & 0x1FFF
                # polarity = (data >> 1) & 1
                x = (data >> 17) & 0x00001FFF
                y = (data >> 2) & 0x00001FFF
                pol = (data >> 1) & 0x00000001

                all_x.append(x)
                all_y.append(y)
                all_p.append(pol)
                all_t.append(ts)

                total_events += 1
                if max_events and total_events >= max_events:
                    break

            if max_events and total_events >= max_events:
                break

    return {
        'x': np.array(all_x, dtype=np.int16),
        'y': np.array(all_y, dtype=np.int16),
        'polarity': np.array(all_p, dtype=np.uint8),
        'timestamp': np.array(all_t, dtype=np.int64),
    }


def read_aedat31_fast(filepath, max_events=None):
    """
    Fast AEDAT 3.1 reader using numpy for bulk parsing.
    """
    with open(filepath, 'rb') as f:
        # Skip header
        line = f.readline()
        while b'END-HEADER' not in line:
            line = f.readline()

        all_x, all_y, all_p, all_t = [], [], [], []

        while True:
            header_bytes = f.read(28)
            if len(header_bytes) < 28:
                break

            event_type = struct.unpack('<H', header_bytes[0:2])[0]
            event_size = struct.unpack('<I', header_bytes[4:8])[0]
            event_number = struct.unpack('<I', header_bytes[20:24])[0]

            if event_type != 1:
                f.seek(event_size * event_number, 1)
                continue

            packet_data = f.read(event_size * event_number)
            if len(packet_data) < event_size * event_number:
                break

            # Parse with numpy
            arr = np.frombuffer(packet_data, dtype=np.uint32).reshape(-1, 2)
            data_col = arr[:, 0]
            ts_col = arr[:, 1].astype(np.int64)

            x = ((data_col >> 17) & 0x1FFF).astype(np.int16)
            y = ((data_col >> 2) & 0x1FFF).astype(np.int16)
            p = ((data_col >> 1) & 1).astype(np.uint8)

            all_x.append(x)
            all_y.append(y)
            all_p.append(p)
            all_t.append(ts_col)

            if max_events:
                total = sum(len(a) for a in all_x)
                if total >= max_events:
                    break

    if not all_x:
        return {'x': np.array([], dtype=np.int16),
                'y': np.array([], dtype=np.int16),
                'polarity': np.array([], dtype=np.uint8),
                'timestamp': np.array([], dtype=np.int64)}

    return {
        'x': np.concatenate(all_x)[:max_events] if max_events else np.concatenate(all_x),
        'y': np.concatenate(all_y)[:max_events] if max_events else np.concatenate(all_y),
        'polarity': np.concatenate(all_p)[:max_events] if max_events else np.concatenate(all_p),
        'timestamp': np.concatenate(all_t)[:max_events] if max_events else np.concatenate(all_t),
    }


def events_to_frames(events, sensor_size=(128, 128), time_window_us=300000):
    """
    Bin events into fixed-duration frames.

    Returns:
        frames: (n_frames, 2, H, W) numpy array (2 channels for ON/OFF polarity)
    """
    H, W = sensor_size
    ts = events['timestamp']
    if len(ts) == 0:
        return np.zeros((1, 2, H, W), dtype=np.float32)

    t_start = ts.min()
    t_end = ts.max()
    n_frames = max(1, int((t_end - t_start) / time_window_us) + 1)
    frames = np.zeros((n_frames, 2, H, W), dtype=np.float32)

    frame_idx = np.clip((ts - t_start) // time_window_us, 0, n_frames - 1).astype(int)
    x = np.clip(events['x'], 0, W - 1)
    y = np.clip(events['y'], 0, H - 1)
    pol = events['polarity']

    for i in range(len(ts)):
        frames[frame_idx[i], pol[i], y[i], x[i]] += 1.0

    return frames


def events_to_spike_input(events, n_input_neurons, duration_ms, dt_ms=1.0):
    """
    Convert events to spike input currents for LSM.
    Maps (x, y) to input neuron index via spatial hashing.

    Returns:
        input_currents: (T, n_input_neurons) tensor
    """
    ts = events['timestamp']
    if len(ts) == 0:
        T = max(1, int(duration_ms / dt_ms))
        return torch.zeros(T, n_input_neurons)

    t_start = ts.min()
    T = max(1, int(duration_ms / dt_ms))
    dt_us = dt_ms * 1000

    input_currents = np.zeros((T, n_input_neurons), dtype=np.float32)
    x = events['x']
    y = events['y']
    pol = events['polarity']

    # Map (x, y, pol) to neuron index
    neuron_idx = ((x * 128 + y) * 2 + pol) % n_input_neurons
    time_idx = np.clip(((ts - t_start) / dt_us).astype(int), 0, T - 1)

    for i in range(len(ts)):
        input_currents[time_idx[i], neuron_idx[i]] += 1.0

    return torch.from_numpy(input_currents)


# ═══════════════════════════════════════════════════════════════════════════════
# DVS128 Gesture — dataset loading
# ═══════════════════════════════════════════════════════════════════════════════
def load_dvs128_trial(aedat_path, csv_path, max_events_per_gesture=50000):
    """
    Load one DVS128 trial (one subject, one lighting condition).

    Returns:
        list of (events_dict, label) tuples
    """
    # Read annotations
    gestures = []
    with open(csv_path, 'r') as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                cls = int(parts[0])
                start = int(parts[1])
                end = int(parts[2])
                gestures.append((cls, start, end))

    # Read all events
    all_events = read_aedat31_fast(aedat_path)

    # Segment by gesture
    samples = []
    for cls, start_us, end_us in gestures:
        if cls < 1 or cls > 11:
            continue
        mask = (all_events['timestamp'] >= start_us) & \
               (all_events['timestamp'] < end_us)
        n_events = mask.sum()
        if n_events < 10:
            continue

        gesture_events = {
            'x': all_events['x'][mask][:max_events_per_gesture],
            'y': all_events['y'][mask][:max_events_per_gesture],
            'polarity': all_events['polarity'][mask][:max_events_per_gesture],
            'timestamp': all_events['timestamp'][mask][:max_events_per_gesture],
        }
        samples.append((gesture_events, cls - 1))  # 0-indexed

    return samples


def load_dvs128_all():
    """
    Load all DVS128 Gesture data, organized by user.

    Returns:
        user_data: dict mapping user_id -> list of (events, label)
        train_users: list of user_ids for training
        test_users: list of user_ids for testing
    """
    # Read train/test splits
    with open(os.path.join(DVS128_DIR, 'trials_to_train.txt'), 'r') as f:
        train_files = [line.strip() for line in f if line.strip()]
    with open(os.path.join(DVS128_DIR, 'trials_to_test.txt'), 'r') as f:
        test_files = [line.strip() for line in f if line.strip()]

    # Extract user IDs from filenames
    def get_user(fname):
        return fname.split('_')[0]

    train_users = sorted(set(get_user(f) for f in train_files))
    test_users = sorted(set(get_user(f) for f in test_files))

    user_data = {}

    # Load training data
    for fname in train_files:
        aedat_path = os.path.join(DVS128_DIR, fname)
        csv_path = aedat_path.replace('.aedat', '_labels.csv')
        if not os.path.exists(aedat_path) or not os.path.exists(csv_path):
            continue
        user_id = get_user(fname)
        try:
            samples = load_dvs128_trial(aedat_path, csv_path)
            if user_id not in user_data:
                user_data[user_id] = {'train': [], 'test': []}
            user_data[user_id]['train'].extend(samples)
        except Exception as e:
            print(f"  Warning: failed to load {fname}: {e}")

    # Load test data
    for fname in test_files:
        aedat_path = os.path.join(DVS128_DIR, fname)
        csv_path = aedat_path.replace('.aedat', '_labels.csv')
        if not os.path.exists(aedat_path) or not os.path.exists(csv_path):
            continue
        user_id = get_user(fname)
        try:
            samples = load_dvs128_trial(aedat_path, csv_path)
            if user_id not in user_data:
                user_data[user_id] = {'train': [], 'test': []}
            user_data[user_id]['test'].extend(samples)
        except Exception as e:
            print(f"  Warning: failed to load {fname}: {e}")

    return user_data, train_users, test_users


# ═══════════════════════════════════════════════════════════════════════════════
# N-Caltech101 — binary file parser
# ═══════════════════════════════════════════════════════════════════════════════
def read_ncaltech101_bin(filepath):
    """
    Read N-Caltech101 binary file.
    Each event: 5 bytes (40 bits)
      bits 39-32: X (8 bits)
      bits 31-24: Y (8 bits)
      bit 23: Polarity
      bits 22-0: Timestamp (us)
    """
    with open(filepath, 'rb') as f:
        raw = f.read()

    n_events = len(raw) // 5
    if n_events == 0:
        return {'x': np.array([], dtype=np.int16),
                'y': np.array([], dtype=np.int16),
                'polarity': np.array([], dtype=np.uint8),
                'timestamp': np.array([], dtype=np.int64)}

    x = np.zeros(n_events, dtype=np.int16)
    y = np.zeros(n_events, dtype=np.int16)
    pol = np.zeros(n_events, dtype=np.uint8)
    ts = np.zeros(n_events, dtype=np.int64)

    raw_bytes = np.frombuffer(raw, dtype=np.uint8)
    # Reshape: each event is 5 bytes
    if len(raw_bytes) >= n_events * 5:
        events_raw = raw_bytes[:n_events * 5].reshape(n_events, 5)
        x = events_raw[:, 0].astype(np.int16)
        y = events_raw[:, 1].astype(np.int16)
        pol = (events_raw[:, 2] >> 7).astype(np.uint8)
        ts = (((events_raw[:, 2].astype(np.int64) & 0x7F) << 16) |
              (events_raw[:, 3].astype(np.int64) << 8) |
              events_raw[:, 4].astype(np.int64))

    return {'x': x, 'y': y, 'polarity': pol, 'timestamp': ts}


def load_ncaltech101(n_classes=20, max_samples_per_class=50):
    """
    Load N-Caltech101 dataset (subset of classes).

    Returns:
        samples: list of (events_dict, label) tuples
        class_names: list of class name strings
    """
    if not os.path.isdir(NCALTECH_DIR):
        raise FileNotFoundError(f"N-Caltech101 not found at {NCALTECH_DIR}")

    # Get all categories sorted, exclude BACKGROUND
    all_cats = sorted([d for d in os.listdir(NCALTECH_DIR)
                       if os.path.isdir(os.path.join(NCALTECH_DIR, d))
                       and d != 'BACKGROUND_Google'])

    # Select first n_classes categories (alphabetically, most standard)
    selected_cats = all_cats[:n_classes]
    print(f"  Selected {len(selected_cats)} classes: {selected_cats[:5]}...")

    samples = []
    for cls_idx, cat in enumerate(selected_cats):
        cat_dir = os.path.join(NCALTECH_DIR, cat)
        bin_files = sorted([f for f in os.listdir(cat_dir) if f.endswith('.bin')])
        bin_files = bin_files[:max_samples_per_class]

        for bf in bin_files:
            try:
                events = read_ncaltech101_bin(os.path.join(cat_dir, bf))
                if len(events['x']) > 10:
                    samples.append((events, cls_idx))
            except Exception:
                continue

    print(f"  Loaded {len(samples)} total samples from {len(selected_cats)} classes")
    return samples, selected_cats


def partition_dirichlet(samples, n_clients, alpha, seed=0):
    """
    Partition samples into clients using Dirichlet distribution (non-IID).

    Returns:
        client_samples: list of n_clients lists of (events, label)
    """
    rng = np.random.RandomState(seed)
    labels = np.array([s[1] for s in samples])
    n_classes = len(np.unique(labels))

    client_indices = [[] for _ in range(n_clients)]

    for c in range(n_classes):
        class_idx = np.where(labels == c)[0]
        rng.shuffle(class_idx)

        proportions = rng.dirichlet(np.repeat(alpha, n_clients))
        proportions = (proportions * len(class_idx)).astype(int)
        # Fix rounding
        proportions[-1] = len(class_idx) - proportions[:-1].sum()

        start = 0
        for k in range(n_clients):
            end = start + proportions[k]
            client_indices[k].extend(class_idx[start:end].tolist())
            start = end

    client_samples = []
    for k in range(n_clients):
        rng.shuffle(client_indices[k])
        client_samples.append([samples[i] for i in client_indices[k]])

    return client_samples
