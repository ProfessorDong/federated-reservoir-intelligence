"""
Experiment 3: Federated event-camera gesture recognition using DVS128 Gesture.
29 users × 11 gesture classes × AEDAT 3.1 spike events.
Each user is a federated client.

Fills: Table 3 (DVS128 Gesture), Supplementary Table 1 (spike features)
"""
import sys, os, json, time, gc
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from config import *
from reservoir import LSM, ridge_regression, compute_sufficient_statistics, \
    ridge_from_statistics, personalized_readout
from federated import FRIStatsAggregator, FRIReadoutAggregator, \
    evaluate_readout, comm_cost_mb
from data_event import load_dvs128_all, events_to_spike_input, events_to_frames
from baselines import LSTMClassifier, SNNClassifier, run_fedavg, run_lfnl


def labels_to_onehot(labels, n_classes):
    oh = torch.zeros(len(labels), n_classes)
    oh[torch.arange(len(labels)), labels] = 1.0
    return oh


def process_dvs_with_lsm(lsm, samples, duration_ms=300, dt_ms=1.0,
                          n_input=128, return_type='traces'):
    """
    Process DVS event samples through LSM.

    Args:
        samples: list of (events_dict, label)

    Returns:
        features: (n_samples, d_phi) tensor
        labels: (n_samples,) tensor
    """
    features_list = []
    labels_list = []

    for events, label in samples:
        # Convert events to input currents
        input_currents = events_to_spike_input(events, n_input,
                                                duration_ms, dt_ms)

        # Run LSM
        result = lsm.run(input_currents, return_type=return_type)

        if return_type == 'traces':
            feat = result['traces']
        elif return_type == 'multiscale':
            feat = result['multiscale']
        else:
            feat = result.get('traces', result.get('multiscale'))

        # Mean-pool over time
        trial_feat = feat.mean(dim=0)  # (d_phi,)
        features_list.append(trial_feat)
        labels_list.append(label)

    if not features_list:
        return torch.zeros(0, lsm.N_s), torch.zeros(0, dtype=torch.long)

    features = torch.stack(features_list)  # (n_samples, d_phi)
    labels = torch.tensor(labels_list, dtype=torch.long)
    return features, labels


def process_dvs_to_frames(samples, sensor_size=(128, 128),
                           time_window_us=50000):
    """Convert DVS events to frame sequences for LSTM/SNN baselines."""
    frame_seqs = []
    labels = []
    for events, label in samples:
        frames = events_to_frames(events, sensor_size, time_window_us)
        # Flatten spatial dims: (n_frames, 2, H, W) -> (n_frames, 2*H*W)
        # Too large — downsample spatially
        # Use 2x2 average pooling on frames
        frames_t = torch.from_numpy(frames)
        if frames_t.shape[2] > 32:
            frames_t = torch.nn.functional.avg_pool2d(frames_t, kernel_size=4)
        n_frames = frames_t.shape[0]
        feat_dim = frames_t.shape[1] * frames_t.shape[2] * frames_t.shape[3]
        frames_flat = frames_t.reshape(n_frames, feat_dim)
        frame_seqs.append(frames_flat)
        labels.append(label)
    return frame_seqs, labels


def pad_sequences(seqs, max_len=None):
    """Pad variable-length sequences to fixed length."""
    if max_len is None:
        max_len = max(s.shape[0] for s in seqs)
    max_len = min(max_len, 100)  # cap at 100 time steps
    d = seqs[0].shape[1]
    padded = torch.zeros(len(seqs), max_len, d)
    for i, s in enumerate(seqs):
        L = min(s.shape[0], max_len)
        padded[i, :L] = s[:L]
    return padded


def run_experiment(seed=0):
    print(f"\n{'='*60}")
    print(f"DVS128 Gesture Experiment — Seed {seed}")
    print(f"{'='*60}")

    # ── Load data ────────────────────────────────────────────────────────────
    print("\nLoading DVS128 Gesture data...")
    t0 = time.time()
    user_data, train_users, test_users = load_dvs128_all()
    print(f"  Loaded in {time.time()-t0:.1f}s")
    print(f"  Users with data: {len(user_data)}")
    print(f"  Train users: {len(train_users)}, Test users: {len(test_users)}")

    # DVS128 splits users 1-23 for training, 24-29 for testing.
    # For federated learning: use training users as clients, split their data 80/20.
    valid_users = [u for u in user_data if user_data[u]['train']]
    print(f"  Valid users (with training data): {len(valid_users)}")

    # Split each user's training data into local train/test
    rng = np.random.RandomState(seed)
    for uid in valid_users:
        samples = user_data[uid]['train']
        rng.shuffle(samples)
        n_test = max(1, int(len(samples) * 0.2))
        user_data[uid]['test'] = samples[:n_test]
        user_data[uid]['train'] = samples[n_test:]

    K = len(valid_users)
    n_classes = DVS128_CFG['n_classes']
    results = {}

    # ── LSM setup ────────────────────────────────────────────────────────────
    print("\nInitializing LSM...")
    n_input = 128  # spatial hash of DVS events
    lsm = LSM(
        input_dim=n_input,
        n_neurons=LSM_CFG['N_s'],
        output_dim=n_classes,
        tau_m=LSM_CFG['tau_m'],
        V_th=LSM_CFG['V_th'],
        V_rest=LSM_CFG['V_rest'],
        V_reset=LSM_CFG['V_reset'],
        tau_ref=LSM_CFG['tau_ref'],
        connectivity=LSM_CFG['connectivity'],
        dt=LSM_CFG['dt'],
        trace_decay=LSM_CFG['trace_decay'],
        seed=seed,
    )
    lam = LSM_CFG['ridge_lambda']

    # ── Process through LSM (traces) ─────────────────────────────────────────
    print("\nProcessing through LSM (traces)...")
    train_features, train_labels = [], []
    test_features, test_labels = [], []

    for uid in valid_users:
        print(f"  User {uid}...", end=' ')
        t1 = time.time()
        tr_feat, tr_lab = process_dvs_with_lsm(
            lsm, user_data[uid]['train'], duration_ms=300, dt_ms=1.0,
            n_input=n_input, return_type='traces')
        te_feat, te_lab = process_dvs_with_lsm(
            lsm, user_data[uid]['test'], duration_ms=300, dt_ms=1.0,
            n_input=n_input, return_type='traces')
        train_features.append(tr_feat.float() if isinstance(tr_feat, torch.Tensor) else tr_feat)
        train_labels.append(tr_lab)
        test_features.append(te_feat.float() if isinstance(te_feat, torch.Tensor) else te_feat)
        test_labels.append(te_lab)
        print(f"train={len(tr_lab)}, test={len(te_lab)}, {time.time()-t1:.1f}s")

    train_labels_oh = [labels_to_onehot(y, n_classes).to(DEVICE) for y in train_labels]

    # ── 1. Local-only ────────────────────────────────────────────────────────
    print("\n[1] Local-only LSM...")
    local_accs = []
    for k in range(K):
        if len(train_features[k]) == 0 or len(test_features[k]) == 0:
            continue
        W_k = ridge_regression(train_features[k], train_labels_oh[k], lam)
        acc, _ = evaluate_readout(W_k, test_features[k], test_labels[k])
        local_accs.append(acc)
    results['local_only'] = {'acc': np.mean(local_accs) if local_accs else 0.0, 'comm_mb': 0.0}
    print(f"  Acc: {results['local_only']['acc']:.4f}")

    # ── 2. Centralized LSM ──────────────────────────────────────────────────
    print("\n[2] Centralized LSM...")
    valid_k = [k for k in range(K) if len(train_features[k]) > 0]
    all_feat = torch.cat([train_features[k] for k in valid_k])
    all_lab = torch.cat([train_labels_oh[k] for k in valid_k])
    W_cent = ridge_regression(all_feat, all_lab, lam)
    cent_accs = []
    for k in valid_k:
        if len(test_features[k]) > 0:
            acc, _ = evaluate_readout(W_cent, test_features[k], test_labels[k])
            cent_accs.append(acc)
    results['centralized'] = {'acc': np.mean(cent_accs) if cent_accs else 0.0}
    print(f"  Acc: {results['centralized']['acc']:.4f}")

    # ── 3. FRI-LSM (traces, stats) ───────────────────────────────────────────
    print("\n[3] FRI-LSM (traces, stats-based)...")
    valid_train = [train_features[k] for k in valid_k]
    valid_train_oh = [train_labels_oh[k] for k in valid_k]
    fri_s = FRIStatsAggregator(lam)
    W_s, comm_s = fri_s.run(valid_train, valid_train_oh)

    s_accs = []
    for k in valid_k:
        if len(test_features[k]) > 0:
            acc, _ = evaluate_readout(W_s, test_features[k], test_labels[k])
            s_accs.append(acc)
    results['fri_lsm_traces'] = {
        'acc': np.mean(s_accs) if s_accs else 0.0,
        'comm_mb': comm_cost_mb(comm_s),
        'bptt': False,
    }
    print(f"  Acc: {results['fri_lsm_traces']['acc']:.4f}, Comm: {results['fri_lsm_traces']['comm_mb']:.4f} MB")

    # ── 4. FRI-LSM (multi-scale) ─────────────────────────────────────────────
    print("\n[4] FRI-LSM (multi-scale)...")
    gc.collect()
    torch.cuda.empty_cache()
    ms_train_features, ms_test_features = [], []
    for uid_idx, uid in enumerate(valid_users):
        tr_feat, tr_lab = process_dvs_with_lsm(
            lsm, user_data[uid]['train'], duration_ms=300, dt_ms=1.0,
            n_input=n_input, return_type='multiscale')
        te_feat, te_lab = process_dvs_with_lsm(
            lsm, user_data[uid]['test'], duration_ms=300, dt_ms=1.0,
            n_input=n_input, return_type='multiscale')
        ms_train_features.append(tr_feat.float() if isinstance(tr_feat, torch.Tensor) else tr_feat)
        ms_test_features.append(te_feat.float() if isinstance(te_feat, torch.Tensor) else te_feat)

    valid_ms = [k for k in range(K) if len(ms_train_features[k]) > 0]
    ms_valid_train = [ms_train_features[k] for k in valid_ms]
    ms_valid_train_oh = [train_labels_oh[k] for k in valid_ms]
    W_ms, comm_ms = fri_s.run(ms_valid_train, ms_valid_train_oh)

    ms_accs = []
    for k in valid_ms:
        if len(ms_test_features[k]) > 0:
            acc, _ = evaluate_readout(W_ms, ms_test_features[k], test_labels[k])
            ms_accs.append(acc)
    results['fri_lsm_multiscale'] = {
        'acc': np.mean(ms_accs) if ms_accs else 0.0,
        'comm_mb': comm_cost_mb(comm_ms),
        'bptt': False,
    }
    print(f"  Acc: {results['fri_lsm_multiscale']['acc']:.4f}")
    del ms_train_features, ms_test_features, ms_valid_train, ms_valid_train_oh, W_ms
    gc.collect()

    # ── 5. FRI-LSM (sketched) ────────────────────────────────────────────────
    print("\n[5] FRI-LSM (sketched)...")
    for m_sketch in [100, 50]:
        # Random projection sketch
        rng = np.random.RandomState(seed)
        d_phi = train_features[0].shape[1] if len(train_features[0]) > 0 else LSM_CFG['N_s']
        S = torch.from_numpy(
            (rng.randn(m_sketch, d_phi) / np.sqrt(m_sketch)).astype(np.float32)
        ).to(DEVICE)

        sketch_train = [f.float().to(DEVICE) @ S.T if len(f) > 0 else torch.zeros(0, m_sketch, device=DEVICE)
                        for f in train_features]
        sketch_test = [f.float().to(DEVICE) @ S.T if len(f) > 0 else torch.zeros(0, m_sketch, device=DEVICE)
                       for f in test_features]
        sketch_train_oh = [labels_to_onehot(y, n_classes).to(DEVICE) for y in train_labels]

        valid_sk = [k for k in range(K) if len(sketch_train[k]) > 0]
        sk_valid_train = [sketch_train[k] for k in valid_sk]
        sk_valid_oh = [sketch_train_oh[k] for k in valid_sk]

        fri_sk = FRIStatsAggregator(lam)
        W_sk, comm_sk = fri_sk.run(sk_valid_train, sk_valid_oh)

        sk_accs = []
        for k in valid_sk:
            if len(sketch_test[k]) > 0:
                acc, _ = evaluate_readout(W_sk, sketch_test[k], test_labels[k])
                sk_accs.append(acc)

        results[f'fri_lsm_sketched_m{m_sketch}'] = {
            'acc': np.mean(sk_accs) if sk_accs else 0.0,
            'comm_mb': comm_cost_mb(comm_sk),
            'bptt': False,
        }
        print(f"  m={m_sketch}: Acc={results[f'fri_lsm_sketched_m{m_sketch}']['acc']:.4f}")

    # Free reservoir features before loading frame-based data
    del train_features, test_features, train_labels_oh
    gc.collect()
    torch.cuda.empty_cache()

    # ── 6. FedAvg-LSTM (frame-based) ─────────────────────────────────────────
    print("\n[6] FedAvg-LSTM (frame-based)...")
    frame_train_X, frame_train_y = [], []
    frame_test_X, frame_test_y = [], []
    for uid in valid_users:
        tr_frames, tr_labels = process_dvs_to_frames(user_data[uid]['train'])
        te_frames, te_labels = process_dvs_to_frames(user_data[uid]['test'])
        if tr_frames and te_frames:
            frame_train_X.append(pad_sequences(tr_frames))
            frame_train_y.append(torch.tensor(tr_labels, dtype=torch.long))
            frame_test_X.append(pad_sequences(te_frames))
            frame_test_y.append(torch.tensor(te_labels, dtype=torch.long))

    if frame_train_X:
        frame_dim = frame_train_X[0].shape[2]

        def make_lstm():
            return LSTMClassifier(frame_dim, LSTM_CFG['hidden_dim'], n_classes,
                                  LSTM_CFG['num_layers'], LSTM_CFG['dropout'])

        _, lstm_m = run_fedavg(
            make_lstm, frame_train_X, frame_train_y,
            frame_test_X, frame_test_y,
            num_rounds=FED_CFG['num_rounds'],
            participation_rate=FED_CFG['participation_dvs'],
            local_epochs=LSTM_CFG['local_epochs'],
            lr=LSTM_CFG['lr'], batch_size=LSTM_CFG['batch_size'],
            seed=seed)
        results['fedavg_lstm'] = {
            'acc': lstm_m['acc'], 'comm_mb': comm_cost_mb(lstm_m['comm_scalars']),
            'bptt': True,
        }
        print(f"  Acc: {lstm_m['acc']:.4f}")

    # ── 7. FedAvg-SNN ────────────────────────────────────────────────────────
    print("\n[7] FedAvg-SNN...")
    # Use same frame data for SNN
    if frame_train_X:
        def make_snn():
            return SNNClassifier(frame_dim, SNN_CFG['hidden_dim'], n_classes,
                                 SNN_CFG['num_steps'], SNN_CFG['beta'])

        _, snn_m = run_fedavg(
            make_snn, frame_train_X, frame_train_y,
            frame_test_X, frame_test_y,
            num_rounds=FED_CFG['num_rounds'],
            participation_rate=FED_CFG['participation_dvs'],
            local_epochs=SNN_CFG['local_epochs'],
            lr=SNN_CFG['lr'], batch_size=SNN_CFG['batch_size'],
            seed=seed)
        results['fedavg_snn'] = {
            'acc': snn_m['acc'], 'comm_mb': comm_cost_mb(snn_m['comm_scalars']),
            'bptt': True,
        }
        print(f"  Acc: {snn_m['acc']:.4f}")

    # ── 8. FedProx-SNN ───────────────────────────────────────────────────────
    print("\n[8] FedProx-SNN...")
    if frame_train_X:
        _, prox_snn_m = run_fedavg(
            make_snn, frame_train_X, frame_train_y,
            frame_test_X, frame_test_y,
            num_rounds=FED_CFG['num_rounds'],
            participation_rate=FED_CFG['participation_dvs'],
            local_epochs=SNN_CFG['local_epochs'],
            lr=SNN_CFG['lr'], batch_size=SNN_CFG['batch_size'],
            fedprox_mu=0.01, seed=seed)
        results['fedprox_snn'] = {
            'acc': prox_snn_m['acc'], 'comm_mb': comm_cost_mb(prox_snn_m['comm_scalars']),
            'bptt': True,
        }
        print(f"  Acc: {prox_snn_m['acc']:.4f}")

    # ── 9. LFNL-style (federated SNN, readout-only aggregation) ─────────────
    print("\n[9] LFNL (SNN readout-only)...")
    if frame_train_X:
        _, lfnl_m = run_lfnl(
            make_snn, frame_train_X, frame_train_y,
            frame_test_X, frame_test_y,
            num_rounds=FED_CFG['num_rounds'],
            participation_rate=FED_CFG['participation_dvs'],
            local_epochs=SNN_CFG['local_epochs'],
            lr=SNN_CFG['lr'], batch_size=SNN_CFG['batch_size'],
            seed=seed)
        results['lfnl'] = {
            'acc': lfnl_m['acc'], 'comm_mb': comm_cost_mb(lfnl_m['comm_scalars']),
            'bptt': True,
        }
        print(f"  Acc: {lfnl_m['acc']:.4f}")

    return results


def main():
    import gc
    all_results = {}
    for seed in SEEDS:
        try:
            all_results[seed] = run_experiment(seed)
        except Exception as e:
            print(f"Seed {seed} failed: {e}")
            import traceback; traceback.print_exc()
        gc.collect()
        torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("AGGREGATED RESULTS")
    print("=" * 60)
    methods = ['local_only', 'centralized', 'fri_lsm_traces',
               'fri_lsm_multiscale', 'fri_lsm_sketched_m100',
               'fri_lsm_sketched_m50', 'fedavg_lstm', 'fedavg_snn',
               'fedprox_snn', 'lfnl']
    for m in methods:
        accs = [all_results[s][m]['acc'] for s in SEEDS if m in all_results[s]]
        if accs:
            print(f"  {m:30s}: {np.mean(accs):.4f} ± {1.96*np.std(accs)/np.sqrt(len(accs)):.4f}")

    save_path = os.path.join(RESULTS_DIR, 'dvs128_results.json')
    def to_ser(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: to_ser(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_ser(v) for v in obj]
        return obj
    with open(save_path, 'w') as f:
        json.dump(to_ser(all_results), f, indent=2)
    print(f"\nResults saved to {save_path}")


if __name__ == '__main__':
    main()
