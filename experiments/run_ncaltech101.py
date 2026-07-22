"""
Experiment 4: Federated event-camera object recognition using N-Caltech101.
101 categories, partitioned into synthetic federated clients via Dirichlet.

Fills: Additional event-camera results (N-Caltech101)
"""
import sys, os, json, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from config import *
from reservoir import LSM, ridge_regression
from federated import FRIStatsAggregator, evaluate_readout, comm_cost_mb
from data_event import load_ncaltech101, partition_dirichlet, events_to_spike_input
from baselines import LSTMClassifier, SNNClassifier, run_fedavg, run_lfnl
from data_event import events_to_frames


def labels_to_onehot(labels, n_classes):
    oh = torch.zeros(len(labels), n_classes)
    oh[torch.arange(len(labels)), labels] = 1.0
    return oh


def process_ncal_with_lsm(lsm, samples, n_input=128, duration_ms=200, dt_ms=1.0):
    """Process N-Caltech101 samples through LSM."""
    features_list = []
    labels_list = []
    for events, label in samples:
        if len(events['x']) < 10:
            continue
        input_currents = events_to_spike_input(events, n_input, duration_ms, dt_ms)
        result = lsm.run(input_currents, return_type='traces')
        feat = result['traces'].mean(dim=0)
        features_list.append(feat)
        labels_list.append(label)

    if not features_list:
        return torch.zeros(0, lsm.N_s), torch.zeros(0, dtype=torch.long)

    return torch.stack(features_list), torch.tensor(labels_list, dtype=torch.long)


def run_experiment(seed=0):
    print(f"\n{'='*60}")
    print(f"N-Caltech101 Experiment — Seed {seed}")
    print(f"{'='*60}")

    # ── Load data ────────────────────────────────────────────────────────────
    print("\nLoading N-Caltech101 data...")
    n_classes = NCALTECH_CFG['n_classes_subset']
    samples, class_names = load_ncaltech101(n_classes=n_classes, max_samples_per_class=50)

    # Partition into federated clients
    n_clients = NCALTECH_CFG['n_clients']
    alpha = NCALTECH_CFG['dirichlet_alpha']
    client_samples = partition_dirichlet(samples, n_clients, alpha, seed=seed)

    # Remove empty clients
    client_samples = [cs for cs in client_samples if len(cs) > 0]
    K = len(client_samples)
    print(f"  {K} non-empty clients")

    # Split each client into train/test (80/20)
    rng = np.random.RandomState(seed)
    train_samples = []
    test_samples = []
    for cs in client_samples:
        rng.shuffle(cs)
        n_test = max(1, int(0.2 * len(cs)))
        test_samples.append(cs[:n_test])
        train_samples.append(cs[n_test:])

    results = {}

    # ── LSM ──────────────────────────────────────────────────────────────────
    print("\nInitializing LSM...")
    n_input = 128
    lsm = LSM(input_dim=n_input, n_neurons=LSM_CFG['N_s'],
              output_dim=n_classes,
              tau_m=LSM_CFG['tau_m'], V_th=LSM_CFG['V_th'],
              connectivity=LSM_CFG['connectivity'],
              dt=LSM_CFG['dt'], trace_decay=LSM_CFG['trace_decay'],
              seed=seed)
    lam = LSM_CFG['ridge_lambda']

    # ── Process through LSM ──────────────────────────────────────────────────
    print("Processing through LSM...")
    train_features, train_labels = [], []
    test_features, test_labels = [], []
    for k in range(K):
        if k % 10 == 0:
            print(f"  Client {k}/{K}...")
        tr_f, tr_l = process_ncal_with_lsm(lsm, train_samples[k], n_input)
        te_f, te_l = process_ncal_with_lsm(lsm, test_samples[k], n_input)
        train_features.append(tr_f)
        train_labels.append(tr_l)
        test_features.append(te_f)
        test_labels.append(te_l)

    train_labels_oh = [labels_to_onehot(y, n_classes).to(DEVICE) for y in train_labels]

    # ── Local-only ───────────────────────────────────────────────────────────
    print("\n[1] Local-only...")
    local_accs = []
    for k in range(K):
        if len(train_features[k]) < 2 or len(test_features[k]) == 0:
            continue
        W_k = ridge_regression(train_features[k], train_labels_oh[k], lam)
        acc, _ = evaluate_readout(W_k, test_features[k], test_labels[k])
        local_accs.append(acc)
    results['local_only'] = {'acc': np.mean(local_accs) if local_accs else 0.0}
    print(f"  Acc: {results['local_only']['acc']:.4f}")

    # ── Centralized ──────────────────────────────────────────────────────────
    print("\n[2] Centralized...")
    valid_k = [k for k in range(K)
               if len(train_features[k]) > 0 and len(test_features[k]) > 0]
    all_feat = torch.cat([train_features[k] for k in valid_k])
    all_lab = torch.cat([train_labels_oh[k] for k in valid_k])
    W_cent = ridge_regression(all_feat, all_lab, lam)
    cent_accs = []
    for k in valid_k:
        acc, _ = evaluate_readout(W_cent, test_features[k], test_labels[k])
        cent_accs.append(acc)
    results['centralized'] = {'acc': np.mean(cent_accs) if cent_accs else 0.0}
    print(f"  Acc: {results['centralized']['acc']:.4f}")

    # ── FRI-LSM (stats) ─────────────────────────────────────────────────────
    print("\n[3] FRI-LSM (stats)...")
    valid_train = [train_features[k] for k in valid_k]
    valid_oh = [train_labels_oh[k] for k in valid_k]
    fri_s = FRIStatsAggregator(lam)
    W_s, comm_s = fri_s.run(valid_train, valid_oh)
    s_accs = []
    for k in valid_k:
        acc, _ = evaluate_readout(W_s, test_features[k], test_labels[k])
        s_accs.append(acc)
    results['fri_lsm_stats'] = {
        'acc': np.mean(s_accs) if s_accs else 0.0,
        'comm_mb': comm_cost_mb(comm_s),
    }
    print(f"  Acc: {results['fri_lsm_stats']['acc']:.4f}")

    # ── FRI-LSM (stats + pers) ───────────────────────────────────────────────
    print("\n[4] FRI-LSM (stats+pers)...")
    _, W_pers, comm_p = fri_s.run_with_personalization(
        valid_train, valid_oh, mu=FED_CFG['personalization_mu'])
    p_accs = []
    for i, k in enumerate(valid_k):
        acc, _ = evaluate_readout(W_pers[i], test_features[k], test_labels[k])
        p_accs.append(acc)
    results['fri_lsm_stats_pers'] = {
        'acc': np.mean(p_accs) if p_accs else 0.0,
        'comm_mb': comm_cost_mb(comm_p),
    }
    print(f"  Acc: {results['fri_lsm_stats_pers']['acc']:.4f}")

    # ── FRI-LSM (sketched) ─────────────────────────────────────────────────
    print("\n[5] FRI-LSM (sketched)...")
    import gc
    for m_sketch in [100, 50]:
        d_phi = train_features[0].shape[1] if len(train_features[0]) > 0 else LSM_CFG['N_s']
        rng_s = np.random.RandomState(seed)
        S = torch.from_numpy(
            (rng_s.randn(m_sketch, d_phi) / np.sqrt(m_sketch)).astype(np.float32)
        ).to(DEVICE)
        sk_train = [f.float().to(DEVICE) @ S.T if len(f) > 0
                     else torch.zeros(0, m_sketch, device=DEVICE) for f in train_features]
        sk_test = [f.float().to(DEVICE) @ S.T if len(f) > 0
                    else torch.zeros(0, m_sketch, device=DEVICE) for f in test_features]
        sk_oh = [labels_to_onehot(y, n_classes).to(DEVICE) for y in train_labels]
        valid_sk = [k for k in range(K) if len(sk_train[k]) > 0]
        fri_sk = FRIStatsAggregator(lam)
        W_sk, comm_sk = fri_sk.run([sk_train[k] for k in valid_sk],
                                    [sk_oh[k] for k in valid_sk])
        sk_accs = []
        for k in valid_sk:
            if len(sk_test[k]) > 0:
                acc, _ = evaluate_readout(W_sk, sk_test[k], test_labels[k])
                sk_accs.append(acc)
        results[f'fri_lsm_sketched_m{m_sketch}'] = {
            'acc': np.mean(sk_accs) if sk_accs else 0.0,
            'comm_mb': comm_cost_mb(comm_sk),
        }
        print(f"  m={m_sketch}: Acc={results[f'fri_lsm_sketched_m{m_sketch}']['acc']:.4f}")

    # Free LSM features
    del train_features, test_features, train_labels_oh
    gc.collect()
    torch.cuda.empty_cache()

    # ── Frame-based baselines ──────────────────────────────────────────────
    print("\n[6] Converting to frames for deep baselines...")
    frame_train_X, frame_train_y = [], []
    frame_test_X, frame_test_y = [], []
    sensor_size = NCALTECH_CFG['sensor_size']
    for k in range(K):
        tr_frames, tr_labels_f = [], []
        for events, label in train_samples[k]:
            if len(events['x']) < 10:
                continue
            frames = events_to_frames(events, sensor_size, time_window_us=100000)
            frames_t = torch.from_numpy(frames)
            if frames_t.shape[2] > 32:
                frames_t = torch.nn.functional.avg_pool2d(frames_t, kernel_size=4)
            n_f = frames_t.shape[0]
            feat_dim = frames_t.shape[1] * frames_t.shape[2] * frames_t.shape[3]
            tr_frames.append(frames_t.reshape(n_f, feat_dim))
            tr_labels_f.append(label)
        te_frames, te_labels_f = [], []
        for events, label in test_samples[k]:
            if len(events['x']) < 10:
                continue
            frames = events_to_frames(events, sensor_size, time_window_us=100000)
            frames_t = torch.from_numpy(frames)
            if frames_t.shape[2] > 32:
                frames_t = torch.nn.functional.avg_pool2d(frames_t, kernel_size=4)
            n_f = frames_t.shape[0]
            feat_dim = frames_t.shape[1] * frames_t.shape[2] * frames_t.shape[3]
            te_frames.append(frames_t.reshape(n_f, feat_dim))
            te_labels_f.append(label)
        if tr_frames and te_frames:
            max_len = min(100, max(s.shape[0] for s in tr_frames + te_frames))
            d_f = tr_frames[0].shape[1]
            def pad_seq(seqs, ml, d):
                out = torch.zeros(len(seqs), ml, d)
                for i, s in enumerate(seqs):
                    L = min(s.shape[0], ml)
                    out[i, :L] = s[:L]
                return out
            frame_train_X.append(pad_seq(tr_frames, max_len, d_f))
            frame_train_y.append(torch.tensor(tr_labels_f, dtype=torch.long))
            frame_test_X.append(pad_seq(te_frames, max_len, d_f))
            frame_test_y.append(torch.tensor(te_labels_f, dtype=torch.long))

    if frame_train_X:
        frame_dim = frame_train_X[0].shape[2]
        print(f"  Frame clients: {len(frame_train_X)}, dim: {frame_dim}")

        # FedAvg-LSTM
        print("\n[7] FedAvg-LSTM...")
        def make_lstm():
            return LSTMClassifier(frame_dim, LSTM_CFG['hidden_dim'], n_classes,
                                  LSTM_CFG['num_layers'], LSTM_CFG['dropout'])
        try:
            _, lstm_m = run_fedavg(
                make_lstm, frame_train_X, frame_train_y,
                frame_test_X, frame_test_y,
                num_rounds=FED_CFG['num_rounds'],
                participation_rate=FED_CFG['participation_ncal'],
                local_epochs=LSTM_CFG['local_epochs'],
                lr=LSTM_CFG['lr'], batch_size=LSTM_CFG['batch_size'], seed=seed)
            results['fedavg_lstm'] = {
                'acc': lstm_m['acc'], 'comm_mb': comm_cost_mb(lstm_m['comm_scalars']),
                'bptt': True}
            print(f"  Acc: {lstm_m['acc']:.4f}")
        except Exception as e:
            print(f"  FedAvg-LSTM failed: {e}")

        # FedAvg-SNN
        print("\n[8] FedAvg-SNN...")
        def make_snn():
            return SNNClassifier(frame_dim, SNN_CFG['hidden_dim'], n_classes,
                                 SNN_CFG['num_steps'], SNN_CFG['beta'])
        try:
            _, snn_m = run_fedavg(
                make_snn, frame_train_X, frame_train_y,
                frame_test_X, frame_test_y,
                num_rounds=FED_CFG['num_rounds'],
                participation_rate=FED_CFG['participation_ncal'],
                local_epochs=SNN_CFG['local_epochs'],
                lr=SNN_CFG['lr'], batch_size=SNN_CFG['batch_size'], seed=seed)
            results['fedavg_snn'] = {
                'acc': snn_m['acc'], 'comm_mb': comm_cost_mb(snn_m['comm_scalars']),
                'bptt': True}
            print(f"  Acc: {snn_m['acc']:.4f}")
        except Exception as e:
            print(f"  FedAvg-SNN failed: {e}")

        # FedProx-SNN
        print("\n[9] FedProx-SNN...")
        try:
            _, prox_m = run_fedavg(
                make_snn, frame_train_X, frame_train_y,
                frame_test_X, frame_test_y,
                num_rounds=FED_CFG['num_rounds'],
                participation_rate=FED_CFG['participation_ncal'],
                local_epochs=SNN_CFG['local_epochs'],
                lr=SNN_CFG['lr'], batch_size=SNN_CFG['batch_size'],
                fedprox_mu=0.01, seed=seed)
            results['fedprox_snn'] = {
                'acc': prox_m['acc'], 'comm_mb': comm_cost_mb(prox_m['comm_scalars']),
                'bptt': True}
            print(f"  Acc: {prox_m['acc']:.4f}")
        except Exception as e:
            print(f"  FedProx-SNN failed: {e}")

        # LFNL
        print("\n[10] LFNL...")
        try:
            _, lfnl_m = run_lfnl(
                make_snn, frame_train_X, frame_train_y,
                frame_test_X, frame_test_y,
                num_rounds=FED_CFG['num_rounds'],
                participation_rate=FED_CFG['participation_ncal'],
                local_epochs=SNN_CFG['local_epochs'],
                lr=SNN_CFG['lr'], batch_size=SNN_CFG['batch_size'], seed=seed)
            results['lfnl'] = {
                'acc': lfnl_m['acc'], 'comm_mb': comm_cost_mb(lfnl_m['comm_scalars']),
                'bptt': True}
            print(f"  Acc: {lfnl_m['acc']:.4f}")
        except Exception as e:
            print(f"  LFNL failed: {e}")

    return results


def main():
    all_results = {}
    for seed in SEEDS:
        all_results[seed] = run_experiment(seed)

    print("\n" + "=" * 60)
    print("AGGREGATED RESULTS")
    print("=" * 60)
    for m in ['local_only', 'centralized', 'fri_lsm_stats', 'fri_lsm_stats_pers',
             'fri_lsm_sketched_m100', 'fri_lsm_sketched_m50',
             'fedavg_lstm', 'fedavg_snn', 'fedprox_snn', 'lfnl']:
        accs = [all_results[s][m]['acc'] for s in SEEDS if m in all_results[s]]
        if accs:
            print(f"  {m:25s}: {np.mean(accs):.4f} ± {1.96*np.std(accs)/np.sqrt(len(accs)):.4f}")

    save_path = os.path.join(RESULTS_DIR, 'ncaltech101_results.json')
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
