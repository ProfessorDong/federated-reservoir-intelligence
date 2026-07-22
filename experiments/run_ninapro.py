"""
Experiment 2: Federated EMG gesture recognition using Ninapro DB5.
10 subjects × 52 gestures × 16 sEMG channels.
Each subject is a federated client.

Fills: Table 1 (Ninapro DB5 columns)
"""
import sys, os, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from config import *
from reservoir import ESN, ridge_regression, compute_sufficient_statistics, \
    ridge_from_statistics, personalized_readout
from federated import FRIReadoutAggregator, FRIStatsAggregator, \
    evaluate_readout, comm_cost_mb
from data_bci import load_ninapro_db5_all, ninapro_to_federated
from baselines import LSTMClassifier, run_fedavg


def labels_to_onehot(labels, n_classes):
    oh = torch.zeros(len(labels), n_classes)
    oh[torch.arange(len(labels)), labels] = 1.0
    return oh


def run_esn_on_clients(esn, client_data, washout):
    """Run ESN on each client's segments, return [mean, log-var] features."""
    client_features = []
    for X in client_data:
        batch_size = 64
        all_feats = []
        for i in range(0, X.shape[0], batch_size):
            batch = X[i:i+batch_size]
            states = esn.run(batch, washout=washout)
            feat_mean = states.mean(dim=1)
            feat_var = torch.log(states.var(dim=1) + 1e-8)
            feats = torch.cat([feat_mean, feat_var], dim=1)
            all_feats.append(feats)
        client_features.append(torch.cat(all_feats, dim=0))
    return client_features


def split_train_test(client_data, client_labels, test_ratio=0.2, seed=0):
    train_X, test_X, train_y, test_y = [], [], [], []
    for X, y in zip(client_data, client_labels):
        n = len(y)
        rng = np.random.RandomState(seed)
        idx = rng.permutation(n)
        n_test = max(1, int(n * test_ratio))
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]
        train_X.append(X[train_idx])
        test_X.append(X[test_idx])
        train_y.append(y[train_idx])
        test_y.append(y[test_idx])
    return train_X, test_X, train_y, test_y


def run_experiment(seed=0):
    print(f"\n{'='*60}")
    print(f"Ninapro DB5 Experiment — Seed {seed}")
    print(f"{'='*60}")

    # ── Load data ────────────────────────────────────────────────────────────
    print("\nLoading Ninapro DB5 data...")
    subjects = load_ninapro_db5_all()
    K = len(subjects)
    print(f"  Loaded {K} subjects")

    # Determine actual number of classes
    all_labels = np.concatenate([s[1] for s in subjects])
    n_classes = len(np.unique(all_labels))
    print(f"  Total classes: {n_classes}")

    client_data, client_labels = ninapro_to_federated(subjects)

    # Split
    train_X, test_X, train_y, test_y = split_train_test(
        client_data, client_labels, test_ratio=0.2, seed=seed)

    d_x = NINAPRO_CFG['n_channels']
    results = {}

    # ── ESN ──────────────────────────────────────────────────────────────────
    print("\nInitializing ESN...")
    esn = ESN(input_dim=d_x, reservoir_dim=ESN_CFG['N_r'],
              output_dim=n_classes,
              spectral_radius=ESN_CFG['spectral_radius'],
              leaking_rate=ESN_CFG['leaking_rate'],
              input_scaling=ESN_CFG['input_scaling'],
              sparsity=ESN_CFG['sparsity'], seed=seed)
    washout = min(ESN_CFG['washout'], NINAPRO_CFG['window_size'] // 4)
    lam = ESN_CFG['ridge_lambda']

    print("Running ESN on training data...")
    train_features = run_esn_on_clients(esn, train_X, washout)
    test_features = run_esn_on_clients(esn, test_X, washout)

    train_labels_oh = [labels_to_onehot(y, n_classes).to(DEVICE) for y in train_y]
    test_labels_int = [y for y in test_y]

    # ── Local-only ───────────────────────────────────────────────────────────
    print("\n[1] Local-only...")
    local_accs = []
    for k in range(K):
        W_k = ridge_regression(train_features[k], train_labels_oh[k], lam)
        acc, _ = evaluate_readout(W_k, test_features[k], test_labels_int[k])
        local_accs.append(acc)
    results['local_only'] = {'acc': np.mean(local_accs), 'comm_mb': 0.0}
    print(f"  Acc: {np.mean(local_accs):.4f}")

    # ── Centralized ──────────────────────────────────────────────────────────
    print("\n[2] Centralized ESN...")
    all_feat = torch.cat(train_features)
    all_lab = torch.cat(train_labels_oh)
    W_cent = ridge_regression(all_feat, all_lab, lam)
    cent_accs = []
    for k in range(K):
        acc, _ = evaluate_readout(W_cent, test_features[k], test_labels_int[k])
        cent_accs.append(acc)
    results['centralized'] = {
        'acc': np.mean(cent_accs),
        'comm_mb': comm_cost_mb(all_feat.shape[0] * (d_x + n_classes)),
    }
    print(f"  Acc: {np.mean(cent_accs):.4f}")

    # ── FRI-ESN (readout) ────────────────────────────────────────────────────
    print("\n[3] FRI-ESN (readout-based)...")
    fri_r = FRIReadoutAggregator(K, FED_CFG['participation_ninapro'],
                                  lam, FED_CFG['num_rounds'])
    W_r, comm_r, _ = fri_r.run(train_features, train_labels_oh, seed=seed)
    r_accs = []
    for k in range(K):
        acc, _ = evaluate_readout(W_r, test_features[k], test_labels_int[k])
        r_accs.append(acc)
    results['fri_readout'] = {
        'acc': np.mean(r_accs), 'comm_mb': comm_cost_mb(comm_r),
    }
    print(f"  Acc: {np.mean(r_accs):.4f}, Comm: {results['fri_readout']['comm_mb']:.4f} MB")

    # ── FRI-ESN (stats) ─────────────────────────────────────────────────────
    print("\n[4] FRI-ESN (stats)...")
    fri_s = FRIStatsAggregator(lam)
    W_s, comm_s = fri_s.run(train_features, train_labels_oh)
    s_accs = []
    for k in range(K):
        acc, _ = evaluate_readout(W_s, test_features[k], test_labels_int[k])
        s_accs.append(acc)
    results['fri_stats'] = {
        'acc': np.mean(s_accs), 'comm_mb': comm_cost_mb(comm_s),
    }
    print(f"  Acc: {np.mean(s_accs):.4f}")

    # ── FRI-ESN (stats + pers) ───────────────────────────────────────────────
    print("\n[5] FRI-ESN (stats+pers)...")
    _, W_pers, comm_p = fri_s.run_with_personalization(
        train_features, train_labels_oh, mu=FED_CFG['personalization_mu'])
    p_accs = []
    for k in range(K):
        acc, _ = evaluate_readout(W_pers[k], test_features[k], test_labels_int[k])
        p_accs.append(acc)
    results['fri_stats_pers'] = {
        'acc': np.mean(p_accs), 'comm_mb': comm_cost_mb(comm_p),
    }
    print(f"  Acc: {np.mean(p_accs):.4f}")

    # ── FedAvg-LSTM ──────────────────────────────────────────────────────────
    print("\n[6] FedAvg-LSTM...")
    def make_lstm():
        return LSTMClassifier(d_x, LSTM_CFG['hidden_dim'], n_classes,
                              LSTM_CFG['num_layers'], LSTM_CFG['dropout'])
    _, lstm_m = run_fedavg(
        make_lstm, train_X, train_y, test_X, test_y,
        num_rounds=FED_CFG['num_rounds'],
        participation_rate=FED_CFG['participation_ninapro'],
        local_epochs=LSTM_CFG['local_epochs'],
        lr=LSTM_CFG['lr'], batch_size=LSTM_CFG['batch_size'],
        seed=seed)
    results['fedavg_lstm'] = {
        'acc': lstm_m['acc'], 'comm_mb': comm_cost_mb(lstm_m['comm_scalars']),
    }
    print(f"  Acc: {lstm_m['acc']:.4f}")

    # ── FedProx-LSTM ─────────────────────────────────────────────────────────
    print("\n[7] FedProx-LSTM...")
    _, prox_m = run_fedavg(
        make_lstm, train_X, train_y, test_X, test_y,
        num_rounds=FED_CFG['num_rounds'],
        participation_rate=FED_CFG['participation_ninapro'],
        local_epochs=LSTM_CFG['local_epochs'],
        lr=LSTM_CFG['lr'], batch_size=LSTM_CFG['batch_size'],
        fedprox_mu=0.01, seed=seed)
    results['fedprox_lstm'] = {
        'acc': prox_m['acc'], 'comm_mb': comm_cost_mb(prox_m['comm_scalars']),
    }
    print(f"  Acc: {prox_m['acc']:.4f}")

    return results


def main():
    all_results = {}
    for seed in SEEDS:
        all_results[seed] = run_experiment(seed)

    # Aggregate
    print("\n" + "=" * 60)
    print("AGGREGATED RESULTS")
    print("=" * 60)
    methods = ['local_only', 'centralized', 'fri_readout', 'fri_stats',
               'fri_stats_pers', 'fedavg_lstm', 'fedprox_lstm']
    for m in methods:
        accs = [all_results[s][m]['acc'] for s in SEEDS if m in all_results[s]]
        if accs:
            print(f"  {m:25s}: {np.mean(accs):.4f} ± {1.96*np.std(accs)/np.sqrt(len(accs)):.4f}")

    save_path = os.path.join(RESULTS_DIR, 'ninapro_db5_results.json')
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
