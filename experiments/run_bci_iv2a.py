"""
Experiment 1: Federated BCI using BCI Competition IV-2a.
9 subjects × 4-class motor imagery × 2 sessions.
Each subject is a federated client.

Fills: Table 1 (BCI-IV-2a columns), Table 2 (session transfer)
"""
import sys, os, json, time
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, os.path.dirname(__file__))
from config import *
from reservoir import ESN, ridge_regression, compute_sufficient_statistics, \
    ridge_from_statistics, discounted_statistics, personalized_readout
from federated import FRIReadoutAggregator, FRIStatsAggregator, FRIDriftAggregator, \
    evaluate_readout, comm_cost_mb
from data_bci import load_bci2a_all, bci2a_to_federated
from baselines import (LSTMClassifier, EEGNet, run_fedavg, run_pfedme,
                       run_fedtl_eeg, train_local, evaluate_model)


def labels_to_onehot(labels, n_classes):
    oh = torch.zeros(len(labels), n_classes)
    oh[torch.arange(len(labels)), labels] = 1.0
    return oh


def run_esn_on_clients(esn, client_data, washout):
    """Run ESN on each client's trial data. Extract [mean, log-var] features.
    Variance captures band-power (ERD/ERS) and mean captures temporal bias."""
    client_features = []
    for X in client_data:
        # X: (n_trials, T, d_x)
        states = esn.run(X, washout=washout)  # (n_trials, T-washout, N_r)
        feat_mean = states.mean(dim=1)         # temporal average
        feat_var = torch.log(states.var(dim=1) + 1e-8)  # log-variance ≈ log band power
        trial_features = torch.cat([feat_mean, feat_var], dim=1)  # (n_trials, 2*N_r)
        client_features.append(trial_features)
    return client_features


def split_train_test(client_data, client_labels, test_ratio=0.2, seed=0):
    """Split each client's data into train/test."""
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
    """Run full BCI-IV-2a experiment for one seed."""
    print(f"\n{'='*60}")
    print(f"BCI-IV-2a Experiment — Seed {seed}")
    print(f"{'='*60}")

    # ── Load data ────────────────────────────────────────────────────────────
    print("\nLoading BCI-IV-2a training session data...")
    subjects_train = load_bci2a_all(session='T')
    K = len(subjects_train)
    print(f"  Loaded {K} subjects")

    # Convert to federated format
    client_data_all, client_labels_all = bci2a_to_federated(subjects_train)

    # Split training data into train/test
    train_X, test_X, train_y, test_y = split_train_test(
        client_data_all, client_labels_all, test_ratio=0.2, seed=seed)

    n_classes = BCI2A_CFG['n_classes']
    # With multiband: 22 channels x 2 sub-bands (mu 8-13 Hz, beta 13-30 Hz) = 44 input dims
    d_x = train_X[0].shape[2]
    print(f"  Input dimension: {d_x} (multiband)")

    results = {}

    # ── ESN setup ────────────────────────────────────────────────────────────
    print("\nInitializing ESN reservoir...")
    esn = ESN(
        input_dim=d_x,
        reservoir_dim=ESN_CFG['N_r'],
        output_dim=n_classes,
        spectral_radius=ESN_CFG['spectral_radius'],
        leaking_rate=ESN_CFG['leaking_rate'],
        input_scaling=ESN_CFG['input_scaling'],
        sparsity=ESN_CFG['sparsity'],
        seed=seed,
    )
    washout = ESN_CFG['washout']
    lam = ESN_CFG['ridge_lambda']

    # ── Run ESN on all clients ───────────────────────────────────────────────
    print("Running ESN on training data...")
    train_features = run_esn_on_clients(esn, train_X, washout)
    test_features = run_esn_on_clients(esn, test_X, washout)

    # Convert labels to one-hot
    train_labels_oh = [labels_to_onehot(y, n_classes).to(DEVICE) for y in train_y]
    test_labels_int = [y for y in test_y]

    # ── 1. Local-only ────────────────────────────────────────────────────────
    print("\n[1] Local-only ESN...")
    local_accs, local_f1s = [], []
    for k in range(K):
        W_k = ridge_regression(train_features[k], train_labels_oh[k], lam)
        acc, f1 = evaluate_readout(W_k, test_features[k], test_labels_int[k])
        local_accs.append(acc)
        local_f1s.append(f1)
    results['local_only'] = {
        'acc': np.mean(local_accs), 'acc_std': np.std(local_accs),
        'f1': np.mean(local_f1s), 'f1_std': np.std(local_f1s),
        'comm_mb': 0.0,
        'per_subject_acc': local_accs,
    }
    print(f"  Acc: {results['local_only']['acc']:.4f} ± {results['local_only']['acc_std']:.4f}")

    # ── 2. Centralized ESN ───────────────────────────────────────────────────
    print("\n[2] Centralized ESN...")
    all_train_feat = torch.cat(train_features, dim=0)
    all_train_lab = torch.cat(train_labels_oh, dim=0)
    W_central = ridge_regression(all_train_feat, all_train_lab, lam)

    cent_accs, cent_f1s = [], []
    for k in range(K):
        acc, f1 = evaluate_readout(W_central, test_features[k], test_labels_int[k])
        cent_accs.append(acc)
        cent_f1s.append(f1)
    results['centralized'] = {
        'acc': np.mean(cent_accs), 'acc_std': np.std(cent_accs),
        'f1': np.mean(cent_f1s), 'f1_std': np.std(cent_f1s),
        'comm_mb': comm_cost_mb(all_train_feat.shape[0] * (d_x + n_classes)),
    }
    print(f"  Acc: {results['centralized']['acc']:.4f} ± {results['centralized']['acc_std']:.4f}")

    # ── 3. FRI-ESN (readout-based) ──────────────────────────────────────────
    print("\n[3] FRI-ESN (readout-based)...")
    fri_readout = FRIReadoutAggregator(
        K, FED_CFG['participation_bci2a'], lam, FED_CFG['num_rounds'])
    W_readout, comm_read, _ = fri_readout.run(train_features, train_labels_oh, seed=seed)

    read_accs, read_f1s = [], []
    for k in range(K):
        acc, f1 = evaluate_readout(W_readout, test_features[k], test_labels_int[k])
        read_accs.append(acc)
        read_f1s.append(f1)
    results['fri_readout'] = {
        'acc': np.mean(read_accs), 'acc_std': np.std(read_accs),
        'f1': np.mean(read_f1s), 'f1_std': np.std(read_f1s),
        'comm_mb': comm_cost_mb(comm_read),
    }
    print(f"  Acc: {results['fri_readout']['acc']:.4f}, Comm: {results['fri_readout']['comm_mb']:.4f} MB")

    # ── 4. FRI-ESN (stats-based) ────────────────────────────────────────────
    print("\n[4] FRI-ESN (statistics-based)...")
    fri_stats = FRIStatsAggregator(lam)
    W_stats, comm_stats = fri_stats.run(train_features, train_labels_oh)

    stats_accs, stats_f1s = [], []
    for k in range(K):
        acc, f1 = evaluate_readout(W_stats, test_features[k], test_labels_int[k])
        stats_accs.append(acc)
        stats_f1s.append(f1)
    results['fri_stats'] = {
        'acc': np.mean(stats_accs), 'acc_std': np.std(stats_accs),
        'f1': np.mean(stats_f1s), 'f1_std': np.std(stats_f1s),
        'comm_mb': comm_cost_mb(comm_stats),
    }
    print(f"  Acc: {results['fri_stats']['acc']:.4f}, Comm: {results['fri_stats']['comm_mb']:.4f} MB")

    # ── 5. FRI-ESN (stats + personalization) ─────────────────────────────────
    print("\n[5] FRI-ESN (stats + personalization)...")
    mu = FED_CFG['personalization_mu']
    _, W_pers_list, comm_pers = fri_stats.run_with_personalization(
        train_features, train_labels_oh, mu=mu)

    pers_accs, pers_f1s = [], []
    for k in range(K):
        acc, f1 = evaluate_readout(W_pers_list[k], test_features[k], test_labels_int[k])
        pers_accs.append(acc)
        pers_f1s.append(f1)
    results['fri_stats_pers'] = {
        'acc': np.mean(pers_accs), 'acc_std': np.std(pers_accs),
        'f1': np.mean(pers_f1s), 'f1_std': np.std(pers_f1s),
        'comm_mb': comm_cost_mb(comm_pers),
        'per_subject_acc': pers_accs,
    }
    print(f"  Acc: {results['fri_stats_pers']['acc']:.4f}, Comm: {results['fri_stats_pers']['comm_mb']:.4f} MB")

    # ── 6. FedAvg-LSTM ──────────────────────────────────────────────────────
    print("\n[6] FedAvg-LSTM...")
    def make_lstm():
        return LSTMClassifier(d_x, LSTM_CFG['hidden_dim'], n_classes,
                              LSTM_CFG['num_layers'], LSTM_CFG['dropout'])

    _, lstm_metrics = run_fedavg(
        make_lstm, train_X, train_y, test_X, test_y,
        num_rounds=FED_CFG['num_rounds'],
        participation_rate=FED_CFG['participation_bci2a'],
        local_epochs=LSTM_CFG['local_epochs'],
        lr=LSTM_CFG['lr'], batch_size=LSTM_CFG['batch_size'],
        seed=seed)
    results['fedavg_lstm'] = {
        'acc': lstm_metrics['acc'], 'f1': lstm_metrics['f1'],
        'comm_mb': comm_cost_mb(lstm_metrics['comm_scalars']),
        'n_params': lstm_metrics['n_params'],
    }
    print(f"  Acc: {lstm_metrics['acc']:.4f}, Comm: {results['fedavg_lstm']['comm_mb']:.4f} MB")

    # ── 7. FedProx-LSTM ─────────────────────────────────────────────────────
    print("\n[7] FedProx-LSTM...")
    _, proxlstm_metrics = run_fedavg(
        make_lstm, train_X, train_y, test_X, test_y,
        num_rounds=FED_CFG['num_rounds'],
        participation_rate=FED_CFG['participation_bci2a'],
        local_epochs=LSTM_CFG['local_epochs'],
        lr=LSTM_CFG['lr'], batch_size=LSTM_CFG['batch_size'],
        fedprox_mu=0.01, seed=seed)
    results['fedprox_lstm'] = {
        'acc': proxlstm_metrics['acc'], 'f1': proxlstm_metrics['f1'],
        'comm_mb': comm_cost_mb(proxlstm_metrics['comm_scalars']),
    }
    print(f"  Acc: {proxlstm_metrics['acc']:.4f}")

    # ── 8. FedAvg-EEGNet ────────────────────────────────────────────────────
    print("\n[8] FedAvg-EEGNet...")
    n_times = train_X[0].shape[1]
    n_ch_eeg = d_x  # multiband channels
    # EEGNet expects (batch, n_channels, n_times) — transpose from (batch, T, d_x)
    eegnet_train_X = [x.transpose(1, 2) for x in train_X]  # (batch, d_x, T)
    eegnet_test_X = [x.transpose(1, 2) for x in test_X]

    def make_eegnet():
        return EEGNet(n_ch_eeg, n_times, n_classes,
                      EEGNET_CFG['F1'], EEGNET_CFG['D'],
                      EEGNET_CFG['F2'], EEGNET_CFG['dropout'])

    try:
        _, eegnet_metrics = run_fedavg(
            make_eegnet, eegnet_train_X, train_y, eegnet_test_X, test_y,
            num_rounds=FED_CFG['num_rounds'],
            participation_rate=FED_CFG['participation_bci2a'],
            local_epochs=EEGNET_CFG['local_epochs'],
            lr=EEGNET_CFG['lr'], batch_size=EEGNET_CFG['batch_size'],
            seed=seed)
        results['fedavg_eegnet'] = {
            'acc': eegnet_metrics['acc'], 'f1': eegnet_metrics['f1'],
            'comm_mb': comm_cost_mb(eegnet_metrics['comm_scalars']),
            'n_params': eegnet_metrics['n_params'],
        }
        print(f"  Acc: {eegnet_metrics['acc']:.4f}")
    except Exception as e:
        print(f"  EEGNet failed: {e}")
        results['fedavg_eegnet'] = {'acc': 0.0, 'f1': 0.0, 'comm_mb': 0.0}

    # ── 9. pFedMe (Dinh et al., NeurIPS 2020) ───────────────────────────
    print("\n[9] pFedMe (LSTM)...")
    _, pfedme_metrics = run_pfedme(
        make_lstm, train_X, train_y, test_X, test_y,
        num_rounds=FED_CFG['num_rounds'],
        participation_rate=FED_CFG['participation_bci2a'],
        local_epochs=LSTM_CFG['local_epochs'],
        lr=LSTM_CFG['lr'], lambd=0.1, beta=1.0,
        batch_size=LSTM_CFG['batch_size'], seed=seed)
    results['per_fedavg'] = {
        'acc': pfedme_metrics['acc'], 'f1': pfedme_metrics['f1'],
        'comm_mb': comm_cost_mb(pfedme_metrics['comm_scalars']),
    }
    print(f"  Acc: {pfedme_metrics['acc']:.4f}")

    # ── 10. FedTL-EEG ─────────────────────────────────────────────────────
    print("\n[10] FedTL-EEG (EEGNet)...")
    try:
        _, fedtl_metrics = run_fedtl_eeg(
            make_eegnet, eegnet_train_X, train_y, eegnet_test_X, test_y,
            num_rounds=FED_CFG['num_rounds'],
            participation_rate=FED_CFG['participation_bci2a'],
            local_epochs=EEGNET_CFG['local_epochs'],
            lr=EEGNET_CFG['lr'], batch_size=EEGNET_CFG['batch_size'],
            ft_epochs=10, ft_lr=5e-4, seed=seed)
        results['fedtl_eeg'] = {
            'acc': fedtl_metrics['acc'], 'f1': fedtl_metrics['f1'],
            'comm_mb': comm_cost_mb(fedtl_metrics['comm_scalars']),
        }
        print(f"  Acc: {fedtl_metrics['acc']:.4f}")
    except Exception as e:
        print(f"  FedTL-EEG failed: {e}")
        results['fedtl_eeg'] = {'acc': 0.0, 'f1': 0.0, 'comm_mb': 0.0}

    # ── 11. Session transfer / drift using BCI-IV-2b ────────────────────────
    # BCI-IV-2b has 9 subjects × 3 training sessions with labels.
    # Train on session 01T, test on 02T/03T to measure session drift.
    print("\n[11] Session transfer (BCI-IV-2b)...")
    try:
        from data_bci_extra import load_bci2b_subject
        drift_results = {}
        n_classes_2b = 2
        d_x_2b = 3  # bipolar EEG

        esn_2b = ESN(d_x_2b, ESN_CFG['N_r'], n_classes_2b,
                      spectral_radius=ESN_CFG['spectral_radius'],
                      leaking_rate=ESN_CFG['leaking_rate'],
                      input_scaling=ESN_CFG['input_scaling'],
                      sparsity=ESN_CFG['sparsity'], seed=seed)

        s1_all_accs = {f'beta_{b}': [] for b in ABLATION_CFG['forgetting_factors']}
        s2_all_accs = {f'beta_{b}': [] for b in ABLATION_CFG['forgetting_factors']}
        lstm_s2_accs = []

        for subj in range(1, 10):
            sess_data = load_bci2b_subject(subj, ['01T', '02T', '03T'])
            if '01T' not in sess_data or '02T' not in sess_data:
                continue

            # Session 1 data
            data1, labels1 = sess_data['01T']
            X1 = torch.from_numpy(data1.transpose(0, 2, 1)).float()
            mu1, std1 = X1.mean(dim=(0,1), keepdim=True), X1.std(dim=(0,1), keepdim=True) + 1e-8
            X1 = (X1 - mu1) / std1
            states1 = esn_2b.run(X1, washout=ESN_CFG['washout'])
            feat1 = torch.cat([states1.mean(dim=1), torch.log(states1.var(dim=1) + 1e-8)], dim=1)
            oh1 = labels_to_onehot(torch.from_numpy(labels1), n_classes_2b).to(DEVICE)

            # Session 2 data
            data2, labels2 = sess_data['02T']
            X2 = torch.from_numpy(data2.transpose(0, 2, 1)).float()
            X2 = (X2 - mu1) / std1  # normalize with session 1 stats
            states2 = esn_2b.run(X2, washout=ESN_CFG['washout'])
            feat2 = torch.cat([states2.mean(dim=1), torch.log(states2.var(dim=1) + 1e-8)], dim=1)
            lab2 = torch.from_numpy(labels2)

            # Session 3 if available (for additional transfer evaluation)
            feat3, lab3 = None, None
            if '03T' in sess_data:
                data3, labels3 = sess_data['03T']
                X3 = torch.from_numpy(data3.transpose(0, 2, 1)).float()
                X3 = (X3 - mu1) / std1
                states3 = esn_2b.run(X3, washout=ESN_CFG['washout'])
                feat3 = torch.cat([states3.mean(dim=1), torch.log(states3.var(dim=1) + 1e-8)], dim=1)
                lab3 = torch.from_numpy(labels3)

            for beta in ABLATION_CFG['forgetting_factors']:
                # Train on session 1, combine with session 2 using discounting
                combined_feat = torch.cat([feat1, feat2], dim=0)
                combined_oh = torch.cat([oh1,
                    labels_to_onehot(lab2, n_classes_2b).to(DEVICE)], dim=0)

                if beta < 1.0:
                    G, H, T_eff = discounted_statistics(combined_feat, combined_oh, beta)
                else:
                    G, H, T_eff = compute_sufficient_statistics(combined_feat, combined_oh)
                    T_eff = float(T_eff)

                W = ridge_from_statistics(G, H, T_eff, lam)

                # Evaluate on session 1 (to report)
                acc1, _ = evaluate_readout(W, feat1, torch.from_numpy(labels1))
                s1_all_accs[f'beta_{beta}'].append(acc1)

                # Evaluate on session 2
                acc2, _ = evaluate_readout(W, feat2, lab2)
                s2_all_accs[f'beta_{beta}'].append(acc2)

        for beta in ABLATION_CFG['forgetting_factors']:
            key = f'beta_{beta}'
            drift_results[key] = {
                'session1_acc': np.mean(s1_all_accs[key]) if s1_all_accs[key] else 0.0,
                'session2_acc': np.mean(s2_all_accs[key]) if s2_all_accs[key] else 0.0,
                'session2_std': np.std(s2_all_accs[key]) if s2_all_accs[key] else 0.0,
            }
            print(f"  β={beta}: S1={drift_results[key]['session1_acc']:.4f}, "
                  f"S2={drift_results[key]['session2_acc']:.4f}")

        # FedAvg-LSTM retrained baseline for drift
        # (Train LSTM on session 1+2, evaluate on session 2)
        print("  FedAvg-LSTM (retrained on sessions 1+2)...")
        lstm_drift_accs_s1, lstm_drift_accs_s2 = [], []
        for subj in range(1, 10):
            sess_data = load_bci2b_subject(subj, ['01T', '02T'])
            if '01T' not in sess_data or '02T' not in sess_data:
                continue
            data1, labels1 = sess_data['01T']
            data2, labels2 = sess_data['02T']
            X1 = torch.from_numpy(data1.transpose(0, 2, 1)).float()
            X2 = torch.from_numpy(data2.transpose(0, 2, 1)).float()
            mu = X1.mean(dim=(0,1), keepdim=True)
            std = X1.std(dim=(0,1), keepdim=True) + 1e-8
            X1 = (X1 - mu) / std
            X2 = (X2 - mu) / std
            y1 = torch.from_numpy(labels1)
            y2 = torch.from_numpy(labels2)

            lstm = LSTMClassifier(d_x_2b, LSTM_CFG['hidden_dim'], n_classes_2b,
                                   LSTM_CFG['num_layers'], LSTM_CFG['dropout']).to(DEVICE)
            X_combined = torch.cat([X1, X2])
            y_combined = torch.cat([y1, y2])
            train_local(lstm, X_combined, y_combined, epochs=20, lr=1e-3, batch_size=32)
            a1, _ = evaluate_model(lstm, X1, y1)
            a2, _ = evaluate_model(lstm, X2, y2)
            lstm_drift_accs_s1.append(a1)
            lstm_drift_accs_s2.append(a2)

        drift_results['lstm_retrained'] = {
            'session1_acc': np.mean(lstm_drift_accs_s1) if lstm_drift_accs_s1 else 0.0,
            'session2_acc': np.mean(lstm_drift_accs_s2) if lstm_drift_accs_s2 else 0.0,
        }
        print(f"  LSTM retrained: S1={drift_results['lstm_retrained']['session1_acc']:.4f}, "
              f"S2={drift_results['lstm_retrained']['session2_acc']:.4f}")

        results['drift'] = drift_results
    except Exception as e:
        print(f"  Drift experiment failed: {e}")
        import traceback; traceback.print_exc()

    # ── 10. Personalization sweep ────────────────────────────────────────────
    print("\n[10] Personalization sweep...")
    pers_sweep = {}
    for mu_val in ABLATION_CFG['personalization_mus']:
        if mu_val == 0:
            # Local-only
            accs = local_accs
        else:
            _, W_pers_mu, _ = fri_stats.run_with_personalization(
                train_features, train_labels_oh, mu=mu_val)
            accs = []
            for k in range(K):
                acc, _ = evaluate_readout(W_pers_mu[k], test_features[k], test_labels_int[k])
                accs.append(acc)
        pers_sweep[f'mu_{mu_val}'] = {
            'mean_acc': np.mean(accs),
            'std_acc': np.std(accs),
            'per_subject': accs,
        }
        print(f"  μ={mu_val}: acc={np.mean(accs):.4f} ± {np.std(accs):.4f}")

    results['personalization_sweep'] = pers_sweep

    return results


def main():
    all_results = {}
    for seed in SEEDS:
        all_results[seed] = run_experiment(seed)

    # Aggregate across seeds
    print("\n" + "=" * 60)
    print("AGGREGATED RESULTS (mean ± 95% CI across seeds)")
    print("=" * 60)

    methods = ['local_only', 'centralized', 'fri_readout', 'fri_stats',
               'fri_stats_pers', 'fedavg_lstm', 'fedprox_lstm', 'fedavg_eegnet',
               'per_fedavg', 'fedtl_eeg']
    for m in methods:
        accs = [all_results[s][m]['acc'] for s in SEEDS if m in all_results[s]]
        if accs:
            mean_acc = np.mean(accs)
            ci = 1.96 * np.std(accs) / np.sqrt(len(accs))
            comm = all_results[SEEDS[0]][m].get('comm_mb', 0)
            print(f"  {m:25s}: {mean_acc:.4f} ± {ci:.4f}  |  {comm:.4f} MB")

    # Save results
    save_path = os.path.join(RESULTS_DIR, 'bci_iv2a_results.json')
    # Convert numpy to python types for JSON
    def to_serializable(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_serializable(v) for v in obj]
        return obj

    with open(save_path, 'w') as f:
        json.dump(to_serializable(all_results), f, indent=2)
    print(f"\nResults saved to {save_path}")


if __name__ == '__main__':
    main()
