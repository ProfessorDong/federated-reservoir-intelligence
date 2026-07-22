#!/usr/bin/env python3
"""
Master runner for all FRI experiments.

Usage:
    python run_all.py              # Run everything
    python run_all.py bci          # BCI-IV-2a only
    python run_all.py ninapro      # Ninapro DB5 only
    python run_all.py dvs          # DVS128 only
    python run_all.py ncaltech     # N-Caltech101 only
    python run_all.py ablation     # Ablation studies only
    python run_all.py bci2b        # BCI-IV-2b extra
    python run_all.py bci4         # BCI-IV-4 ECoG extra
"""
import sys, os, time, json, traceback
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


def run_with_timing(name, func):
    print(f"\n{'#'*70}")
    print(f"# {name}")
    print(f"{'#'*70}")
    t0 = time.time()
    try:
        func()
        elapsed = time.time() - t0
        print(f"\n✓ {name} completed in {elapsed/60:.1f} minutes")
        return True
    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n✗ {name} FAILED after {elapsed/60:.1f} minutes: {e}")
        traceback.print_exc()
        return False


def run_bci2b():
    """Extra experiment: BCI-IV-2b (2-class, 9 subjects, multi-session)."""
    from config import DEVICE, ESN_CFG, FED_CFG, RESULTS_DIR, SEEDS, ABLATION_CFG
    from reservoir import ESN, ridge_regression, compute_sufficient_statistics, \
        ridge_from_statistics, discounted_statistics
    from federated import FRIStatsAggregator, evaluate_readout, comm_cost_mb
    from data_bci_extra import load_bci2b_all, bci2b_to_federated, load_bci2b_subject
    import torch

    def labels_to_onehot(labels, n_classes):
        oh = torch.zeros(len(labels), n_classes)
        oh[torch.arange(len(labels)), labels] = 1.0
        return oh

    print("\nLoading BCI-IV-2b data...")
    subjects = load_bci2b_all()
    K = len(subjects)
    n_classes = 2
    d_x = 3  # 3 bipolar EEG channels

    client_data, client_labels = bci2b_to_federated(subjects)

    all_results = {}
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        rng = np.random.RandomState(seed)

        # Split
        train_X, test_X, train_y, test_y = [], [], [], []
        for X, y in zip(client_data, client_labels):
            n = len(y)
            idx = rng.permutation(n)
            n_test = max(1, int(0.2 * n))
            train_X.append(X[idx[n_test:]])
            test_X.append(X[idx[:n_test]])
            train_y.append(y[idx[n_test:]])
            test_y.append(y[idx[:n_test]])

        esn = ESN(d_x, ESN_CFG['N_r'], n_classes,
                  spectral_radius=ESN_CFG['spectral_radius'],
                  leaking_rate=ESN_CFG['leaking_rate'],
                  input_scaling=ESN_CFG['input_scaling'],
                  sparsity=ESN_CFG['sparsity'], seed=seed)
        washout = ESN_CFG['washout']
        lam = ESN_CFG['ridge_lambda']

        train_feat = [esn.run(X, washout).mean(dim=1) for X in train_X]
        test_feat = [esn.run(X, washout).mean(dim=1) for X in test_X]
        train_oh = [labels_to_onehot(y, n_classes).to(DEVICE) for y in train_y]

        # Local-only
        local_accs = []
        for k in range(K):
            W = ridge_regression(train_feat[k], train_oh[k], lam)
            acc, _ = evaluate_readout(W, test_feat[k], test_y[k])
            local_accs.append(acc)

        # FRI stats
        fri = FRIStatsAggregator(lam)
        W_s, comm = fri.run(train_feat, train_oh)
        s_accs = []
        for k in range(K):
            acc, _ = evaluate_readout(W_s, test_feat[k], test_y[k])
            s_accs.append(acc)

        # FRI stats + pers
        _, W_pers, _ = fri.run_with_personalization(train_feat, train_oh,
                                                     mu=FED_CFG['personalization_mu'])
        p_accs = []
        for k in range(K):
            acc, _ = evaluate_readout(W_pers[k], test_feat[k], test_y[k])
            p_accs.append(acc)

        all_results[seed] = {
            'local': np.mean(local_accs),
            'fri_stats': np.mean(s_accs),
            'fri_stats_pers': np.mean(p_accs),
            'comm_mb': comm_cost_mb(comm),
        }
        print(f"  Local: {np.mean(local_accs):.4f}, Stats: {np.mean(s_accs):.4f}, "
              f"Pers: {np.mean(p_accs):.4f}")

        # Multi-session drift analysis
        print("  Session drift analysis...")
        drift_results = {}
        for s_idx in range(1, K + 1):
            sess_data = load_bci2b_subject(s_idx, ['01T', '02T', '03T'])
            if len(sess_data) < 2:
                continue
            sessions = sorted(sess_data.keys())
            # Train on first session, test on later sessions
            s1_data, s1_labels = sess_data[sessions[0]]
            s1_X = torch.from_numpy(s1_data.transpose(0, 2, 1)).float()
            s1_mean = s1_X.mean(dim=(0,1), keepdim=True)
            s1_std = s1_X.std(dim=(0,1), keepdim=True) + 1e-8
            s1_X = (s1_X - s1_mean) / s1_std
            s1_feat = esn.run(s1_X, washout).mean(dim=1)
            s1_oh = labels_to_onehot(torch.from_numpy(s1_labels), n_classes).to(DEVICE)
            W1 = ridge_regression(s1_feat, s1_oh, lam)

            for si, sess in enumerate(sessions[1:], 1):
                s_data, s_labels = sess_data[sess]
                s_X = torch.from_numpy(s_data.transpose(0, 2, 1)).float()
                s_X = (s_X - s1_mean) / s1_std
                s_feat = esn.run(s_X, washout).mean(dim=1)
                acc, _ = evaluate_readout(W1, s_feat,
                                          torch.from_numpy(s_labels))
                drift_results[f'sub{s_idx}_s{sessions[0]}_to_s{sess}'] = acc

        all_results[seed]['drift'] = drift_results

    # Save
    save_path = os.path.join(RESULTS_DIR, 'bci_2b_results.json')
    def to_ser(obj):
        if isinstance(obj, (np.floating, np.integer)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {str(k): to_ser(v) for k, v in obj.items()}
        if isinstance(obj, list): return [to_ser(v) for v in obj]
        return obj
    with open(save_path, 'w') as f:
        json.dump(to_ser(all_results), f, indent=2)
    print(f"\nBCI-IV-2b results saved to {save_path}")


def run_bci4():
    """Extra experiment: BCI-IV-4 (ECoG, 3 subjects, 5-class finger)."""
    from config import DEVICE, ESN_CFG, RESULTS_DIR, SEEDS
    from reservoir import ESN, ridge_regression
    from federated import FRIStatsAggregator, evaluate_readout, comm_cost_mb
    from data_bci_extra import load_bci4_all, bci4_to_federated
    import torch

    def labels_to_onehot(labels, n_classes):
        oh = torch.zeros(len(labels), n_classes)
        oh[torch.arange(len(labels)), labels] = 1.0
        return oh

    print("\nLoading BCI-IV-4 (ECoG) data...")
    subjects = load_bci4_all()
    client_data, client_labels = bci4_to_federated(subjects)
    K = len(client_data)
    n_classes = 5
    d_x = client_data[0].shape[2]

    all_results = {}
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        rng = np.random.RandomState(seed)
        train_X, test_X, train_y, test_y = [], [], [], []
        for X, y in zip(client_data, client_labels):
            n = len(y)
            idx = rng.permutation(n)
            nt = max(1, int(0.2 * n))
            train_X.append(X[idx[nt:]])
            test_X.append(X[idx[:nt]])
            train_y.append(y[idx[nt:]])
            test_y.append(y[idx[:nt]])

        esn = ESN(d_x, ESN_CFG['N_r'], n_classes,
                  spectral_radius=ESN_CFG['spectral_radius'],
                  leaking_rate=ESN_CFG['leaking_rate'],
                  input_scaling=ESN_CFG['input_scaling'],
                  sparsity=ESN_CFG['sparsity'], seed=seed)
        washout = min(ESN_CFG['washout'], 50)
        lam = ESN_CFG['ridge_lambda']

        train_feat = []
        test_feat = []
        for X in train_X:
            # Process in batches (ECoG segments can be large)
            feats = []
            for i in range(0, X.shape[0], 32):
                s = esn.run(X[i:i+32], washout)
                feats.append(s.mean(dim=1))
            train_feat.append(torch.cat(feats))
        for X in test_X:
            feats = []
            for i in range(0, X.shape[0], 32):
                s = esn.run(X[i:i+32], washout)
                feats.append(s.mean(dim=1))
            test_feat.append(torch.cat(feats))

        train_oh = [labels_to_onehot(y, n_classes).to(DEVICE) for y in train_y]

        # FRI stats
        fri = FRIStatsAggregator(lam)
        W_s, comm = fri.run(train_feat, train_oh)
        accs = []
        for k in range(K):
            acc, _ = evaluate_readout(W_s, test_feat[k], test_y[k])
            accs.append(acc)

        # Local
        local_accs = []
        for k in range(K):
            W = ridge_regression(train_feat[k], train_oh[k], lam)
            acc, _ = evaluate_readout(W, test_feat[k], test_y[k])
            local_accs.append(acc)

        all_results[seed] = {
            'local': np.mean(local_accs),
            'fri_stats': np.mean(accs),
            'comm_mb': comm_cost_mb(comm),
        }
        print(f"  Local: {np.mean(local_accs):.4f}, Stats: {np.mean(accs):.4f}")

    save_path = os.path.join(RESULTS_DIR, 'bci_4_ecog_results.json')
    def to_ser(obj):
        if isinstance(obj, (np.floating, np.integer)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {str(k): to_ser(v) for k, v in obj.items()}
        if isinstance(obj, list): return [to_ser(v) for v in obj]
        return obj
    with open(save_path, 'w') as f:
        json.dump(to_ser(all_results), f, indent=2)
    print(f"\nBCI-IV-4 results saved to {save_path}")


def main():
    args = sys.argv[1:] if len(sys.argv) > 1 else ['all']
    tasks = args[0].lower()

    status = {}

    if tasks in ('all', 'bci'):
        import run_bci_iv2a
        status['bci_iv2a'] = run_with_timing('BCI-IV-2a', run_bci_iv2a.main)

    if tasks in ('all', 'bci2b'):
        status['bci_2b'] = run_with_timing('BCI-IV-2b', run_bci2b)

    if tasks in ('all', 'bci4'):
        status['bci_4'] = run_with_timing('BCI-IV-4 ECoG', run_bci4)

    if tasks in ('all', 'ninapro'):
        import run_ninapro
        status['ninapro'] = run_with_timing('Ninapro DB5', run_ninapro.main)

    if tasks in ('all', 'dvs'):
        import run_dvs128
        status['dvs128'] = run_with_timing('DVS128 Gesture', run_dvs128.main)

    if tasks in ('all', 'ncaltech'):
        import run_ncaltech101
        status['ncaltech'] = run_with_timing('N-Caltech101', run_ncaltech101.main)

    if tasks in ('all', 'ablation'):
        import run_ablation
        status['ablation'] = run_with_timing('Ablation Studies', run_ablation.main)

    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    for name, success in status.items():
        icon = "✓" if success else "✗"
        print(f"  {icon} {name}")


if __name__ == '__main__':
    main()
