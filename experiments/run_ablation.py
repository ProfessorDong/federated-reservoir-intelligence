"""
Ablation studies for the FRI paper.

1. Reservoir dimension N_r sweep                    (validates Thm 2)
2. Convergence: readout vs proximal vs stats         (validates Thm 3)
3. Participation rate sweep                          (validates Thm 3)
4. Heterogeneous reservoirs                          (validates Thm 4 / Cor 1)
5. Differential privacy (per-client noise on stats)  (validates Prop 2)
6. Generalization bound (sample-size sweep)          (validates Thm 6)
7. Privacy scaling (T_k sweep at fixed epsilon)      (validates Prop 2 1/T_k^2)
"""
import sys, os, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from config import *
from reservoir import (ESN, ridge_regression, compute_sufficient_statistics,
                       ridge_from_statistics, personalized_readout)
from federated import (FRIReadoutAggregator, FRIReadoutProximalAggregator,
                       FRIStatsAggregator, evaluate_readout, comm_cost_mb)
from data_bci import load_bci2a_all, bci2a_to_federated


def labels_to_onehot(labels, n_classes):
    oh = torch.zeros(len(labels), n_classes)
    oh[torch.arange(len(labels)), labels] = 1.0
    return oh


def run_esn_on_clients(esn, client_data, washout):
    """Run ESN and extract [mean, log-var] features."""
    client_features = []
    for X in client_data:
        states = esn.run(X, washout=washout)
        feat_mean = states.mean(dim=1)
        feat_var = torch.log(states.var(dim=1) + 1e-8)
        client_features.append(torch.cat([feat_mean, feat_var], dim=1))
    return client_features


def run_esn_on_single(esn, X, washout):
    """Run ESN on a single client's data and extract [mean, log-var] features."""
    states = esn.run(X, washout=washout)
    feat_mean = states.mean(dim=1)
    feat_var = torch.log(states.var(dim=1) + 1e-8)
    return torch.cat([feat_mean, feat_var], dim=1)


def split_data(client_data, client_labels, test_ratio=0.2, seed=0):
    train_X, test_X, train_y, test_y = [], [], [], []
    for X, y in zip(client_data, client_labels):
        n = len(y)
        rng = np.random.RandomState(seed)
        idx = rng.permutation(n)
        n_test = max(1, int(n * test_ratio))
        train_X.append(X[idx[n_test:]])
        test_X.append(X[idx[:n_test]])
        train_y.append(y[idx[n_test:]])
        test_y.append(y[idx[:n_test]])
    return train_X, test_X, train_y, test_y


def _make_esn(d_x, n_classes, seed, N_r=None):
    return ESN(d_x, N_r or ESN_CFG['N_r'], n_classes,
               spectral_radius=ESN_CFG['spectral_radius'],
               leaking_rate=ESN_CFG['leaking_rate'],
               input_scaling=ESN_CFG['input_scaling'],
               sparsity=ESN_CFG['sparsity'], seed=seed)


def _eval_all(W, test_feat, test_y):
    accs = []
    for k in range(len(test_feat)):
        acc, _ = evaluate_readout(W, test_feat[k], test_y[k])
        accs.append(acc)
    return np.mean(accs), np.std(accs)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Reservoir dimension sweep (Thm 2)
# ═══════════════════════════════════════════════════════════════════════════════
def ablation_reservoir_dim(subjects, seed=0):
    print("\n" + "=" * 60)
    print("Ablation 1: Reservoir Dimension")
    print("=" * 60)

    client_data, client_labels = bci2a_to_federated(subjects)
    train_X, test_X, train_y, test_y = split_data(client_data, client_labels, seed=seed)
    n_classes = 4
    d_x = train_X[0].shape[2]
    lam = ESN_CFG['ridge_lambda']

    results = {}
    for N_r in ABLATION_CFG['reservoir_dims']:
        esn = _make_esn(d_x, n_classes, seed, N_r)
        train_feat = run_esn_on_clients(esn, train_X, ESN_CFG['washout'])
        test_feat = run_esn_on_clients(esn, test_X, ESN_CFG['washout'])
        train_oh = [labels_to_onehot(y, n_classes).to(DEVICE) for y in train_y]

        fri_s = FRIStatsAggregator(lam)
        W_s, comm_s = fri_s.run(train_feat, train_oh)
        acc_m, acc_s = _eval_all(W_s, test_feat, test_y)

        results[N_r] = {'acc': acc_m, 'acc_std': acc_s,
                        'comm_mb': comm_cost_mb(comm_s)}
        print(f"  N_r={N_r}: acc={acc_m:.4f}")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Convergence: readout vs proximal vs stats (Thm 3)
# ═══════════════════════════════════════════════════════════════════════════════
def ablation_convergence(subjects, seed=0):
    print("\n" + "=" * 60)
    print("Ablation 2: Convergence Comparison")
    print("=" * 60)

    client_data, client_labels = bci2a_to_federated(subjects)
    train_X, test_X, train_y, test_y = split_data(client_data, client_labels, seed=seed)
    K = len(train_X)
    n_classes = 4
    d_x = train_X[0].shape[2]
    lam = ESN_CFG['ridge_lambda']

    esn = _make_esn(d_x, n_classes, seed)
    train_feat = run_esn_on_clients(esn, train_X, ESN_CFG['washout'])
    test_feat = run_esn_on_clients(esn, test_X, ESN_CFG['washout'])
    train_oh = [labels_to_onehot(y, n_classes).to(DEVICE) for y in train_y]

    # Stats-based: single round (reference)
    fri_s = FRIStatsAggregator(lam)
    W_s, _ = fri_s.run(train_feat, train_oh)
    stats_acc, _ = _eval_all(W_s, test_feat, test_y)

    # Centralized reference
    W_cent = ridge_regression(torch.cat(train_feat), torch.cat(train_oh), lam)
    cent_acc, _ = _eval_all(W_cent, test_feat, test_y)

    def _track_rounds(per_round_W):
        accs, dists = [], []
        for W_r in per_round_W:
            a, _ = _eval_all(W_r, test_feat, test_y)
            accs.append(a)
            dists.append(torch.norm(W_r - W_s).item())
        return accs, dists

    # Readout-based (vanilla)
    fri_r = FRIReadoutAggregator(K, 0.5, lam, num_rounds=50)
    _, _, per_round_W_r = fri_r.run(train_feat, train_oh, seed=seed)
    readout_accs, readout_dists = _track_rounds(per_round_W_r)

    # Proximal readout-based (Thm 3 variant)
    fri_p = FRIReadoutProximalAggregator(
        K, 0.5, lam, num_rounds=50, prox_mu=ABLATION_CFG['proximal_mu'])
    _, _, per_round_W_p = fri_p.run(train_feat, train_oh, seed=seed)
    prox_accs, prox_dists = _track_rounds(per_round_W_p)

    results = {
        'readout_per_round': readout_accs,
        'readout_distances': readout_dists,
        'proximal_per_round': prox_accs,
        'proximal_distances': prox_dists,
        'stats_acc': stats_acc,
        'centralized_acc': cent_acc,
    }
    print(f"  Stats (1 round): {stats_acc:.4f}")
    print(f"  Readout (round 50): {readout_accs[-1]:.4f}")
    print(f"  Proximal (round 50): {prox_accs[-1]:.4f}")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Participation rate sweep (Thm 3)
# ═══════════════════════════════════════════════════════════════════════════════
def ablation_participation(subjects, seed=0):
    print("\n" + "=" * 60)
    print("Ablation 3: Participation Rate")
    print("=" * 60)

    client_data, client_labels = bci2a_to_federated(subjects)
    train_X, test_X, train_y, test_y = split_data(client_data, client_labels, seed=seed)
    K = len(train_X)
    n_classes = 4
    lam = ESN_CFG['ridge_lambda']

    esn = _make_esn(train_X[0].shape[2], n_classes, seed)
    train_feat = run_esn_on_clients(esn, train_X, ESN_CFG['washout'])
    test_feat = run_esn_on_clients(esn, test_X, ESN_CFG['washout'])
    train_oh = [labels_to_onehot(y, n_classes).to(DEVICE) for y in train_y]

    results = {}
    for q in ABLATION_CFG['participation_rates']:
        fri_r = FRIReadoutAggregator(K, q, lam, num_rounds=50)
        W_r, comm_r, _ = fri_r.run(train_feat, train_oh, seed=seed)
        acc_m, acc_s = _eval_all(W_r, test_feat, test_y)
        results[q] = {'acc': acc_m, 'acc_std': acc_s,
                      'comm_mb': comm_cost_mb(comm_r)}
        print(f"  q={q}: acc={acc_m:.4f}")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Heterogeneous reservoirs (Thm 4 / Cor 1)  — FIXED: matching features
# ═══════════════════════════════════════════════════════════════════════════════
def ablation_heterogeneous(subjects, seed=0):
    print("\n" + "=" * 60)
    print("Ablation 4: Heterogeneous Reservoirs (FIXED)")
    print("=" * 60)

    client_data, client_labels = bci2a_to_federated(subjects)
    train_X, test_X, train_y, test_y = split_data(client_data, client_labels, seed=seed)
    K = len(train_X)
    n_classes = 4
    d_x = train_X[0].shape[2]
    lam = ESN_CFG['ridge_lambda']
    N_r = ESN_CFG['N_r']

    # Homogeneous baseline (shared reservoir)
    esn_shared = _make_esn(d_x, n_classes, seed)
    train_feat_hom = run_esn_on_clients(esn_shared, train_X, ESN_CFG['washout'])
    test_feat_hom = run_esn_on_clients(esn_shared, test_X, ESN_CFG['washout'])
    train_oh = [labels_to_onehot(y, n_classes).to(DEVICE) for y in train_y]

    fri_s = FRIStatsAggregator(lam)
    W_hom, _ = fri_s.run(train_feat_hom, train_oh)
    hom_acc, _ = _eval_all(W_hom, test_feat_hom, test_y)
    print(f"  Homogeneous: acc={hom_acc:.4f}")

    results = {'homogeneous': hom_acc}

    for spectral_spread in ABLATION_CFG['spectral_spreads']:
        train_feat_het, test_feat_het = [], []
        for k in range(K):
            sr = ESN_CFG['spectral_radius'] + np.random.RandomState(seed+k).uniform(
                -spectral_spread, spectral_spread)
            sr = np.clip(sr, 0.5, 0.99)
            esn_k = ESN(d_x, N_r, n_classes,
                        spectral_radius=sr,
                        leaking_rate=ESN_CFG['leaking_rate'],
                        input_scaling=ESN_CFG['input_scaling'],
                        sparsity=ESN_CFG['sparsity'],
                        seed=seed + k + 100)
            # FIXED: use [mean, logvar] features (matching homogeneous)
            train_feat_het.append(run_esn_on_single(esn_k, train_X[k], ESN_CFG['washout']))
            test_feat_het.append(run_esn_on_single(esn_k, test_X[k], ESN_CFG['washout']))

        # Measure feature misalignment Delta^2 for Thm 4 verification
        Gs = []
        for k in range(K):
            G_k = train_feat_het[k].T @ train_feat_het[k] / train_feat_het[k].shape[0]
            Gs.append(G_k)
        G_mean = torch.stack(Gs).mean(dim=0)
        delta_sq = torch.stack([(G - G_mean).pow(2).sum() for G in Gs]).mean().item()

        W_het, _ = fri_s.run(train_feat_het, train_oh)
        het_acc, het_std = _eval_all(W_het, test_feat_het, test_y)

        results[f'spread_{spectral_spread}'] = {
            'acc': het_acc, 'acc_std': het_std, 'delta_sq': delta_sq,
        }
        print(f"  spread={spectral_spread}: acc={het_acc:.4f}, Δ²={delta_sq:.2f}")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Differential privacy — per-client noise on statistics (Prop 2)  — FIXED
# ═══════════════════════════════════════════════════════════════════════════════
def ablation_privacy(subjects, seed=0):
    print("\n" + "=" * 60)
    print("Ablation 5: Differential Privacy (FIXED — per-client noise)")
    print("=" * 60)

    client_data, client_labels = bci2a_to_federated(subjects)
    train_X, test_X, train_y, test_y = split_data(client_data, client_labels, seed=seed)
    K = len(train_X)
    n_classes = 4
    lam = ESN_CFG['ridge_lambda']

    esn = _make_esn(train_X[0].shape[2], n_classes, seed)
    train_feat = run_esn_on_clients(esn, train_X, ESN_CFG['washout'])
    test_feat = run_esn_on_clients(esn, test_X, ESN_CFG['washout'])
    train_oh = [labels_to_onehot(y, n_classes).to(DEVICE) for y in train_y]
    d = train_feat[0].shape[1]

    # No-DP baseline
    fri_s = FRIStatsAggregator(lam)
    W_base, _ = fri_s.run(train_feat, train_oh)
    base_acc, _ = _eval_all(W_base, test_feat, test_y)
    results = {'no_dp': base_acc}
    print(f"  No DP: acc={base_acc:.4f}")

    B, B_y = 1.0, 1.0
    delta = ABLATION_CFG['dp_delta']
    c_delta = np.sqrt(2 * np.log(1.25 / delta))

    for epsilon in ABLATION_CFG['dp_epsilons']:
        G_agg = torch.zeros(d, d, device=DEVICE)
        H_agg = torch.zeros(n_classes, d, device=DEVICE)
        T_total = 0

        for k in range(K):
            G_k, H_k, T_k = compute_sufficient_statistics(train_feat[k], train_oh[k])
            # Per-client sensitivity and noise
            sens_G = B ** 2
            sens_H = B * B_y
            sigma_G = sens_G * c_delta / (epsilon * max(1.0, np.sqrt(T_k)))
            sigma_H = sens_H * c_delta / (epsilon * max(1.0, np.sqrt(T_k)))

            rng_dp = np.random.RandomState(seed * 1000 + k * 100 + int(epsilon * 10))
            noise_G = torch.from_numpy(
                (rng_dp.randn(d, d) * sigma_G).astype(np.float32)).to(DEVICE)
            noise_G = (noise_G + noise_G.T) / 2  # keep symmetric
            noise_H = torch.from_numpy(
                (rng_dp.randn(n_classes, d) * sigma_H).astype(np.float32)).to(DEVICE)

            G_agg += G_k + noise_G
            H_agg += H_k + noise_H
            T_total += T_k

        W_dp = ridge_from_statistics(G_agg, H_agg, T_total, lam)
        dp_acc, _ = _eval_all(W_dp, test_feat, test_y)
        results[f'eps_{epsilon}'] = {'acc': dp_acc, 'excess_risk': base_acc - dp_acc}
        print(f"  ε={epsilon}: acc={dp_acc:.4f}")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Generalization bound verification (Thm 6)  — NEW
# ═══════════════════════════════════════════════════════════════════════════════
def ablation_generalization(subjects, seed=0):
    print("\n" + "=" * 60)
    print("Ablation 6: Generalization Bound (NEW)")
    print("=" * 60)

    client_data, client_labels = bci2a_to_federated(subjects)
    train_X, test_X, train_y, test_y = split_data(client_data, client_labels, seed=seed)
    K = len(train_X)
    n_classes = 4
    lam = ESN_CFG['ridge_lambda']
    N_r = ESN_CFG['N_r']

    esn = _make_esn(train_X[0].shape[2], n_classes, seed)
    full_train_feat = run_esn_on_clients(esn, train_X, ESN_CFG['washout'])
    test_feat = run_esn_on_clients(esn, test_X, ESN_CFG['washout'])

    results = {}
    for frac in ABLATION_CFG['generalization_fracs']:
        sub_feat, sub_oh = [], []
        T_eff = 0
        for k in range(K):
            n_k = full_train_feat[k].shape[0]
            n_sub = max(2, int(frac * n_k))
            rng = np.random.RandomState(seed * 100 + k)
            idx = rng.choice(n_k, n_sub, replace=False)
            sub_feat.append(full_train_feat[k][idx])
            sub_oh.append(labels_to_onehot(train_y[k][idx], n_classes).to(DEVICE))
            T_eff += n_sub

        fri_s = FRIStatsAggregator(lam)
        W_sub, _ = fri_s.run(sub_feat, sub_oh)

        # Train accuracy (use same subsampled indices for labels)
        sub_y = []
        for k in range(K):
            n_k = full_train_feat[k].shape[0]
            n_sub = max(2, int(frac * n_k))
            rng_k = np.random.RandomState(seed * 100 + k)
            idx_k = rng_k.choice(n_k, n_sub, replace=False)
            sub_y.append(train_y[k][idx_k])
        train_acc, _ = _eval_all(W_sub, sub_feat, sub_y)
        # Test accuracy
        test_acc, _ = _eval_all(W_sub, test_feat, test_y)
        gen_gap = train_acc - test_acc
        theoretical = np.sqrt(N_r * np.log(N_r + 1) / max(1, T_eff))

        results[f'frac_{frac}'] = {
            'T_eff': T_eff, 'train_acc': train_acc,
            'test_acc': test_acc, 'gen_gap': gen_gap,
            'theoretical_bound': theoretical,
        }
        print(f"  frac={frac}: T_eff={T_eff}, train={train_acc:.4f}, "
              f"test={test_acc:.4f}, gap={gen_gap:.4f}")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Privacy scaling — T_k sweep at fixed epsilon (Prop 2)  — NEW
# ═══════════════════════════════════════════════════════════════════════════════
def ablation_privacy_scaling(subjects, seed=0):
    print("\n" + "=" * 60)
    print("Ablation 7: Privacy Scaling 1/T_k² (NEW)")
    print("=" * 60)

    client_data, client_labels = bci2a_to_federated(subjects)
    train_X, test_X, train_y, test_y = split_data(client_data, client_labels, seed=seed)
    K = len(train_X)
    n_classes = 4
    lam = ESN_CFG['ridge_lambda']
    d_feat = 2 * ESN_CFG['N_r']  # [mean, logvar]

    esn = _make_esn(train_X[0].shape[2], n_classes, seed)
    full_train_feat = run_esn_on_clients(esn, train_X, ESN_CFG['washout'])
    test_feat = run_esn_on_clients(esn, test_X, ESN_CFG['washout'])

    epsilon = ABLATION_CFG['privacy_scaling_epsilon']
    delta = ABLATION_CFG['dp_delta']
    B, B_y = 1.0, 1.0
    c_delta = np.sqrt(2 * np.log(1.25 / delta))

    results = {}
    for frac in ABLATION_CFG['generalization_fracs']:
        # Subsample each client
        sub_feat, sub_oh, sub_Tks = [], [], []
        for k in range(K):
            n_k = full_train_feat[k].shape[0]
            n_sub = max(2, int(frac * n_k))
            rng = np.random.RandomState(seed * 100 + k)
            idx = rng.choice(n_k, n_sub, replace=False)
            sub_feat.append(full_train_feat[k][idx])
            sub_oh.append(labels_to_onehot(train_y[k][idx], n_classes).to(DEVICE))
            sub_Tks.append(n_sub)

        T_k_avg = np.mean(sub_Tks)

        # Non-private
        fri_s = FRIStatsAggregator(lam)
        W_clean, _ = fri_s.run(sub_feat, sub_oh)
        clean_acc, _ = _eval_all(W_clean, test_feat, test_y)

        # Private (per-client noise calibrated to T_k)
        G_agg = torch.zeros(d_feat, d_feat, device=DEVICE)
        H_agg = torch.zeros(n_classes, d_feat, device=DEVICE)
        T_total = 0
        for k in range(K):
            G_k, H_k, T_k = compute_sufficient_statistics(sub_feat[k], sub_oh[k])
            sigma_G = B**2 * c_delta / (epsilon * max(1.0, np.sqrt(T_k)))
            sigma_H = B*B_y * c_delta / (epsilon * max(1.0, np.sqrt(T_k)))
            rng_dp = np.random.RandomState(seed * 1000 + k * 100 + int(frac * 100))
            nG = torch.from_numpy((rng_dp.randn(d_feat, d_feat)*sigma_G).astype(np.float32)).to(DEVICE)
            nG = (nG + nG.T) / 2
            nH = torch.from_numpy((rng_dp.randn(n_classes, d_feat)*sigma_H).astype(np.float32)).to(DEVICE)
            G_agg += G_k + nG
            H_agg += H_k + nH
            T_total += T_k

        W_dp = ridge_from_statistics(G_agg, H_agg, T_total, lam)
        dp_acc, _ = _eval_all(W_dp, test_feat, test_y)

        excess = clean_acc - dp_acc
        results[f'frac_{frac}'] = {
            'T_k_avg': T_k_avg, 'clean_acc': clean_acc,
            'dp_acc': dp_acc, 'excess_risk': excess,
            'theoretical_1_T2': 1.0 / (T_k_avg ** 2),
        }
        print(f"  frac={frac}: T_k_avg={T_k_avg:.0f}, clean={clean_acc:.4f}, "
              f"dp={dp_acc:.4f}, excess={excess:.4f}")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("\nLoading BCI-IV-2a data for ablation studies...")
    subjects = load_bci2a_all(session='T')

    all_results = {}
    for seed in ABLATION_SEEDS:
        print(f"\n{'#'*60}")
        print(f"# SEED {seed}")
        print(f"{'#'*60}")

        all_results[seed] = {
            'reservoir_dim': ablation_reservoir_dim(subjects, seed),
            'convergence': ablation_convergence(subjects, seed),
            'participation': ablation_participation(subjects, seed),
            'heterogeneous': ablation_heterogeneous(subjects, seed),
            'privacy': ablation_privacy(subjects, seed),
            'generalization': ablation_generalization(subjects, seed),
            'privacy_scaling': ablation_privacy_scaling(subjects, seed),
        }

    save_path = os.path.join(RESULTS_DIR, 'ablation_results.json')
    def to_ser(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): to_ser(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_ser(v) for v in obj]
        return obj
    with open(save_path, 'w') as f:
        json.dump(to_ser(all_results), f, indent=2)
    print(f"\nAblation results saved to {save_path}")


if __name__ == '__main__':
    main()
