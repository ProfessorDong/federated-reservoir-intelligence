"""
Self-contained synthetic demo of Federated Reservoir Intelligence (FRI).

Runs the full FRI pipeline on synthetic data with no external dataset required:
  1. ESN reservoir construction (fixed random weights, leaky integration)
  2. Local sufficient-statistics computation
  3. Statistics-based federated aggregation (one-shot exact recovery)
  4. Personalization with global-prior regularization

Expected runtime: 15-30 seconds on CPU.
"""
import os
# Force CPU for the demo to ensure portability and avoid CUDA-version mismatches
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import time
import numpy as np
import torch

from reservoir import ESN, ridge_regression
from federated import FRIStatsAggregator, evaluate_readout, comm_cost_mb


def make_synthetic_clients(K=4, T_k=240, d_x=8, d_y=3, T_test=60, seq_len=40, seed=0):
    """Generate K clients with class-specific sequence templates plus noise.

    Each trial is a (seq_len, d_x) sequence formed by repeating a class-specific
    template across time and adding Gaussian noise. The class-specific signal is
    visible in the trial-level mean; an ESN + mean-pool readout should solve it.
    """
    rng = np.random.default_rng(seed)
    # Shared class templates across clients
    templates = rng.standard_normal((d_y, d_x)).astype(np.float32) * 1.5

    train_X, train_y_oh, test_X, test_y = [], [], [], []
    for k in range(K):
        labels = rng.integers(0, d_y, size=T_k + T_test)
        Xk = np.zeros((T_k + T_test, seq_len, d_x), dtype=np.float32)
        for i, lbl in enumerate(labels):
            base = templates[lbl][None, :]  # (1, d_x)
            Xk[i] = base + 0.6 * rng.standard_normal((seq_len, d_x)).astype(np.float32)
        y_oh = np.eye(d_y, dtype=np.float32)[labels]
        train_X.append(torch.from_numpy(Xk[:T_k]))
        train_y_oh.append(torch.from_numpy(y_oh[:T_k]))
        test_X.append(torch.from_numpy(Xk[T_k:]))
        test_y.append(torch.from_numpy(labels[T_k:]))
    return train_X, train_y_oh, test_X, test_y


def main():
    K, T_k, d_x, d_y, N_r = 4, 240, 8, 3, 80
    seed = 0
    print("[FRI synthetic demo]")
    print(f"  K={K} clients, T_k={T_k} samples each, N_r={N_r} reservoir, d_y={d_y} outputs")
    print("  Generating synthetic time-series data with shared reservoir...\n")

    t0 = time.time()
    train_X, train_y_oh, test_X, test_y = make_synthetic_clients(
        K=K, T_k=T_k, d_x=d_x, d_y=d_y, seed=seed
    )

    # Shared ESN across all clients
    esn = ESN(input_dim=d_x, reservoir_dim=N_r, output_dim=d_y,
              spectral_radius=0.9, leaking_rate=0.5,
              input_scaling=0.5, sparsity=0.9, seed=seed)

    # Compute reservoir-state features per client (washout=10)
    def featurize(client_X):
        out = []
        for Xk in client_X:
            states = esn.run(Xk, washout=10)  # (n_samples, T_eff, N_r)
            feat = states.mean(dim=1)  # mean-pool per trial → (n_samples, N_r)
            out.append(feat)
        return out

    train_feat = featurize(train_X)
    test_feat = featurize(test_X)
    lam = 0.01

    # ── 1. Local-only ridge readout (no federation) ────────────────
    local_accs = []
    for k in range(K):
        W_k = ridge_regression(train_feat[k], train_y_oh[k], lam)
        acc, _ = evaluate_readout(W_k, test_feat[k], test_y[k])
        local_accs.append(acc)
    print(f"[1/4] Local-only ridge readout (no federation):")
    print(f"  Mean test accuracy: {np.mean(local_accs):.4f} ± {np.std(local_accs):.4f}\n")

    # ── 2. Centralized ridge (oracle, all data pooled) ─────────────
    feat_all = torch.cat(train_feat, dim=0)
    y_all = torch.cat(train_y_oh, dim=0)
    W_central = ridge_regression(feat_all, y_all, lam)
    cent_accs = [evaluate_readout(W_central, test_feat[k], test_y[k])[0] for k in range(K)]
    print(f"[2/4] Centralized ridge (oracle, all data pooled):")
    print(f"  Test accuracy: {np.mean(cent_accs):.4f}\n")

    # ── 3. FRI-Stats (one-shot federated aggregation) ──────────────
    fri_stats = FRIStatsAggregator(lam)
    W_stats, comm_scalars = fri_stats.run(train_feat, train_y_oh)
    stats_accs = [evaluate_readout(W_stats, test_feat[k], test_y[k])[0] for k in range(K)]
    matches_central = bool(np.allclose(W_stats.cpu().numpy(), W_central.cpu().numpy(), atol=1e-4))
    print(f"[3/4] FRI-Stats (one-shot federated aggregation):")
    print(f"  Test accuracy: {np.mean(stats_accs):.4f}")
    print(f"  Communication: {comm_cost_mb(comm_scalars):.2f} MiB")
    print(f"  Recovers centralized solution exactly: {matches_central}\n")

    # ── 4. FRI-Stats + personalization ─────────────────────────────
    mu = 0.01
    _, W_pers_list, _ = fri_stats.run_with_personalization(
        train_feat, train_y_oh, mu=mu
    )
    pers_accs = [evaluate_readout(W_pers_list[k], test_feat[k], test_y[k])[0] for k in range(K)]
    print(f"[4/4] FRI-Stats + personalization (mu={mu}):")
    print(f"  Mean test accuracy: {np.mean(pers_accs):.4f} ± {np.std(pers_accs):.4f}\n")

    print(f"Demo complete. Total runtime: {time.time()-t0:.1f} seconds on "
          f"{'GPU' if torch.cuda.is_available() else 'CPU'}.")


if __name__ == "__main__":
    main()
