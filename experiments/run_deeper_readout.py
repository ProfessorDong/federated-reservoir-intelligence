"""
Experiment: Deeper readout for FRI-LSM.

Tests MLP readouts of varying depth/width on top of fixed LSM reservoirs,
federated via FedAvg on the MLP parameters only.

This creates a Pareto frontier between accuracy and communication cost,
bridging the gap between linear FRI-LSM and full FedAvg-SNN.
"""
import sys, os, json, time, gc, copy
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from config import *
from reservoir import LSM
from federated import evaluate_readout, comm_cost_mb
from data_event import load_dvs128_all, events_to_spike_input, events_to_frames
from baselines import run_fedavg, train_local


# ═══════════════════════════════════════════════════════════════════════════════
# MLP Readout model (lightweight, placed on top of fixed reservoir features)
# ═══════════════════════════════════════════════════════════════════════════════
class MLPReadout(nn.Module):
    """MLP readout on top of fixed reservoir features."""
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


class TwoLayerMLPReadout(nn.Module):
    """2-hidden-layer MLP readout."""
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def labels_to_onehot(labels, n_classes):
    oh = torch.zeros(len(labels), n_classes)
    oh[torch.arange(len(labels)), labels] = 1.0
    return oh


def process_dvs_with_lsm(lsm, samples, duration_ms=300, dt_ms=1.0, n_input=128):
    """Process DVS event samples through LSM, return trace features."""
    features_list, labels_list = [], []
    for events, label in samples:
        input_currents = events_to_spike_input(events, n_input, duration_ms, dt_ms)
        result = lsm.run(input_currents, return_type='traces')
        feat = result['traces'].mean(dim=0)
        features_list.append(feat)
        labels_list.append(label)
    if not features_list:
        return torch.zeros(0, lsm.N_s), torch.zeros(0, dtype=torch.long)
    return torch.stack(features_list), torch.tensor(labels_list, dtype=torch.long)


def run_dvs128_deeper(seed=0):
    """Run DVS128 with deeper readout variants."""
    print(f"\n{'='*60}")
    print(f"DVS128 Deeper Readout — Seed {seed}")
    print(f"{'='*60}")

    # Load data
    print("\nLoading DVS128 Gesture data...")
    t0 = time.time()
    user_data, train_users, test_users = load_dvs128_all()
    print(f"  Loaded in {time.time()-t0:.1f}s")

    valid_users = [u for u in user_data if user_data[u]['train']]
    rng = np.random.RandomState(seed)
    for uid in valid_users:
        samples = user_data[uid]['train']
        rng.shuffle(samples)
        n_test = max(1, int(len(samples) * 0.2))
        user_data[uid]['test'] = samples[:n_test]
        user_data[uid]['train'] = samples[n_test:]

    K = len(valid_users)
    n_classes = DVS128_CFG['n_classes']
    n_input = 128

    # Initialize LSM (fixed, shared across all readout experiments)
    print("Initializing LSM...")
    lsm = LSM(
        input_dim=n_input, n_neurons=LSM_CFG['N_s'],
        output_dim=n_classes, tau_m=LSM_CFG['tau_m'],
        V_th=LSM_CFG['V_th'], V_rest=LSM_CFG['V_rest'],
        V_reset=LSM_CFG['V_reset'], tau_ref=LSM_CFG['tau_ref'],
        connectivity=LSM_CFG['connectivity'], dt=LSM_CFG['dt'],
        trace_decay=LSM_CFG['trace_decay'], seed=seed,
    )

    # Process through LSM once (features are reused for all readout types)
    print("Processing through LSM...")
    train_features, train_labels = [], []
    test_features, test_labels = [], []
    for uid in valid_users:
        tr_f, tr_l = process_dvs_with_lsm(lsm, user_data[uid]['train'],
                                            n_input=n_input)
        te_f, te_l = process_dvs_with_lsm(lsm, user_data[uid]['test'],
                                            n_input=n_input)
        train_features.append(tr_f.float())
        train_labels.append(tr_l)
        test_features.append(te_f.float())
        test_labels.append(te_l)

    d_phi = train_features[0].shape[1]  # 500
    print(f"  Feature dim: {d_phi}, Users: {K}")

    results = {}

    # ── Linear readout (baseline, ridge regression) ────────────────────────
    print("\n[0] Linear readout (ridge regression)...")
    from reservoir import ridge_regression
    from federated import FRIStatsAggregator
    lam = LSM_CFG['ridge_lambda']
    train_oh = [labels_to_onehot(y, n_classes).to(DEVICE) for y in train_labels]
    valid_k = [k for k in range(K) if len(train_features[k]) > 0]
    fri_s = FRIStatsAggregator(lam)
    W_lin, comm_lin = fri_s.run(
        [train_features[k] for k in valid_k],
        [train_oh[k] for k in valid_k])
    lin_accs = []
    for k in valid_k:
        if len(test_features[k]) > 0:
            acc, _ = evaluate_readout(W_lin, test_features[k], test_labels[k])
            lin_accs.append(acc)
    results['linear'] = {
        'acc': np.mean(lin_accs), 'n_params': d_phi * n_classes,
        'comm_mb': comm_cost_mb(comm_lin),
    }
    print(f"  Acc: {results['linear']['acc']:.4f}, "
          f"Params: {results['linear']['n_params']}")

    # ── MLP readouts of varying hidden dimensions ──────────────────────────
    hidden_dims = [32, 64, 128, 256]

    for h_dim in hidden_dims:
        print(f"\n[MLP h={h_dim}] FRI-LSM + MLP readout (FedAvg on MLP)...")

        def make_mlp():
            return MLPReadout(d_phi, h_dim, n_classes, dropout=0.2)

        _, mlp_m = run_fedavg(
            make_mlp, train_features, train_labels,
            test_features, test_labels,
            num_rounds=FED_CFG['num_rounds'],
            participation_rate=FED_CFG['participation_dvs'],
            local_epochs=5, lr=1e-3, batch_size=32, seed=seed)

        results[f'mlp_h{h_dim}'] = {
            'acc': mlp_m['acc'],
            'n_params': mlp_m['n_params'],
            'comm_mb': comm_cost_mb(mlp_m['comm_scalars']),
        }
        print(f"  Acc: {mlp_m['acc']:.4f}, "
              f"Params: {mlp_m['n_params']}, "
              f"Comm: {results[f'mlp_h{h_dim}']['comm_mb']:.1f} MB")

    # ── 2-layer MLP (deeper) ──────────────────────────────────────────────
    for h_dim in [128, 256]:
        print(f"\n[2-layer MLP h={h_dim}] FRI-LSM + 2-layer MLP...")

        def make_deep_mlp():
            return TwoLayerMLPReadout(d_phi, h_dim, n_classes, dropout=0.2)

        _, dm = run_fedavg(
            make_deep_mlp, train_features, train_labels,
            test_features, test_labels,
            num_rounds=FED_CFG['num_rounds'],
            participation_rate=FED_CFG['participation_dvs'],
            local_epochs=5, lr=1e-3, batch_size=32, seed=seed)

        results[f'mlp2_h{h_dim}'] = {
            'acc': dm['acc'],
            'n_params': dm['n_params'],
            'comm_mb': comm_cost_mb(dm['comm_scalars']),
        }
        print(f"  Acc: {dm['acc']:.4f}, "
              f"Params: {dm['n_params']}, "
              f"Comm: {results[f'mlp2_h{h_dim}']['comm_mb']:.1f} MB")

    return results


def main():
    all_results = {}
    for seed in SEEDS:
        try:
            all_results[seed] = run_dvs128_deeper(seed)
        except Exception as e:
            print(f"Seed {seed} failed: {e}")
            import traceback; traceback.print_exc()
        gc.collect()
        torch.cuda.empty_cache()

    # Aggregate
    print("\n" + "=" * 60)
    print("DEEPER READOUT RESULTS")
    print("=" * 60)
    methods = ['linear'] + [f'mlp_h{h}' for h in [32, 64, 128, 256]] + \
              [f'mlp2_h{h}' for h in [128, 256]]
    for m in methods:
        accs = [all_results[s][m]['acc'] for s in SEEDS if s in all_results and m in all_results[s]]
        if accs:
            params = all_results[SEEDS[0]][m]['n_params']
            comm = all_results[SEEDS[0]][m]['comm_mb']
            ci = 1.96 * np.std(accs) / np.sqrt(len(accs))
            print(f"  {m:18s}: acc={np.mean(accs):.4f}±{ci:.4f}  "
                  f"params={params:>7d}  comm={comm:.1f} MB")

    save_path = os.path.join(RESULTS_DIR, 'deeper_readout_results.json')
    def to_ser(obj):
        if isinstance(obj, (np.floating, np.integer)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {str(k): to_ser(v) for k, v in obj.items()}
        if isinstance(obj, list): return [to_ser(v) for v in obj]
        return obj
    with open(save_path, 'w') as f:
        json.dump(to_ser(all_results), f, indent=2)
    print(f"\nResults saved to {save_path}")


if __name__ == '__main__':
    main()
