"""
FRI-ESN on frame-based event data: closing the accuracy gap.

Key insight: use the SAME frame-based input representation as SNN baselines,
but with a fixed ESN reservoir instead of a trained SNN. This isolates the
effect of fixed vs. learned features from input representation differences.

Features:
- Frame-based input (same as baselines): events → frames → avg_pool → flatten
- Large ESN reservoir (N_r up to 2000)
- Temporal windowed features: divide ESN state sequence into W windows,
  extract [mean, var] per window → richer temporal representation
- Ridge regression readout (closed-form, no BPTT)
- Stats-based or readout-based federated aggregation
"""
import sys, os, json, time, gc
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from config import *
from reservoir import ESN, ridge_regression, compute_sufficient_statistics, \
    ridge_from_statistics
from federated import FRIStatsAggregator, FRIReadoutAggregator, \
    evaluate_readout, comm_cost_mb
from data_event import load_dvs128_all, events_to_frames


def labels_to_onehot(labels, n_classes):
    oh = torch.zeros(len(labels), n_classes)
    oh[torch.arange(len(labels)), labels] = 1.0
    return oh


def dvs_to_frame_sequences(samples, sensor_size=(128, 128),
                            time_window_us=50000, pool_k=4, max_frames=100):
    """Convert DVS event samples to frame sequences (same as SNN baselines)."""
    seqs, labels = [], []
    for events, label in samples:
        frames = events_to_frames(events, sensor_size, time_window_us)
        ft = torch.from_numpy(frames).float()
        if ft.shape[2] > 32:
            ft = F.avg_pool2d(ft, kernel_size=pool_k)
        n_f = ft.shape[0]
        d = ft.shape[1] * ft.shape[2] * ft.shape[3]
        flat = ft.reshape(n_f, d)
        # Truncate or pad to max_frames
        if n_f > max_frames:
            flat = flat[:max_frames]
        elif n_f < max_frames:
            pad = torch.zeros(max_frames - n_f, d)
            flat = torch.cat([flat, pad], dim=0)
        seqs.append(flat)
        labels.append(label)
    if not seqs:
        return torch.zeros(0, max_frames, 1), torch.zeros(0, dtype=torch.long)
    return torch.stack(seqs), torch.tensor(labels, dtype=torch.long)


def extract_windowed_features(states, n_windows=5):
    """Extract temporal-windowed [mean, logvar] features from ESN states.

    Args:
        states: (n_trials, T, N_r) ESN state tensor
        n_windows: number of temporal windows

    Returns:
        features: (n_trials, n_windows * 2 * N_r) tensor
    """
    n, T, N_r = states.shape
    win_size = T // n_windows
    feats = []
    for w in range(n_windows):
        start = w * win_size
        end = (w + 1) * win_size if w < n_windows - 1 else T
        window = states[:, start:end, :]
        w_mean = window.mean(dim=1)           # (n, N_r)
        w_var = torch.log(window.var(dim=1) + 1e-8)  # (n, N_r)
        feats.extend([w_mean, w_var])
    return torch.cat(feats, dim=1)  # (n, n_windows * 2 * N_r)


def extract_mean_logvar(states):
    """Standard [mean, logvar] features (no windowing)."""
    m = states.mean(dim=1)
    v = torch.log(states.var(dim=1) + 1e-8)
    return torch.cat([m, v], dim=1)


def run_experiment(seed=0):
    print(f"\n{'='*60}")
    print(f"FRI-ESN Frame-based Events — Seed {seed}")
    print(f"{'='*60}")

    # ── Load DVS128 data ──────────────────────────────────────────────────
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

    # ── Convert to frame sequences ────────────────────────────────────────
    print("Converting events to frame sequences...")
    train_frames, train_labels = [], []
    test_frames, test_labels = [], []
    for uid in valid_users:
        tr_f, tr_l = dvs_to_frame_sequences(user_data[uid]['train'])
        te_f, te_l = dvs_to_frame_sequences(user_data[uid]['test'])
        train_frames.append(tr_f)
        train_labels.append(tr_l)
        test_frames.append(te_f)
        test_labels.append(te_l)

    d_x = train_frames[0].shape[2]  # frame feature dim (2048)
    print(f"  Frame dim: {d_x}, Users: {K}")

    results = {}

    # ── Test multiple reservoir sizes ─────────────────────────────────────
    for N_r in [500, 1000, 2000]:
        print(f"\n{'─'*50}")
        print(f"ESN N_r = {N_r}")
        print(f"{'─'*50}")

        esn = ESN(
            input_dim=d_x, reservoir_dim=N_r, output_dim=n_classes,
            spectral_radius=0.95, leaking_rate=0.3, input_scaling=0.1,
            sparsity=0.95, seed=seed)

        # Process through ESN
        print("  Processing through ESN...")
        t1 = time.time()
        train_states, test_states = [], []
        for k in range(K):
            if len(train_frames[k]) > 0:
                # Process in batches to manage memory
                batch_size = 16
                states_k = []
                for i in range(0, len(train_frames[k]), batch_size):
                    batch = train_frames[k][i:i+batch_size]
                    s = esn.run(batch, washout=0)  # no washout for short sequences
                    states_k.append(s)
                train_states.append(torch.cat(states_k))
            else:
                train_states.append(torch.zeros(0, 100, N_r))

            if len(test_frames[k]) > 0:
                states_k = []
                for i in range(0, len(test_frames[k]), batch_size):
                    batch = test_frames[k][i:i+batch_size]
                    s = esn.run(batch, washout=0)
                    states_k.append(s)
                test_states.append(torch.cat(states_k))
            else:
                test_states.append(torch.zeros(0, 100, N_r))
        print(f"  Done in {time.time()-t1:.1f}s")

        # ── Feature extraction variants ───────────────────────────────────
        feature_configs = {
            'mean_logvar': (extract_mean_logvar, 2 * N_r),
            'windowed_W3': (lambda s: extract_windowed_features(s, 3), 6 * N_r),
            'windowed_W5': (lambda s: extract_windowed_features(s, 5), 10 * N_r),
        }

        for feat_name, (feat_fn, d_phi) in feature_configs.items():
            key = f'esn{N_r}_{feat_name}'
            print(f"\n  [{key}] d_phi={d_phi}")

            train_feat = [feat_fn(s).float().to(DEVICE) if len(s) > 0
                          else torch.zeros(0, d_phi, device=DEVICE)
                          for s in train_states]
            test_feat = [feat_fn(s).float().to(DEVICE) if len(s) > 0
                         else torch.zeros(0, d_phi, device=DEVICE)
                         for s in test_states]
            train_oh = [labels_to_onehot(y, n_classes).to(DEVICE)
                        for y in train_labels]

            # Stats-based aggregation
            valid_k = [k for k in range(K)
                       if len(train_feat[k]) > 0 and len(test_feat[k]) > 0]
            lam = 0.05  # Same as ESN config

            fri_s = FRIStatsAggregator(lam)
            W_s, comm_s = fri_s.run(
                [train_feat[k] for k in valid_k],
                [train_oh[k] for k in valid_k])

            accs = []
            for k in valid_k:
                acc, _ = evaluate_readout(W_s, test_feat[k], test_labels[k])
                accs.append(acc)

            results[key] = {
                'acc': np.mean(accs) if accs else 0.0,
                'n_params_readout': d_phi * n_classes,
                'comm_mb': comm_cost_mb(comm_s),
                'd_phi': d_phi,
                'N_r': N_r,
            }
            print(f"    Acc: {results[key]['acc']:.4f}, "
                  f"Comm: {results[key]['comm_mb']:.1f} MB")

        # Clean up states
        del train_states, test_states
        gc.collect()
        torch.cuda.empty_cache()

    return results


def main():
    all_results = {}
    for seed in SEEDS:
        try:
            all_results[seed] = run_experiment(seed)
        except Exception as e:
            print(f"Seed {seed} failed: {e}")
            import traceback; traceback.print_exc()
        gc.collect()
        torch.cuda.empty_cache()

    # Aggregate
    print("\n" + "=" * 60)
    print("FRI-ESN FRAME-BASED RESULTS")
    print("=" * 60)

    # Collect all method keys
    if all_results:
        methods = list(all_results[SEEDS[0]].keys())
        for m in sorted(methods):
            accs = [all_results[s][m]['acc'] for s in SEEDS
                    if s in all_results and m in all_results[s]]
            if accs:
                ci = 1.96 * np.std(accs) / np.sqrt(len(accs))
                info = all_results[SEEDS[0]][m]
                print(f"  {m:30s}: acc={np.mean(accs):.4f}±{ci:.4f}  "
                      f"d_phi={info['d_phi']:>5d}  "
                      f"comm={info['comm_mb']:.1f} MB")

    # Reference: current FRI-LSM and SNN baselines
    print("\n  Reference (from main experiments):")
    print("  FRI-LSM (traces, linear):     acc=0.4238  comm=11.5 MB")
    print("  FedProx-SNN:                  acc=0.7728  comm=352 MB")
    print("  FedAvg-SNN:                   acc=0.7333  comm=352 MB")
    print("  LFNL:                         acc=0.7263  comm=1.9 MB")

    save_path = os.path.join(RESULTS_DIR, 'fri_esn_events_results.json')
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
