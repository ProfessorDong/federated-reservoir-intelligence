"""
Input-size / compression-ratio accounting (Reviewer 2, point 1).

For each dataset and client, reports:
  * raw transmit size  = (sum over trials of timesteps) x d_x  scalars
    (what it would cost to ship raw inputs to a server holding the reservoir);
  * statistics transmit = d_phi(d_phi+1)/2 + d_y*d_phi  scalars (FRI-Stats, one round);
  * readout transmit    = d_y*d_phi scalars per round (FRI-Readout);
with d_phi = 2*N_r (mean/log-var pooling). Identifies the crossover where shipping
raw inputs would be cheaper than shipping statistics (when raw < statistics).

Saves ../results/input_sizes.json (per-client means).
"""
import sys, os, json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from config import ESN_CFG, BCI2A_CFG, NINAPRO_CFG, DVS128_CFG, RESULTS_DIR

N_R = ESN_CFG['N_r']
D_PHI = 2 * N_R


def summarize(name, client_X, d_y):
    """client_X: list of (n_trials, T, d_x) arrays/tensors."""
    d_x = client_X[0].shape[2]
    raw_per_client, ntrials = [], []
    for X in client_X:
        n, T, dx = X.shape[0], X.shape[1], X.shape[2]
        raw_per_client.append(n * T * dx)
        ntrials.append(n)
    raw = float(np.mean(raw_per_client))
    stats = D_PHI * (D_PHI + 1) / 2 + d_y * D_PHI
    readout = d_y * D_PHI
    rec = {
        'K': len(client_X), 'd_x': int(d_x), 'd_phi': D_PHI, 'd_y': int(d_y),
        'mean_trials_per_client': float(np.mean(ntrials)),
        'mean_timesteps_per_trial': float(np.mean([X.shape[1] for X in client_X])),
        'raw_scalars_per_client': raw,
        'stats_scalars_per_client': float(stats),
        'readout_scalars_per_round': float(readout),
        'stats_vs_raw_ratio': float(stats / raw),
        'stats_cheaper_than_raw': bool(stats < raw),
        # crossover: raw becomes cheaper than stats when total timesteps*d_x < stats
        'raw_cheaper_below_timesteps': float(stats / d_x),
    }
    print(f"\n[{name}] K={rec['K']} d_x={d_x} d_phi={D_PHI} d_y={d_y}")
    print(f"  mean trials/client={rec['mean_trials_per_client']:.0f}, "
          f"timesteps/trial={rec['mean_timesteps_per_trial']:.0f}")
    print(f"  raw/client={raw:,.0f}  stats={stats:,.0f}  readout/round={readout:,.0f}")
    print(f"  stats/raw={rec['stats_vs_raw_ratio']:.3f}  stats_cheaper={rec['stats_cheaper_than_raw']}")
    return rec


def main():
    out = {}
    # BCI-IV-2a
    try:
        from data_bci import load_bci2a_all, bci2a_to_federated
        cdata, _ = bci2a_to_federated(load_bci2a_all(session='T'))
        out['bci_iv2a'] = summarize('BCI-IV-2a', cdata, BCI2A_CFG['n_classes'])
    except Exception as e:
        print("BCI failed:", e)
    # Ninapro DB5
    try:
        from data_bci import load_ninapro_db5_all, ninapro_to_federated
        import torch
        cdata, clabels = ninapro_to_federated(load_ninapro_db5_all())
        all_l = torch.cat([y if torch.is_tensor(y) else torch.tensor(y) for y in clabels])
        out['ninapro_db5'] = summarize('Ninapro DB5', cdata, int(all_l.max())+1)
    except Exception as e:
        print("Ninapro failed:", e)
    path = os.path.join(RESULTS_DIR, 'input_sizes.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {path}")


if __name__ == '__main__':
    main()
