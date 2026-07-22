"""
Additional deep-learning baselines requested by Reviewer 1 (point 1):

  * Local-only LSTM (no federation) -- isolates whether the low federated-LSTM
    accuracy is due to the datasets or the federated setting. Trained with a
    GENEROUS per-client budget (LOCAL_ONLY_EPOCHS), comparable to the effective
    number of local epochs the federated baselines receive
    (participation x rounds x local_epochs ~ 75-125), so the comparison is fair.
  * Federated (FedAvg) GRU, 1D-CNN, and TCN -- more communication-efficient /
    appropriate deep architectures than the LSTM.

Run across BCI-IV-2a, Ninapro DB5, and DVS128 Gesture, using each dataset's
exact existing pipeline (same train/test splits, participation rates, and local
training budget as the published baselines). Results (n = 5 seeds) are saved to
../results/extra_baselines_results.json. Already-computed federated entries are
reused; local-only is always (re)computed with the fair budget.

Usage:
    python run_extra_baselines.py                 # all three datasets
    python run_extra_baselines.py bci ninapro     # a subset
"""
import sys, os, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from config import (DEVICE, SEEDS, LSTM_CFG, FED_CFG, BCI2A_CFG, NINAPRO_CFG,
                    DVS128_CFG, RESULTS_DIR)
from baselines import (GRUClassifier, CNN1DClassifier, TCNClassifier,
                       LSTMClassifier, run_fedavg, run_local_only)
from federated import comm_cost_mb

LOCAL_ONLY_EPOCHS = 100          # fair budget for the local-only baseline
FED_NAMES = ('fedavg_gru', 'fedavg_cnn', 'fedavg_tcn')
OUT_PATH = os.path.join(RESULTS_DIR, 'extra_baselines_results.json')


def split_train_test(client_data, client_labels, test_ratio=0.2, seed=0):
    train_X, test_X, train_y, test_y = [], [], [], []
    for X, y in zip(client_data, client_labels):
        n = len(y)
        rng = np.random.RandomState(seed)
        idx = rng.permutation(n)
        n_test = max(1, int(n * test_ratio))
        train_X.append(X[idx[n_test:]]); test_X.append(X[idx[:n_test]])
        train_y.append(y[idx[n_test:]]); test_y.append(y[idx[:n_test]])
    return train_X, test_X, train_y, test_y


def model_fns(d_x, n_classes):
    h = LSTM_CFG['hidden_dim']
    return {
        'local_lstm': lambda: LSTMClassifier(d_x, h, n_classes,
                                             num_layers=LSTM_CFG['num_layers'],
                                             dropout=LSTM_CFG['dropout']),
        'fedavg_gru': lambda: GRUClassifier(d_x, h, n_classes,
                                            num_layers=LSTM_CFG['num_layers'],
                                            dropout=LSTM_CFG['dropout']),
        'fedavg_cnn': lambda: CNN1DClassifier(d_x, n_classes),
        'fedavg_tcn': lambda: TCNClassifier(d_x, n_classes),
    }


def _evaluate(train_X, train_y, test_X, test_y, d_x, n_classes,
              participation, seed, prev=None):
    fns = model_fns(d_x, n_classes)
    res = dict(prev) if prev else {}

    # Local-only LSTM (no communication), fair budget -- always (re)computed.
    m = run_local_only(fns['local_lstm'], train_X, train_y, test_X, test_y,
                       local_epochs=LOCAL_ONLY_EPOCHS, lr=LSTM_CFG['lr'],
                       batch_size=LSTM_CFG['batch_size'], seed=seed)
    res['local_only_lstm'] = {'acc': m['acc'], 'f1': m['f1'], 'comm_mb': 0.0,
                              'n_params': m['n_params'],
                              'local_epochs': LOCAL_ONLY_EPOCHS}

    # Federated GRU / CNN / TCN -- compute only if missing.
    for name in FED_NAMES:
        if name in res and 'acc' in res[name]:
            continue
        _, m = run_fedavg(fns[name], train_X, train_y, test_X, test_y,
                          num_rounds=FED_CFG['num_rounds'],
                          participation_rate=participation,
                          local_epochs=LSTM_CFG['local_epochs'], lr=LSTM_CFG['lr'],
                          batch_size=LSTM_CFG['batch_size'], seed=seed)
        res[name] = {'acc': m['acc'], 'f1': m['f1'],
                     'comm_mb': comm_cost_mb(m['comm_scalars']),
                     'n_params': m['n_params']}
    for k in ('local_only_lstm',) + FED_NAMES:
        v = res[k]
        print(f"    {k:>16}: acc={v['acc']:.4f}  comm={v['comm_mb']:.2f} MB"
              f"  ({v['n_params']:,} params)")
    return res


# ── BCI-IV-2a ────────────────────────────────────────────────────────────────
def run_bci(existing):
    from data_bci import load_bci2a_all, bci2a_to_federated
    print("\n########## BCI-IV-2a ##########")
    cdata, clabels = bci2a_to_federated(load_bci2a_all(session='T'))
    n_classes = BCI2A_CFG['n_classes']
    out = {}
    for seed in SEEDS:
        print(f"  -- seed {seed} --")
        tr_X, te_X, tr_y, te_y = split_train_test(cdata, clabels, seed=seed)
        out[str(seed)] = _evaluate(tr_X, tr_y, te_X, te_y, tr_X[0].shape[2],
                                   n_classes, FED_CFG['participation_bci2a'], seed,
                                   prev=existing.get(str(seed)))
    return out


# ── Ninapro DB5 ──────────────────────────────────────────────────────────────
def run_ninapro(existing):
    from data_bci import load_ninapro_db5_all, ninapro_to_federated
    print("\n########## Ninapro DB5 ##########")
    cdata, clabels = ninapro_to_federated(load_ninapro_db5_all())
    all_labels = torch.cat([y if torch.is_tensor(y) else torch.tensor(y)
                            for y in clabels])
    n_classes = int(all_labels.max().item()) + 1
    out = {}
    for seed in SEEDS:
        print(f"  -- seed {seed} --")
        tr_X, te_X, tr_y, te_y = split_train_test(cdata, clabels, seed=seed)
        out[str(seed)] = _evaluate(tr_X, tr_y, te_X, te_y, tr_X[0].shape[2],
                                   n_classes, FED_CFG['participation_ninapro'], seed,
                                   prev=existing.get(str(seed)))
    return out


# ── DVS128 Gesture (frame-based; each user is a client, own data split 80/20) ──
def run_dvs(existing):
    from data_event import load_dvs128_all
    from run_dvs128 import process_dvs_to_frames, pad_sequences
    print("\n########## DVS128 Gesture (frames) ##########")
    user_data, _, _ = load_dvs128_all()
    valid_users = [u for u in user_data if user_data[u]['train']]
    print(f"  {len(valid_users)} valid users")

    # Build per-user frame sequences ONCE (seed-independent); split per seed.
    user_frames = {}
    for uid in valid_users:
        frames, labels = process_dvs_to_frames(user_data[uid]['train'])
        if frames:
            user_frames[uid] = (frames, labels)
    if not user_frames:
        print("  No DVS frames produced; skipping.")
        return {}
    d_x = user_frames[next(iter(user_frames))][0][0].shape[1]
    n_classes = DVS128_CFG['n_classes']
    print(f"  {len(user_frames)} clients, frame_dim={d_x}")

    out = {}
    for seed in SEEDS:
        print(f"  -- seed {seed} --")
        rng = np.random.RandomState(seed)
        trX, trY, teX, teY = [], [], [], []
        for frames, labels in user_frames.values():
            idx = rng.permutation(len(frames))
            n_test = max(1, int(len(frames) * 0.2))
            te_i, tr_i = idx[:n_test], idx[n_test:]
            trX.append(pad_sequences([frames[i] for i in tr_i]))
            trY.append(torch.tensor([labels[i] for i in tr_i], dtype=torch.long))
            teX.append(pad_sequences([frames[i] for i in te_i]))
            teY.append(torch.tensor([labels[i] for i in te_i], dtype=torch.long))
        out[str(seed)] = _evaluate(trX, trY, teX, teY, d_x, n_classes,
                                   FED_CFG['participation_dvs'], seed,
                                   prev=existing.get(str(seed)))
    return out


DATASETS = {'bci': run_bci, 'ninapro': run_ninapro, 'dvs': run_dvs}


def main():
    which = [a for a in sys.argv[1:] if a in DATASETS] or list(DATASETS)
    results = {}
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as f:
            results = json.load(f)
    for name in which:
        try:
            results[name] = DATASETS[name](results.get(name, {}))
            with open(OUT_PATH, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"  [saved {name} -> {OUT_PATH}]")
        except Exception:
            import traceback; traceback.print_exc()
            print(f"  !! {name} failed")
    print(f"\nDone. Results in {OUT_PATH}")


if __name__ == '__main__':
    main()
