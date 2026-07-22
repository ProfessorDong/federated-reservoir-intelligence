"""
Leakage-free session-adaptation (drift) experiment on BCI-IV-2b.

Fixes the train-on-test leakage in the original drift code, where the readout
was fit on [Session1 + Session2] features and then evaluated on the SAME
Session-2 features. Here each session is split into an adaptation part (used for
fitting / LSTM training) and a HELD-OUT test part (used only for evaluation):

  * FRI: discounted sufficient statistics over the ordered concatenation
    [S1_train then S2_train]; evaluated on held-out S1_test and S2_test.
  * FedAvg-LSTM (retrained): trained on [S1_train + S2_train]; evaluated on the
    same held-out S1_test / S2_test (matched supervision, no leakage).

n = 5 seeds; results -> ../results/drift_heldout_results.json
"""
import sys, os, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from config import (DEVICE, SEEDS, ESN_CFG, ABLATION_CFG, LSTM_CFG, RESULTS_DIR)
from reservoir import (ESN, compute_sufficient_statistics, ridge_from_statistics,
                       discounted_statistics)
from federated import evaluate_readout
from baselines import LSTMClassifier, train_local, evaluate_model
from data_bci_extra import load_bci2b_subject
from run_ablation import labels_to_onehot

WASHOUT = ESN_CFG['washout']; LAM = ESN_CFG['ridge_lambda']
N_CLASSES = 2; D_X = 3
BETAS = ABLATION_CFG['forgetting_factors']
TEST_RATIO = 0.3


def esn_feats(esn, X):
    st = esn.run(X, washout=WASHOUT)
    return torch.cat([st.mean(dim=1), torch.log(st.var(dim=1) + 1e-8)], dim=1)


def split_idx(n, ratio, rng):
    idx = rng.permutation(n)
    nt = max(1, int(n * ratio))
    return idx[nt:], idx[:nt]            # train, test


def main():
    results = {}
    for seed in SEEDS:
        s1a = {f'beta_{b}': [] for b in BETAS}
        s2a = {f'beta_{b}': [] for b in BETAS}
        lstm_s1, lstm_s2 = [], []
        for subj in range(1, 10):
            sd = load_bci2b_subject(subj, ['01T', '02T', '03T'])
            if '01T' not in sd or '02T' not in sd:
                continue
            esn = ESN(D_X, ESN_CFG['N_r'], N_CLASSES,
                      spectral_radius=ESN_CFG['spectral_radius'],
                      leaking_rate=ESN_CFG['leaking_rate'],
                      input_scaling=ESN_CFG['input_scaling'],
                      sparsity=ESN_CFG['sparsity'], seed=seed)
            d1, l1 = sd['01T']; d2, l2 = sd['02T']
            X1 = torch.from_numpy(d1.transpose(0, 2, 1)).float()
            mu, std = X1.mean(dim=(0, 1), keepdim=True), X1.std(dim=(0, 1), keepdim=True) + 1e-8
            X1 = (X1 - mu) / std
            X2 = (torch.from_numpy(d2.transpose(0, 2, 1)).float() - mu) / std   # S1-normalized
            f1, f2 = esn_feats(esn, X1), esn_feats(esn, X2)
            l1t, l2t = torch.from_numpy(l1).long(), torch.from_numpy(l2).long()
            oh1 = labels_to_onehot(l1t, N_CLASSES).to(DEVICE)
            oh2 = labels_to_onehot(l2t, N_CLASSES).to(DEVICE)

            rng = np.random.RandomState(seed * 100 + subj)
            tr1, te1 = split_idx(len(l1t), TEST_RATIO, rng)
            tr2, te2 = split_idx(len(l2t), TEST_RATIO, rng)

            # ordered: S1_train (older) then S2_train (newer) for discounting
            ctr_f = torch.cat([f1[tr1], f2[tr2]], dim=0)
            ctr_oh = torch.cat([oh1[tr1], oh2[tr2]], dim=0)
            for b in BETAS:
                if b < 1.0:
                    G, H, Te = discounted_statistics(ctr_f, ctr_oh, b)
                else:
                    G, H, Te = compute_sufficient_statistics(ctr_f, ctr_oh); Te = float(Te)
                W = ridge_from_statistics(G, H, Te, LAM)
                s1a[f'beta_{b}'].append(evaluate_readout(W, f1[te1], l1t[te1])[0])
                s2a[f'beta_{b}'].append(evaluate_readout(W, f2[te2], l2t[te2])[0])

            # LSTM retrained on S1_train + S2_train, eval on held-out test splits
            xtr = torch.cat([X1[tr1], X2[tr2]], dim=0)
            ytr = torch.cat([l1t[tr1], l2t[tr2]], dim=0)
            torch.manual_seed(seed)
            lstm = LSTMClassifier(D_X, LSTM_CFG['hidden_dim'], N_CLASSES,
                                  LSTM_CFG['num_layers'], LSTM_CFG['dropout']).to(DEVICE)
            train_local(lstm, xtr, ytr, epochs=20, lr=LSTM_CFG['lr'],
                        batch_size=LSTM_CFG['batch_size'])
            lstm_s1.append(evaluate_model(lstm, X1[te1], l1t[te1])[0])
            lstm_s2.append(evaluate_model(lstm, X2[te2], l2t[te2])[0])

        seed_res = {}
        for b in BETAS:
            seed_res[f'beta_{b}'] = {
                'session1_acc': float(np.mean(s1a[f'beta_{b}'])),
                'session2_acc': float(np.mean(s2a[f'beta_{b}'])),
            }
        seed_res['lstm_retrained'] = {
            'session1_acc': float(np.mean(lstm_s1)),
            'session2_acc': float(np.mean(lstm_s2)),
        }
        results[str(seed)] = seed_res
        print(f"seed {seed}: " + "  ".join(
            f"b{b} S2={seed_res[f'beta_{b}']['session2_acc']:.3f}" for b in BETAS)
            + f"  LSTM S2={seed_res['lstm_retrained']['session2_acc']:.3f}")

    path = os.path.join(RESULTS_DIR, 'drift_heldout_results.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {path}")
    print("\n=== SUMMARY (mean +/- 95% CI over seeds) ===")
    def agg(getter):
        v = [getter(results[str(s)]) for s in SEEDS]
        return np.mean(v), 1.96 * np.std(v) / np.sqrt(len(v))
    for b in BETAS:
        m1, c1 = agg(lambda r, b=b: r[f'beta_{b}']['session1_acc'])
        m2, c2 = agg(lambda r, b=b: r[f'beta_{b}']['session2_acc'])
        print(f"  beta={b:<5} S1={m1:.3f}+/-{c1:.3f}  S2={m2:.3f}+/-{c2:.3f}")
    m1, c1 = agg(lambda r: r['lstm_retrained']['session1_acc'])
    m2, c2 = agg(lambda r: r['lstm_retrained']['session2_acc'])
    print(f"  LSTM       S1={m1:.3f}+/-{c1:.3f}  S2={m2:.3f}+/-{c2:.3f}")


if __name__ == '__main__':
    main()
