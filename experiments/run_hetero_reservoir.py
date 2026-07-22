"""
Heterogeneous-reservoir study (revision; addresses Reviewer 2).

Two complementary experiments on BCI-IV-2a (n = 5 independent seeds; results
reported as mean +/- 95% CI), written to ../results/hetero_reservoir_results.json.

Part A -- Coordinate-aligned perturbation (a genuine test of Theorem 4 /
    Corollary 1). All clients share ONE reservoir (same public seed); each
    client's reservoir weights are then perturbed by a controlled *relative*
    magnitude eps:  W^(k) = W + eps * (||W||_F / ||E||_F) * E,  E ~ N(0,1).
    Because the perturbation is small and neuron identities are preserved, the
    feature coordinate system stays aligned across clients. We sweep eps, and
    for each client measure the reservoir-induced feature misalignment
        Delta^2_k = || Ghat(feat_pert_k) - Ghat(feat_shared_k) ||_F^2
                    / || Ghat(feat_shared_k) ||_F^2
    (computed on the SAME data, so it isolates the reservoir perturbation), and
    the excess test risk of one-shot statistics-based aggregation relative to
    the shared-reservoir (eps = 0) solution. Theorem 4 / Corollary 1 predict
    excess risk = O(Delta^2). The original ablation instead used fully
    independent reservoirs for every "spread", so Delta^2 was huge and constant
    and did NOT probe the theorem's small-perturbation regime.

Part B -- Independent-seed stress test (Reviewer 2's "re-randomised reservoirs")
    with the sanity checks the reviewer requested. Every client gets an
    INDEPENDENT random reservoir (different seed), so there is no shared feature
    coordinate system. We report, per configuration:
      (1) FRI-Stats (one global readout from aggregated statistics)  -- the
          questioned result;
      (2) Local-only per reservoir (each client fits its own readout on its own
          reservoir) -- the natural per-client upper bound;
      (3) Linear estimator on the RAW inputs (no reservoir), both local-only and
          global statistics-aggregated -- raw electrode channels ARE a shared
          coordinate system, so this isolates the coordinate-stable, linearly
          decodable part of the signal that survives any random projection;
      (4) FRI-Readout (averaging locally optimal readouts) -- predicted to fail
          across incompatible bases;
      (5) FRI-Stats + personalization (mu = 0.01) -- to test whether local
          adaptation, not global aggregation, is what recovers accuracy;
      (6) Shared-reservoir FRI-Stats and FRI-Stats+Pers (the default, recommended
          configuration: the server broadcasts one public seed so all reservoirs
          are identical) -- the reference.
    Diagnostics: relative Frobenius distance and mean cosine similarity between
    each client's local readout and the global statistics-aggregated readout,
    quantifying how little independent reservoirs actually share.
"""
import sys, os, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from config import (DEVICE, SEEDS, ESN_CFG, RESULTS_DIR, FED_CFG)
from reservoir import (ESN, ridge_regression, compute_sufficient_statistics,
                       ridge_from_statistics, personalized_readout)
from federated import (FRIReadoutAggregator, FRIStatsAggregator, evaluate_readout)
from data_bci import load_bci2a_all, bci2a_to_federated
from run_ablation import (labels_to_onehot, run_esn_on_single, split_data, _make_esn)

WASHOUT = ESN_CFG['washout']
LAM = ESN_CFG['ridge_lambda']
N_R = ESN_CFG['N_r']
MU = FED_CFG['personalization_mu']
N_CLASSES = 4
EPS_GRID = [0.0, 0.003, 0.01, 0.03, 0.1, 0.3]


# ── feature helpers ──────────────────────────────────────────────────────────
def esn_features(esn, X):
    """[mean, log-var] pooled reservoir features for a batch of trials."""
    return run_esn_on_single(esn, X, WASHOUT)


def raw_features(X):
    """[mean, log-var] pooled RAW-input features (no reservoir). Shared coords."""
    X = X.to(DEVICE)
    feat_mean = X.mean(dim=1)
    feat_var = torch.log(X.var(dim=1) + 1e-8)
    return torch.cat([feat_mean, feat_var], dim=1)


def perturb_esn(base_seed, d_x, eps, pert_seed):
    """Shared base reservoir, then a controlled relative perturbation of W_r, W_in.

    Coordinate-aligned: identical neuron set, so feature axes stay comparable.
    """
    esn = _make_esn(d_x, N_CLASSES, base_seed)
    if eps > 0:
        rng = np.random.RandomState(pert_seed)
        for attr in ('W_r', 'W_in'):
            W = getattr(esn, attr).detach().cpu().numpy()
            E = rng.randn(*W.shape).astype(np.float32)
            scale = np.linalg.norm(W) / (np.linalg.norm(E) + 1e-12)
            Wp = (W + eps * scale * E).astype(np.float32)
            setattr(esn, attr, torch.from_numpy(Wp).to(DEVICE))
    return esn


def norm_gram(feat):
    """Per-sample normalised Gram matrix Ghat = F^T F / n."""
    n = feat.shape[0]
    return (feat.T @ feat) / max(n, 1)


def eval_mse(W, feat, oh):
    """Mean squared error of readout W on (feat, one-hot targets)."""
    pred = feat.to(DEVICE) @ W.T
    return torch.mean((pred - oh.to(DEVICE)) ** 2).item()


def eval_global(W, test_feat, test_y):
    accs = [evaluate_readout(W, test_feat[k], test_y[k])[0]
            for k in range(len(test_feat))]
    return float(np.mean(accs))


def eval_personal(W_list, test_feat, test_y):
    accs = [evaluate_readout(W_list[k], test_feat[k], test_y[k])[0]
            for k in range(len(test_feat))]
    return float(np.mean(accs))


# ── Part A: coordinate-aligned perturbation (Theorem 4 / Corollary 1) ────────
def part_a(train_X, test_X, train_y, test_y, seed):
    K = len(train_X)
    d_x = train_X[0].shape[2]
    train_oh = [labels_to_onehot(y, N_CLASSES).to(DEVICE) for y in train_y]

    # Shared reservoir (eps = 0) reference features, per client, on same data.
    shared = _make_esn(d_x, N_CLASSES, seed)
    shared_train = [esn_features(shared, train_X[k]) for k in range(K)]
    shared_test = [esn_features(shared, test_X[k]) for k in range(K)]

    fri = FRIStatsAggregator(LAM)
    W0, _ = fri.run(shared_train, train_oh)
    mse0 = float(np.mean([eval_mse(W0, shared_test[k],
                                   labels_to_onehot(test_y[k], N_CLASSES))
                          for k in range(K)]))

    out = {}
    for eps in EPS_GRID:
        deltas, ptrain, ptest = [], [], []
        for k in range(K):
            esn_k = perturb_esn(seed, d_x, eps, pert_seed=seed * 100 + k)
            ftr = esn_features(esn_k, train_X[k])
            fte = esn_features(esn_k, test_X[k])
            ptrain.append(ftr); ptest.append(fte)
            # reservoir-induced misalignment on identical (training) data
            Gp, Gs = norm_gram(ftr), norm_gram(shared_train[k])
            deltas.append(((Gp - Gs).pow(2).sum() / (Gs.pow(2).sum() + 1e-12)).item())
        W, _ = fri.run(ptrain, train_oh)
        acc = eval_global(W, ptest, test_y)
        mse = float(np.mean([eval_mse(W, ptest[k],
                                      labels_to_onehot(test_y[k], N_CLASSES))
                             for k in range(K)]))
        out[f'eps_{eps}'] = {
            'acc': acc,
            'delta_sq': float(np.mean(deltas)),
            'mse': mse,
            'excess_risk': mse - mse0,
        }
    return out


# ── Part B: independent reservoirs + sanity checks ───────────────────────────
def part_b(train_X, test_X, train_y, test_y, seed):
    K = len(train_X)
    d_x = train_X[0].shape[2]
    train_oh = [labels_to_onehot(y, N_CLASSES).to(DEVICE) for y in train_y]
    fri = FRIStatsAggregator(LAM)

    # (6) Shared reservoir (default / recommended): homogeneous reference
    shared = _make_esn(d_x, N_CLASSES, seed)
    sh_tr = [esn_features(shared, train_X[k]) for k in range(K)]
    sh_te = [esn_features(shared, test_X[k]) for k in range(K)]
    Wsh, _ = fri.run(sh_tr, train_oh)
    shared_stats_acc = eval_global(Wsh, sh_te, test_y)
    _, Wsh_pers, _ = fri.run_with_personalization(sh_tr, train_oh, MU)
    shared_pers_acc = eval_personal(Wsh_pers, sh_te, test_y)

    # Independent reservoirs: one per client (different seeds -> no shared coords)
    ind_tr, ind_te = [], []
    for k in range(K):
        esn_k = _make_esn(d_x, N_CLASSES, seed * 1000 + k + 7)
        ind_tr.append(esn_features(esn_k, train_X[k]))
        ind_te.append(esn_features(esn_k, test_X[k]))

    # (1) FRI-Stats global on independent reservoirs
    Wg, _ = fri.run(ind_tr, train_oh)
    indep_stats_acc = eval_global(Wg, ind_te, test_y)

    # (5) FRI-Stats + personalization on independent reservoirs
    _, Wg_pers, _ = fri.run_with_personalization(ind_tr, train_oh, MU)
    indep_pers_acc = eval_personal(Wg_pers, ind_te, test_y)

    # (2) Local-only per reservoir (each client fits its own readout)
    local_accs, local_W = [], []
    for k in range(K):
        Wk = ridge_regression(ind_tr[k], train_oh[k], LAM)
        local_W.append(Wk)
        local_accs.append(evaluate_readout(Wk, ind_te[k], test_y[k])[0])
    indep_local_acc = float(np.mean(local_accs))

    # (4) FRI-Readout (averaging local readouts) on independent reservoirs
    ro = FRIReadoutAggregator(K, participation_rate=1.0, ridge_lambda=LAM,
                              num_rounds=FED_CFG['num_rounds'])
    Wro, _, _ = ro.run(ind_tr, train_oh, seed=seed)
    indep_readout_acc = eval_global(Wro, ind_te, test_y)

    # Diagnostics: how different are the per-client local readouts from the
    # global statistics-aggregated readout? (relative Frobenius dist + cosine)
    rel_fro, cosines = [], []
    g = Wg.flatten()
    for k in range(K):
        wk = local_W[k].flatten()
        rel_fro.append((torch.norm(local_W[k] - Wg) / (torch.norm(local_W[k]) + 1e-12)).item())
        cosines.append((torch.dot(wk, g) /
                        ((torch.norm(wk) * torch.norm(g)) + 1e-12)).item())

    # (3) Linear estimator on RAW inputs (no reservoir; shared coordinates)
    raw_tr = [raw_features(train_X[k]) for k in range(K)]
    raw_te = [raw_features(test_X[k]) for k in range(K)]
    # local-only raw
    raw_local = float(np.mean([
        evaluate_readout(ridge_regression(raw_tr[k], train_oh[k], LAM),
                         raw_te[k], test_y[k])[0] for k in range(K)]))
    # global stats-aggregated raw (raw channels are aligned across clients)
    Wraw, _ = fri.run(raw_tr, train_oh)
    raw_global = eval_global(Wraw, raw_te, test_y)

    chance = 1.0 / N_CLASSES
    return {
        'chance': chance,
        'shared_stats_acc': shared_stats_acc,
        'shared_pers_acc': shared_pers_acc,
        'indep_stats_acc': indep_stats_acc,
        'indep_pers_acc': indep_pers_acc,
        'indep_local_acc': indep_local_acc,
        'indep_readout_acc': indep_readout_acc,
        'raw_linear_local_acc': raw_local,
        'raw_linear_global_acc': raw_global,
        'local_vs_global_rel_fro': float(np.mean(rel_fro)),
        'local_vs_global_cosine': float(np.mean(cosines)),
    }


def main():
    print("Loading BCI-IV-2a ...")
    subjects = load_bci2a_all(session='T')
    client_data, client_labels = bci2a_to_federated(subjects)
    results = {}
    for seed in SEEDS:
        print(f"\n===== seed {seed} =====")
        train_X, test_X, train_y, test_y = split_data(client_data, client_labels, seed=seed)
        a = part_a(train_X, test_X, train_y, test_y, seed)
        b = part_b(train_X, test_X, train_y, test_y, seed)
        results[str(seed)] = {'part_a': a, 'part_b': b}
        print("  Part A (aligned perturbation):")
        for eps in EPS_GRID:
            r = a[f'eps_{eps}']
            print(f"    eps={eps:<5}: acc={r['acc']:.4f}  Delta^2={r['delta_sq']:.4f}  "
                  f"excess_risk={r['excess_risk']:+.5f}")
        print("  Part B (independent reservoirs):")
        for key in ('shared_stats_acc', 'shared_pers_acc', 'indep_stats_acc',
                    'indep_pers_acc', 'indep_local_acc', 'indep_readout_acc',
                    'raw_linear_local_acc', 'raw_linear_global_acc',
                    'local_vs_global_cosine'):
            print(f"    {key:>24} = {b[key]:.4f}")

    out_path = os.path.join(RESULTS_DIR, 'hetero_reservoir_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out_path}")

    # Aggregate summary across seeds
    def agg(getter):
        vals = [getter(results[str(s)]) for s in SEEDS]
        m = float(np.mean(vals)); ci = 1.96 * float(np.std(vals)) / np.sqrt(len(vals))
        return m, ci
    print("\n===== SUMMARY (mean +/- 95% CI over 5 seeds) =====")
    print("Part A:")
    for eps in EPS_GRID:
        m, ci = agg(lambda r, e=eps: r['part_a'][f'eps_{e}']['acc'])
        md, _ = agg(lambda r, e=eps: r['part_a'][f'eps_{e}']['delta_sq'])
        mx, _ = agg(lambda r, e=eps: r['part_a'][f'eps_{e}']['excess_risk'])
        print(f"  eps={eps:<5}: acc={m:.4f}+/-{ci:.4f}  Delta^2={md:.4f}  excess={mx:+.5f}")
    print("Part B:")
    for key in ('shared_stats_acc', 'shared_pers_acc', 'indep_stats_acc',
                'indep_pers_acc', 'indep_local_acc', 'indep_readout_acc',
                'raw_linear_local_acc', 'raw_linear_global_acc',
                'local_vs_global_rel_fro', 'local_vs_global_cosine'):
        m, ci = agg(lambda r, k=key: r['part_b'][k])
        print(f"  {key:>24} = {m:.4f} +/- {ci:.4f}")


if __name__ == '__main__':
    main()
