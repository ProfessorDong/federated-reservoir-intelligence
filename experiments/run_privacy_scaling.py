"""
Privacy scaling experiment (Proposition 2 verification).

Goal
----
Verify the O(1/T_k^2) parameter mean-squared-error (MSE) scaling of the
closed-form readout under the Gaussian output-perturbation mechanism, and
the O(1/T_k) sample-level sensitivity that underlies it (Proposition 2,
Steps 1 and 3).

Mechanism (faithful to the main statement of Proposition 2)
-----------------------------------------------------------
Output perturbation of the aggregated ridge readout W* (shape d_y x d_phi):
    W~ = W* + Z,   Z ~ N(0, sigma_dp^2 I),
    sigma_dp = Delta_2 * sqrt(2 ln(1.25/delta)) / epsilon,
    Delta_2  = 2 B B_y (1 + B/sqrt(lambda)) / (lambda T_k)      [Prop. 2, Step 1]
The DP guarantee requires calibrating sigma_dp to the *worst-case* sensitivity
bound Delta_2 (a sup over neighbouring datasets), which scales as O(1/T_k); the
induced parameter MSE is E||Z||_F^2 = sigma_dp^2 d_phi d_y = O(1/T_k^2).

We additionally *measure* the empirical replace-one-sample sensitivity
  Delta_hat(T_k) = ||W(D) - W(D')||_F  over many single-sample replacements,
to confirm that the actual ridge sensitivity decays with T_k and stays within
the O(1/T_k) bound. This is the genuinely empirical (non-tautological) core of
the proposition.

Why parameter MSE and not accuracy: the quantity Proposition 2 bounds is the
parameter MSE. Accuracy under a *valid* epsilon=1 mechanism on a high-dimensional
(d_phi=1000) readout is heavily and non-monotonically perturbed and is therefore
not an informative y-axis for the scaling law (we record clean accuracy only as
context). Features are the manuscript's [mean, log-var] pooled reservoir states
(un-normalized); B is taken as the 99th-percentile per-sample feature norm (a
data-dependent a-priori bound). The verified quantity is the *scaling exponent*,
which is independent of the constant B.

Outputs results/privacy_scaling_results.json.
"""
import sys, os, json, math
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from config import *
from reservoir import ridge_from_statistics, compute_sufficient_statistics
from federated import evaluate_readout
from data_bci import load_bci2a_all, bci2a_to_federated
from run_ablation import (run_esn_on_clients, split_data, _make_esn,
                          labels_to_onehot)

EPSILON = ABLATION_CFG['privacy_scaling_epsilon']   # 1.0
DELTA   = ABLATION_CFG['dp_delta']                   # 1e-5
FRACS   = ABLATION_CFG['generalization_fracs']       # [0.1,0.2,0.3,0.5,0.7,1.0]
SEEDS_PS = SEEDS                                      # [0,1,2,3,4]
M_NOISE = 300          # Monte-Carlo draws of the DP noise (MSE concentration)
R_SENS  = 300          # replace-one-sample draws for empirical sensitivity
B_y     = 1.0          # one-hot targets => ||y|| = 1


def _agg_readout(sub_feat, sub_oh, lam):
    """Aggregate sufficient statistics across clients and solve ridge."""
    d_feat = sub_feat[0].shape[1]
    n_cls = sub_oh[0].shape[1]
    G = torch.zeros(d_feat, d_feat, device=DEVICE)
    H = torch.zeros(n_cls, d_feat, device=DEVICE)
    T = 0
    per_client = []
    for Fk, Yk in zip(sub_feat, sub_oh):
        Gk, Hk, Tk = compute_sufficient_statistics(Fk, Yk)
        G += Gk; H += Hk; T += Tk
        per_client.append((Fk, Yk, Tk))
    W = ridge_from_statistics(G, H, T, lam)
    return W, G, H, T, per_client


def privacy_scaling_seed(subjects, seed):
    lam = ESN_CFG['ridge_lambda']
    c_delta = math.sqrt(2 * math.log(1.25 / DELTA))

    client_data, client_labels = bci2a_to_federated(subjects)
    train_X, test_X, train_y, test_y = split_data(client_data, client_labels, seed=seed)
    n_classes = 4
    K = len(train_X)

    esn = _make_esn(train_X[0].shape[2], n_classes, seed)
    full_train_feat = run_esn_on_clients(esn, train_X, ESN_CFG['washout'])
    test_feat = run_esn_on_clients(esn, test_X, ESN_CFG['washout'])
    d_feat = full_train_feat[0].shape[1]

    # Data-dependent a-priori feature-norm bound B (99th percentile).
    allnorm = torch.cat([f.norm(dim=1) for f in full_train_feat]).cpu().numpy()
    B = float(np.percentile(allnorm, 99))

    out = {'_B': B, '_d_feat': d_feat}
    for frac in FRACS:
        sub_feat, sub_oh, Tks = [], [], []
        for k in range(K):
            n_k = full_train_feat[k].shape[0]
            n_sub = max(2, int(frac * n_k))
            rng = np.random.RandomState(seed * 100 + k)
            idx = rng.choice(n_k, n_sub, replace=False)
            sub_feat.append(full_train_feat[k][idx])
            sub_oh.append(labels_to_onehot(train_y[k][idx], n_classes).to(DEVICE))
            Tks.append(n_sub)
        T_k_avg = float(np.mean(Tks))

        W_clean, G, H, T_tot, per_client = _agg_readout(sub_feat, sub_oh, lam)
        W_norm2 = float((W_clean.norm() ** 2).item())
        clean_acc = float(np.mean([evaluate_readout(W_clean, test_feat[k], test_y[k])[0]
                                   for k in range(K)]))

        # --- Worst-case sensitivity bound (Prop. 2, Step 1) and DP noise scale ---
        sens_bound = 2 * B * B_y * (1 + B / math.sqrt(lam)) / (lam * T_k_avg)
        sigma_dp = sens_bound * c_delta / EPSILON

        # --- Monte-Carlo parameter MSE of the valid output-perturbation mechanism ---
        mse_draws = []
        for m in range(M_NOISE):
            rng = np.random.RandomState(seed * 100003 + int(frac * 1000) * 131 + m)
            Z = torch.from_numpy((rng.randn(n_classes, d_feat) * sigma_dp).astype(np.float32)).to(DEVICE)
            mse_draws.append(float((Z.norm() ** 2).item()))
        mse_mean = float(np.mean(mse_draws))
        mse_theory = sigma_dp ** 2 * d_feat * n_classes   # E||Z||_F^2

        # --- Empirical replace-one-sample readout sensitivity (Step 1, measured) ---
        sens_draws = []
        rs = np.random.RandomState(seed * 7777 + int(frac * 1000))
        for r in range(R_SENS):
            k = rs.randint(K)
            nk = sub_feat[k].shape[0]
            i = rs.randint(nk); j = rs.randint(nk)
            phi_o = sub_feat[k][i:i+1]; y_o = sub_oh[k][i:i+1]
            phi_n = sub_feat[k][j:j+1]; y_n = sub_oh[k][j:j+1]
            dG = phi_n.T @ phi_n - phi_o.T @ phi_o
            dH = y_n.T @ phi_n - y_o.T @ phi_o
            Wp = ridge_from_statistics(G + dG, H + dH, T_tot, lam)
            sens_draws.append(float((Wp - W_clean).norm().item()))
        sens_emp_mean = float(np.mean(sens_draws))
        sens_emp_p95 = float(np.percentile(sens_draws, 95))

        out[f'frac_{frac}'] = {
            'T_k_avg': T_k_avg,
            'sigma_dp': sigma_dp,
            'sens_bound': sens_bound,
            'param_mse': mse_mean,
            'param_mse_theory': mse_theory,
            'param_mse_rel': mse_mean / W_norm2,
            'W_norm2': W_norm2,
            'sens_emp_mean': sens_emp_mean,
            'sens_emp_p95': sens_emp_p95,
            'clean_acc': clean_acc,
        }
        print(f"  frac={frac}: T_k={T_k_avg:.0f}  sigma={sigma_dp:.3e}  "
              f"MSE={mse_mean:.3e}  sens_emp={sens_emp_mean:.3e}  clean={clean_acc:.3f}")
    return out


def main():
    print("Loading BCI-IV-2a data for privacy-scaling experiment...")
    subjects = load_bci2a_all(session='T')
    print(f"epsilon={EPSILON}, delta={DELTA}, M_noise={M_NOISE}, R_sens={R_SENS}, "
          f"seeds={list(SEEDS_PS)}")

    all_results = {}
    for seed in SEEDS_PS:
        print(f"\n{'#'*50}\n# SEED {seed}\n{'#'*50}")
        all_results[seed] = privacy_scaling_seed(subjects, seed)

    def to_ser(o):
        if isinstance(o, (np.floating, np.integer)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, dict): return {str(k): to_ser(v) for k, v in o.items()}
        if isinstance(o, list): return [to_ser(v) for v in o]
        return o

    save_path = os.path.join(RESULTS_DIR, 'privacy_scaling_results.json')
    with open(save_path, 'w') as f:
        json.dump(to_ser(all_results), f, indent=2)
    print(f"\nSaved {save_path}")

    # quick exponent summary across seeds
    Tks = [np.mean([all_results[s][f'frac_{fr}']['T_k_avg'] for s in SEEDS_PS]) for fr in FRACS]
    mse = [np.mean([all_results[s][f'frac_{fr}']['param_mse'] for s in SEEDS_PS]) for fr in FRACS]
    sens = [np.mean([all_results[s][f'frac_{fr}']['sens_emp_mean'] for s in SEEDS_PS]) for fr in FRACS]
    print(f"MSE  log-log slope = {np.polyfit(np.log(Tks), np.log(mse), 1)[0]:.3f}  (Prop2: -2)")
    print(f"sens log-log slope = {np.polyfit(np.log(Tks), np.log(sens), 1)[0]:.3f}  (Prop2: -1 bound)")


if __name__ == '__main__':
    main()
