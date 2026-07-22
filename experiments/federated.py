"""
Federated Reservoir Intelligence — aggregation schemes.
"""
import torch
import numpy as np
from config import DEVICE
from reservoir import (ridge_regression, compute_sufficient_statistics,
                       ridge_from_statistics, discounted_statistics,
                       personalized_readout)


class FRIReadoutAggregator:
    """Readout-based FRI aggregation (FedAvg on readout weights)."""

    def __init__(self, n_clients, participation_rate, ridge_lambda, num_rounds=50):
        self.K = n_clients
        self.q = participation_rate
        self.lam = ridge_lambda
        self.R = num_rounds

    def run(self, client_features, client_labels, seed=0):
        """
        Run readout-based federated aggregation.

        Args:
            client_features: list of K tensors (T_k, d)
            client_labels: list of K tensors (T_k, c)

        Returns:
            W_global: (c, d) global readout
            comm_cost: total scalars communicated
            per_round_weights: list of W at each round
        """
        rng = np.random.RandomState(seed)
        K = len(client_features)
        d = client_features[0].shape[1]
        c = client_labels[0].shape[1]
        T_all = [f.shape[0] for f in client_features]
        n_select = max(1, int(self.q * K))

        # Precompute local solutions
        local_solutions = []
        for k in range(K):
            W_k = ridge_regression(client_features[k], client_labels[k], self.lam)
            local_solutions.append(W_k)

        # Initialize global readout
        W_global = torch.zeros(c, d, device=DEVICE)
        per_round_weights = [W_global.clone()]
        comm_cost = 0

        for r in range(self.R):
            selected = sorted(rng.choice(K, n_select, replace=False))
            T_sel = sum(T_all[k] for k in selected)

            W_agg = torch.zeros(c, d, device=DEVICE)
            for k in selected:
                alpha_k = T_all[k] / T_sel
                W_agg += alpha_k * local_solutions[k]
                comm_cost += c * d  # upload

            W_global = W_agg
            per_round_weights.append(W_global.clone())
            comm_cost += c * d  # broadcast

        return W_global, comm_cost, per_round_weights


class FRIReadoutProximalAggregator:
    """Proximal readout-based aggregation for Theorem 3 validation.

    Each client solves a proximal problem centred on the current global readout:
        min_W  F_k(W) + prox_mu * ||W - W_global||_F^2
    which has closed-form solution:
        W_k = (H_k + prox_mu*T_k*W_global)(G_k + (lam+prox_mu)*T_k*I)^{-1}
    """

    def __init__(self, n_clients, participation_rate, ridge_lambda,
                 num_rounds=50, prox_mu=0.01):
        self.K = n_clients
        self.q = participation_rate
        self.lam = ridge_lambda
        self.R = num_rounds
        self.prox_mu = prox_mu

    def run(self, client_features, client_labels, seed=0):
        rng = np.random.RandomState(seed)
        K = len(client_features)
        d = client_features[0].shape[1]
        c = client_labels[0].shape[1]
        T_all = [f.shape[0] for f in client_features]
        n_select = max(1, int(self.q * K))

        # Pre-compute per-client sufficient statistics
        client_stats = []
        for k in range(K):
            R_k = client_features[k].to(DEVICE)
            Y_k = client_labels[k].to(DEVICE)
            G_k = R_k.T @ R_k
            H_k = Y_k.T @ R_k
            client_stats.append((G_k, H_k, T_all[k]))

        W_global = torch.zeros(c, d, device=DEVICE)
        per_round_weights = [W_global.clone()]
        comm_cost = 0
        I_d = torch.eye(d, device=DEVICE)

        for r in range(self.R):
            selected = sorted(rng.choice(K, n_select, replace=False))
            T_sel = sum(T_all[k] for k in selected)

            W_agg = torch.zeros(c, d, device=DEVICE)
            for k in selected:
                G_k, H_k, T_k = client_stats[k]
                # Proximal local solution
                num = H_k + self.prox_mu * T_k * W_global
                den = G_k + (self.lam + self.prox_mu) * T_k * I_d
                W_k = num @ torch.linalg.solve(den, I_d)
                alpha_k = T_all[k] / T_sel
                W_agg += alpha_k * W_k
                comm_cost += c * d

            W_global = W_agg
            per_round_weights.append(W_global.clone())
            comm_cost += c * d

        return W_global, comm_cost, per_round_weights


class FRIStatsAggregator:
    """Statistics-based FRI aggregation (one-shot exact)."""

    def __init__(self, ridge_lambda):
        self.lam = ridge_lambda

    def run(self, client_features, client_labels):
        """
        One-shot statistics-based aggregation.

        Returns:
            W_global: (c, d)
            comm_cost: total scalars communicated
        """
        K = len(client_features)
        d = client_features[0].shape[1]
        c = client_labels[0].shape[1]

        G_agg = torch.zeros(d, d, device=DEVICE)
        H_agg = torch.zeros(c, d, device=DEVICE)
        T_total = 0
        comm_cost = 0

        for k in range(K):
            G_k, H_k, T_k = compute_sufficient_statistics(
                client_features[k], client_labels[k])
            G_agg += G_k
            H_agg += H_k
            T_total += T_k
            # Communication: upper triangle of G_k + H_k
            comm_cost += d * (d + 1) // 2 + c * d

        W_global = ridge_from_statistics(G_agg, H_agg, T_total, self.lam)
        comm_cost += c * d  # broadcast

        return W_global, comm_cost

    def run_with_personalization(self, client_features, client_labels, mu):
        """Stats-based aggregation + personalization."""
        K = len(client_features)
        d = client_features[0].shape[1]
        c = client_labels[0].shape[1]

        # Aggregate
        G_agg = torch.zeros(d, d, device=DEVICE)
        H_agg = torch.zeros(c, d, device=DEVICE)
        T_total = 0
        client_stats = []

        for k in range(K):
            G_k, H_k, T_k = compute_sufficient_statistics(
                client_features[k], client_labels[k])
            G_agg += G_k
            H_agg += H_k
            T_total += T_k
            client_stats.append((G_k, H_k, T_k))

        W_global = ridge_from_statistics(G_agg, H_agg, T_total, self.lam)

        # Personalize
        W_personal = []
        for k in range(K):
            G_k, H_k, T_k = client_stats[k]
            W_k = personalized_readout(G_k, H_k, T_k, W_global, self.lam, mu)
            W_personal.append(W_k)

        comm_cost = K * (d * (d + 1) // 2 + c * d) + c * d
        return W_global, W_personal, comm_cost


class FRIDriftAggregator:
    """Statistics-based aggregation with drift adaptation."""

    def __init__(self, ridge_lambda, beta=0.95):
        self.lam = ridge_lambda
        self.beta = beta

    def run(self, client_features, client_labels, mu=0.0):
        """
        Discounted statistics-based aggregation.

        Returns:
            W_global, W_personal (if mu > 0), comm_cost
        """
        K = len(client_features)
        d = client_features[0].shape[1]
        c = client_labels[0].shape[1]

        G_agg = torch.zeros(d, d, device=DEVICE)
        H_agg = torch.zeros(c, d, device=DEVICE)
        T_eff_total = 0.0
        client_stats = []

        for k in range(K):
            G_k, H_k, T_eff_k = discounted_statistics(
                client_features[k], client_labels[k], self.beta)
            G_agg += G_k
            H_agg += H_k
            T_eff_total += T_eff_k
            client_stats.append((G_k, H_k, T_eff_k))

        W_global = ridge_from_statistics(G_agg, H_agg, T_eff_total, self.lam)

        comm_cost = K * (d * (d + 1) // 2 + c * d) + c * d

        if mu > 0:
            W_personal = []
            for k in range(K):
                G_k, H_k, T_eff_k = client_stats[k]
                W_k = personalized_readout(G_k, H_k, T_eff_k, W_global, self.lam, mu)
                W_personal.append(W_k)
            return W_global, W_personal, comm_cost
        return W_global, None, comm_cost


# ═════════════════════════════════════════════════════════════════════════════
# Evaluation helpers
# ═════════════════════════════════════════════════════════════════════════════
def evaluate_readout(W, features, labels):
    """
    Evaluate readout accuracy and F1.

    Args:
        W: (c, d)
        features: (T, d)
        labels: (T,) integer class labels or (T, c) one-hot

    Returns:
        acc, f1_macro
    """
    from sklearn.metrics import accuracy_score, f1_score
    features = features.to(DEVICE)
    preds = (features @ W.T)  # (T, c)
    pred_classes = preds.argmax(dim=1).cpu().numpy()

    if labels.dim() > 1:
        true_classes = labels.argmax(dim=1).cpu().numpy()
    else:
        true_classes = labels.cpu().numpy()

    acc = accuracy_score(true_classes, pred_classes)
    f1 = f1_score(true_classes, pred_classes, average='macro', zero_division=0)
    return acc, f1


def comm_cost_mb(n_scalars, bits=32):
    """Convert a scalar count to mebibytes (MiB = 2^20 bytes).

    Reported as ``MB`` in the paper tables for brevity, with the binary-units
    convention documented in the Table 2 footnote of the manuscript.
    """
    return n_scalars * bits / (8 * 1024 * 1024)
