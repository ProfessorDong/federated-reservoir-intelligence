"""
Echo State Network (ESN) and Liquid State Machine (LSM) implementations,
plus closed-form ridge readout, sufficient-statistics utilities, exponentially
discounted statistics for drift adaptation, and the personalized readout. All
tensor computations use PyTorch and run on GPU when available.
"""
import torch
import numpy as np
from config import DEVICE


# ═══════════════════════════════════════════════════════════════════════════════
# Echo State Network
# ═══════════════════════════════════════════════════════════════════════════════
class ESN:
    """Echo State Network with fixed reservoir and linear readout."""

    def __init__(self, input_dim, reservoir_dim, output_dim,
                 spectral_radius=0.95, leaking_rate=0.3, input_scaling=0.5,
                 sparsity=0.9, seed=0):
        self.N_r = reservoir_dim
        self.d_x = input_dim
        self.d_y = output_dim
        self.gamma = leaking_rate

        rng = np.random.RandomState(seed)

        # Input weights
        W_in = rng.randn(reservoir_dim, input_dim).astype(np.float32)
        W_in *= input_scaling / np.sqrt(input_dim)
        self.W_in = torch.from_numpy(W_in).to(DEVICE)

        # Reservoir weights (sparse)
        W_r = rng.randn(reservoir_dim, reservoir_dim).astype(np.float32)
        mask = (rng.rand(reservoir_dim, reservoir_dim) > sparsity).astype(np.float32)
        W_r *= mask
        # Scale to desired spectral radius
        eigs = np.abs(np.linalg.eigvals(W_r))
        if eigs.max() > 0:
            W_r *= spectral_radius / eigs.max()
        self.W_r = torch.from_numpy(W_r).to(DEVICE)

        # Bias
        self.b_r = torch.from_numpy(
            rng.uniform(-0.1, 0.1, reservoir_dim).astype(np.float32)
        ).to(DEVICE)

    def run(self, X, washout=100):
        """
        Run reservoir on input sequence(s).

        Args:
            X: (T, d_x) or (N, T, d_x) tensor
            washout: discard first `washout` steps

        Returns:
            states: (T-washout, N_r) or (N, T-washout, N_r)
        """
        batched = X.dim() == 3
        if not batched:
            X = X.unsqueeze(0)

        N, T, d_x = X.shape
        X = X.to(DEVICE)
        states = torch.zeros(N, T, self.N_r, device=DEVICE)
        r = torch.zeros(N, self.N_r, device=DEVICE)

        for t in range(T):
            pre = X[:, t] @ self.W_in.T + r @ self.W_r.T + self.b_r
            r = (1 - self.gamma) * r + self.gamma * torch.tanh(pre)
            states[:, t] = r

        states = states[:, washout:]
        if not batched:
            states = states.squeeze(0)
        return states

    def run_trials(self, trials, washout=100):
        """
        Run reservoir on a list of trial tensors of varying length.

        Args:
            trials: list of (T_i, d_x) tensors

        Returns:
            list of (T_i - washout, N_r) tensors
        """
        results = []
        for trial in trials:
            results.append(self.run(trial, washout=washout))
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# Liquid State Machine (LIF Reservoir)
# ═══════════════════════════════════════════════════════════════════════════════
class LSM:
    """Liquid State Machine with LIF neurons and spike-derived features."""

    def __init__(self, input_dim, n_neurons, output_dim,
                 tau_m=20e-3, V_th=1.0, V_rest=0.0, V_reset=0.0,
                 tau_ref=2e-3, connectivity=0.1, dt=1e-3,
                 trace_decay=0.95, seed=0):
        self.N_s = n_neurons
        self.d_x = input_dim
        self.d_y = output_dim
        self.tau_m = tau_m
        self.V_th = V_th
        self.V_rest = V_rest
        self.V_reset = V_reset
        self.tau_ref = tau_ref
        self.dt = dt
        self.trace_decay = trace_decay
        self.ref_steps = max(1, int(tau_ref / dt))

        rng = np.random.RandomState(seed)

        # Input weights (sparse, from input channels to neurons)
        W_in = rng.randn(n_neurons, input_dim).astype(np.float32)
        in_mask = (rng.rand(n_neurons, input_dim) > 0.7).astype(np.float32)
        W_in *= in_mask * 0.5
        self.W_in = torch.from_numpy(W_in).to(DEVICE)

        # Recurrent weights (log-normal, sparse)
        W_rec = np.abs(rng.lognormal(0, 0.5, (n_neurons, n_neurons)).astype(np.float32))
        rec_mask = (rng.rand(n_neurons, n_neurons) < connectivity).astype(np.float32)
        np.fill_diagonal(rec_mask, 0)
        W_rec *= rec_mask
        # Excitatory/inhibitory: 80% excitatory, 20% inhibitory
        ei_mask = np.ones(n_neurons, dtype=np.float32)
        ei_mask[rng.choice(n_neurons, int(0.2 * n_neurons), replace=False)] = -1.0
        W_rec *= ei_mask[np.newaxis, :]
        # Scale for stability
        max_row = np.abs(W_rec).sum(axis=1).max()
        if max_row > 0:
            W_rec *= 0.8 / max_row  # ensure row-sum < 1 for stability
        self.W_rec = torch.from_numpy(W_rec).to(DEVICE)

    def run(self, X, return_type='traces'):
        """
        Simulate LIF reservoir.

        Args:
            X: (T, d_x) input currents
            return_type: 'traces' | 'multiscale' | 'both'

        Returns:
            features: dict with 'traces' and/or 'multiscale' arrays
        """
        X = X.to(DEVICE)
        T = X.shape[0]

        V = torch.full((self.N_s,), self.V_rest, device=DEVICE)
        ref_count = torch.zeros(self.N_s, device=DEVICE, dtype=torch.int32)
        traces = torch.zeros(self.N_s, device=DEVICE)

        all_traces = torch.zeros(T, self.N_s, device=DEVICE)
        all_spikes = torch.zeros(T, self.N_s, device=DEVICE)

        alpha = self.dt / self.tau_m

        for t in range(T):
            # Input current
            I_ext = X[t] @ self.W_in.T

            # Recurrent current from previous spikes
            if t > 0:
                I_rec = all_spikes[t-1] @ self.W_rec.T
            else:
                I_rec = torch.zeros(self.N_s, device=DEVICE)

            # LIF dynamics (neurons not in refractory period)
            active = (ref_count == 0)
            dV = alpha * (-(V - self.V_rest) + I_ext + I_rec)
            V = V + dV * active.float()

            # Spike detection
            spikes = (V >= self.V_th).float()
            all_spikes[t] = spikes

            # Reset
            V = torch.where(V >= self.V_th,
                            torch.full_like(V, self.V_reset), V)
            ref_count = torch.where(spikes > 0,
                                    torch.full_like(ref_count, self.ref_steps),
                                    torch.clamp(ref_count - 1, min=0))

            # Exponential trace
            traces = self.trace_decay * traces + spikes
            all_traces[t] = traces

        result = {}
        if return_type in ('traces', 'both'):
            result['traces'] = all_traces  # (T, N_s)

        if return_type in ('multiscale', 'both'):
            # Multi-scale spike counts: windows of [10, 50, 100] time steps
            windows = [10, 50, min(100, T)]
            ms_features = []
            for w in windows:
                if T >= w:
                    # Count spikes in last w steps at each time point
                    # Use cumsum for efficiency
                    cumsum = torch.cumsum(all_spikes, dim=0)
                    shifted = torch.zeros_like(cumsum)
                    shifted[w:] = cumsum[:-w]
                    counts = cumsum - shifted  # (T, N_s)
                    ms_features.append(counts)
                else:
                    ms_features.append(all_spikes.cumsum(dim=0))
            result['multiscale'] = torch.cat(ms_features, dim=1)  # (T, N_s * n_windows)

        if not result:
            result['traces'] = all_traces
        return result

    def run_batch(self, batch_X, return_type='traces'):
        """Run on a batch of variable-length inputs."""
        results = []
        for x in batch_X:
            results.append(self.run(x, return_type))
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# Ridge Regression (closed-form readout)
# ═══════════════════════════════════════════════════════════════════════════════
def ridge_regression(R, Y, lam):
    """
    Closed-form ridge regression: W = Y^T R (R^T R + λI)^{-1}

    Args:
        R: (T, d) features
        Y: (T, c) targets
        lam: regularization

    Returns:
        W: (c, d) readout weights
    """
    R = R.to(DEVICE)
    Y = Y.to(DEVICE)
    G = R.T @ R  # (d, d)
    H = Y.T @ R  # (c, d)
    T = R.shape[0]
    I = torch.eye(G.shape[0], device=DEVICE)
    W = H @ torch.linalg.solve(G + lam * T * I, I)
    return W


def compute_sufficient_statistics(R, Y):
    """
    Compute sufficient statistics G = R^T R, H = Y^T R.

    Args:
        R: (T, d) features
        Y: (T, c) targets

    Returns:
        G: (d, d), H: (c, d), T: int
    """
    R = R.to(DEVICE)
    Y = Y.to(DEVICE)
    G = R.T @ R
    H = Y.T @ R
    T = R.shape[0]
    return G, H, T


def ridge_from_statistics(G, H, T, lam):
    """Solve ridge from aggregated statistics."""
    I = torch.eye(G.shape[0], device=DEVICE)
    W = H @ torch.linalg.solve(G + lam * T * I, I)
    return W


def discounted_statistics(R, Y, beta):
    """
    Compute exponentially discounted statistics.

    Args:
        R: (T, d), Y: (T, c), beta: forgetting factor
    """
    T, d = R.shape
    G = torch.zeros(d, d, device=DEVICE)
    H = torch.zeros(Y.shape[1], d, device=DEVICE)
    T_eff = 0.0
    for t in range(T):
        r = R[t:t+1]  # (1, d)
        y = Y[t:t+1]  # (1, c)
        G = beta * G + r.T @ r
        H = beta * H + y.T @ r
        T_eff = beta * T_eff + 1.0
    return G, H, T_eff


def personalized_readout(G_k, H_k, T_k, W_global, lam, mu):
    """
    Personalized readout with global-prior regularization.
    W_pers = (H_k + mu * T_k * W_global)(G_k + (lam + mu) * T_k * I)^{-1}
    """
    d = G_k.shape[0]
    I = torch.eye(d, device=DEVICE)
    numerator = H_k + mu * T_k * W_global
    denominator = G_k + (lam + mu) * T_k * I
    W_pers = numerator @ torch.linalg.solve(denominator, I)
    return W_pers
