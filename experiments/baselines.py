"""
Baseline models: LSTM, EEGNet, SNN, and federated training loops.
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import DEVICE, LSTM_CFG, EEGNET_CFG, SNN_CFG


# ═══════════════════════════════════════════════════════════════════════════════
# LSTM Baseline
# ═══════════════════════════════════════════════════════════════════════════════
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (batch, T, d_x)
        out, (h_n, _) = self.lstm(x)
        h = self.dropout(h_n[-1])  # last layer hidden state
        return self.fc(h)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ═══════════════════════════════════════════════════════════════════════════════
# EEGNet Baseline (for BCI)
# ═══════════════════════════════════════════════════════════════════════════════
class EEGNet(nn.Module):
    """
    EEGNet (Lawhern et al., 2018).
    Input: (batch, 1, n_channels, n_times)
    """
    def __init__(self, n_channels, n_times, n_classes,
                 F1=8, D=2, F2=16, dropout=0.25):
        super().__init__()
        self.F1 = F1
        self.D = D
        self.F2 = F2

        # Block 1: temporal convolution
        self.conv1 = nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        # Block 1: depthwise spatial convolution
        self.conv2 = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)

        # Block 2: separable convolution
        self.conv3 = nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8),
                               groups=F1 * D, bias=False)
        self.conv4 = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)

        # Classifier
        # Calculate output size
        t_out = n_times // 32  # after two pooling layers
        self.fc = nn.Linear(F2 * t_out, n_classes)

    def forward(self, x):
        # x: (batch, n_channels, n_times) or (batch, 1, n_channels, n_times)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)
        x = x.flatten(1)
        return self.fc(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ═══════════════════════════════════════════════════════════════════════════════
# SNN Baseline (surrogate gradient)
# ═══════════════════════════════════════════════════════════════════════════════
class SurrogateSpikeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        # Fast sigmoid surrogate gradient
        grad = grad_output / (1 + 100 * x.abs()) ** 2
        return grad


surrogate_spike = SurrogateSpikeFunction.apply


class SNNClassifier(nn.Module):
    """Simple feedforward SNN with surrogate gradient."""

    def __init__(self, input_dim, hidden_dim, num_classes, num_steps=50, beta=0.95):
        super().__init__()
        self.num_steps = num_steps
        self.beta = beta
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (batch, T, d_x) — process T steps through SNN
        batch_size = x.shape[0]
        T = min(x.shape[1], self.num_steps)

        mem1 = torch.zeros(batch_size, self.fc1.out_features, device=x.device)
        mem2 = torch.zeros(batch_size, self.fc2.out_features, device=x.device)
        out_acc = torch.zeros(batch_size, self.fc2.out_features, device=x.device)

        for t in range(T):
            cur1 = self.fc1(x[:, t])
            mem1 = self.beta * mem1 + cur1
            spk1 = surrogate_spike(mem1 - 1.0)
            mem1 = mem1 * (1 - spk1)  # reset

            cur2 = self.fc2(spk1)
            mem2 = self.beta * mem2 + cur2
            out_acc += mem2

        return out_acc / T

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ═══════════════════════════════════════════════════════════════════════════════
# Additional deep-learning baselines (revision; Reviewer 1, point 1)
# GRU, 1D temporal CNN, and a Temporal Convolutional Network (TCN).
# All consume (batch, T, d_x) and expose count_params(), matching the
# LSTM/SNN baselines so they plug directly into run_fedavg / run_local_only.
# ═══════════════════════════════════════════════════════════════════════════════
class GRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=1, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out, h_n = self.gru(x)
        return self.fc(self.dropout(h_n[-1]))

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


class CNN1DClassifier(nn.Module):
    """1D temporal CNN over (batch, T, d_x); d_x are the input channels."""

    def __init__(self, input_dim, num_classes, channels=64, kernel_size=7, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(channels), nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(channels), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)               # (batch, d_x, T)
        h = self.net(x).squeeze(-1)         # (batch, channels)
        return self.fc(self.dropout(h))

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


class _Chomp1d(nn.Module):
    def __init__(self, chomp):
        super().__init__()
        self.chomp = chomp

    def forward(self, x):
        return x[:, :, :-self.chomp].contiguous() if self.chomp > 0 else x


class _TemporalBlock(nn.Module):
    def __init__(self, n_in, n_out, k, dilation, dropout):
        super().__init__()
        pad = (k - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(n_in, n_out, k, padding=pad, dilation=dilation), _Chomp1d(pad),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(n_out, n_out, k, padding=pad, dilation=dilation), _Chomp1d(pad),
            nn.ReLU(), nn.Dropout(dropout),
        )
        self.down = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.down is None else self.down(x)
        return self.relu(out + res)


class TCNClassifier(nn.Module):
    """Temporal Convolutional Network (dilated causal convolutions)."""

    def __init__(self, input_dim, num_classes, channels=64, levels=4,
                 kernel_size=5, dropout=0.2):
        super().__init__()
        layers, ch_in = [], input_dim
        for i in range(levels):
            layers.append(_TemporalBlock(ch_in, channels, kernel_size, 2 ** i, dropout))
            ch_in = channels
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)               # (batch, d_x, T)
        h = self.tcn(x).mean(dim=2)         # global average pool over time
        return self.fc(h)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ═══════════════════════════════════════════════════════════════════════════════
# Federated training loops
# ═══════════════════════════════════════════════════════════════════════════════
def train_local(model, train_X, train_y, epochs=5, lr=1e-3, batch_size=32):
    """Train model locally for E epochs."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    n = train_X.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            xb = train_X[idx].to(DEVICE)
            yb = train_y[idx].to(DEVICE)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()


def train_local_fedprox(model, train_X, train_y, global_params,
                        mu_prox=0.01, epochs=5, lr=1e-3, batch_size=32):
    """Train with FedProx proximal term."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    n = train_X.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            xb = train_X[idx].to(DEVICE)
            yb = train_y[idx].to(DEVICE)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            # Proximal term
            prox = 0.0
            for p, gp in zip(model.parameters(), global_params):
                prox += (p - gp).pow(2).sum()
            loss += (mu_prox / 2) * prox
            loss.backward()
            optimizer.step()


def fedavg_aggregate(global_model, client_models, client_weights):
    """Aggregate client models by weighted averaging."""
    global_dict = global_model.state_dict()
    for key in global_dict:
        global_dict[key] = torch.zeros_like(global_dict[key], dtype=torch.float32)
        for i, cm in enumerate(client_models):
            global_dict[key] += client_weights[i] * cm.state_dict()[key].float()
    global_model.load_state_dict(global_dict)


def run_fedavg(model_fn, client_train_X, client_train_y,
               client_test_X, client_test_y,
               num_rounds=50, participation_rate=0.5,
               local_epochs=5, lr=1e-3, batch_size=32,
               fedprox_mu=0.0, seed=0):
    """
    Run FedAvg or FedProx.

    Returns:
        global_model, metrics dict
    """
    rng = np.random.RandomState(seed)
    K = len(client_train_X)
    n_select = max(1, int(participation_rate * K))
    T_all = [x.shape[0] for x in client_train_X]

    global_model = model_fn().to(DEVICE)
    total_comm = 0
    n_params = global_model.count_params()

    best_acc = 0.0
    for r in range(num_rounds):
        selected = sorted(rng.choice(K, n_select, replace=False))
        T_sel = sum(T_all[k] for k in selected)

        client_models = []
        weights = []
        for k in selected:
            local_model = copy.deepcopy(global_model)
            if fedprox_mu > 0:
                global_params = [p.clone().detach() for p in global_model.parameters()]
                train_local_fedprox(local_model, client_train_X[k], client_train_y[k],
                                    global_params, fedprox_mu,
                                    local_epochs, lr, batch_size)
            else:
                train_local(local_model, client_train_X[k], client_train_y[k],
                            local_epochs, lr, batch_size)
            client_models.append(local_model)
            weights.append(T_all[k] / T_sel)
            total_comm += n_params  # upload

        fedavg_aggregate(global_model, client_models, weights)
        total_comm += n_params  # broadcast

    # Evaluate
    global_model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for k in range(K):
            if client_test_X[k] is not None and len(client_test_X[k]) > 0:
                out = global_model(client_test_X[k].to(DEVICE))
                preds = out.argmax(dim=1).cpu()
                all_preds.append(preds)
                all_labels.append(client_test_y[k].cpu())

    if all_preds:
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        from sklearn.metrics import accuracy_score, f1_score
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    else:
        acc, f1 = 0.0, 0.0

    return global_model, {
        'acc': acc, 'f1': f1,
        'comm_scalars': total_comm,
        'n_params': n_params,
    }


def evaluate_model(model, test_X, test_y):
    """Evaluate a model on test data."""
    model.eval()
    from sklearn.metrics import accuracy_score, f1_score
    with torch.no_grad():
        out = model(test_X.to(DEVICE))
        preds = out.argmax(dim=1).cpu().numpy()
    labels = test_y.cpu().numpy()
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro', zero_division=0)
    return acc, f1


def run_local_only(model_fn, client_train_X, client_train_y,
                   client_test_X, client_test_y,
                   local_epochs=5, lr=1e-3, batch_size=32, seed=0):
    """Train an INDEPENDENT model per client with no federation (Reviewer 1.1).

    Reports the mean per-client test accuracy/F1 and zero communication, the
    natural no-communication reference for the federated deep baselines.
    """
    torch.manual_seed(seed)
    n_params = model_fn().count_params()
    accs, f1s = [], []
    for k in range(len(client_train_X)):
        model = model_fn().to(DEVICE)
        train_local(model, client_train_X[k], client_train_y[k],
                    local_epochs, lr, batch_size)
        if client_test_X[k] is not None and len(client_test_X[k]) > 0:
            acc, f1 = evaluate_model(model, client_test_X[k], client_test_y[k])
            accs.append(acc); f1s.append(f1)
    return {
        'acc': float(np.mean(accs)) if accs else 0.0,
        'f1': float(np.mean(f1s)) if f1s else 0.0,
        'comm_scalars': 0,
        'n_params': n_params,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# pFedMe (Dinh et al., NeurIPS 2020) — Personalized FL with Moreau Envelopes
# ═══════════════════════════════════════════════════════════════════════════════
def run_pfedme(model_fn, client_train_X, client_train_y,
               client_test_X, client_test_y,
               num_rounds=50, participation_rate=0.5,
               local_epochs=5, lr=1e-3, lambd=15.0,
               beta=1.0, batch_size=32, seed=0):
    """
    pFedMe: Personalized Federated Learning with Moreau Envelopes.
    (Dinh et al., NeurIPS 2020)

    Each client solves: min_w F_k(w) + (lambd/2)||w - theta||^2
    Server update: theta <- (1-beta)*theta + beta * avg(w_k)

    Returns:
        global_model, metrics dict
    """
    rng = np.random.RandomState(seed)
    K = len(client_train_X)
    n_select = max(1, int(participation_rate * K))
    T_all = [x.shape[0] for x in client_train_X]

    global_model = model_fn().to(DEVICE)
    total_comm = 0
    n_params = global_model.count_params()

    # Store personalized models for each client
    personal_models = [copy.deepcopy(global_model) for _ in range(K)]

    for r in range(num_rounds):
        selected = sorted(rng.choice(K, n_select, replace=False))
        T_sel = sum(T_all[k] for k in selected)

        client_models = []
        weights = []
        for k in selected:
            # Initialize from global model
            local_model = copy.deepcopy(global_model)
            global_params = [p.clone().detach() for p in global_model.parameters()]

            # Proximal SGD: min_w F_k(w) + (lambd/2)||w - theta||^2
            local_model.train()
            optimizer = torch.optim.Adam(local_model.parameters(), lr=lr)
            X_k = client_train_X[k].to(DEVICE)
            y_k = client_train_y[k].to(DEVICE)
            n_samples = X_k.shape[0]

            for epoch in range(local_epochs):
                indices = torch.randperm(n_samples, device=DEVICE)
                for start in range(0, n_samples, batch_size):
                    batch_idx = indices[start:start + batch_size]
                    batch_X = X_k[batch_idx]
                    batch_y = y_k[batch_idx]

                    optimizer.zero_grad()
                    out = local_model(batch_X)
                    loss = torch.nn.functional.cross_entropy(out, batch_y)

                    # Add Moreau envelope proximal term
                    prox_term = 0.0
                    for p, gp in zip(local_model.parameters(), global_params):
                        prox_term += ((p - gp) ** 2).sum()
                    loss = loss + (lambd / 2.0) * prox_term

                    loss.backward()
                    optimizer.step()

            client_models.append(local_model)
            personal_models[k] = copy.deepcopy(local_model)
            weights.append(T_all[k] / T_sel)
            total_comm += n_params  # upload

        # Server update: theta <- (1-beta)*theta + beta * avg(w_k)
        avg_dict = {}
        global_dict = global_model.state_dict()
        for key in global_dict:
            avg_dict[key] = torch.zeros_like(global_dict[key], dtype=torch.float32)
            for i, cm in enumerate(client_models):
                avg_dict[key] += weights[i] * cm.state_dict()[key].float()
            avg_dict[key] = (1 - beta) * global_dict[key].float() + beta * avg_dict[key]
        global_model.load_state_dict(avg_dict)
        total_comm += n_params  # broadcast

    # Final personalization: each client does one more proximal solve
    for k in range(K):
        local_model = copy.deepcopy(global_model)
        global_params = [p.clone().detach() for p in global_model.parameters()]
        local_model.train()
        optimizer = torch.optim.Adam(local_model.parameters(), lr=lr)
        X_k = client_train_X[k].to(DEVICE)
        y_k = client_train_y[k].to(DEVICE)
        n_samples = X_k.shape[0]
        for epoch in range(local_epochs):
            indices = torch.randperm(n_samples, device=DEVICE)
            for start in range(0, n_samples, batch_size):
                batch_idx = indices[start:start + batch_size]
                optimizer.zero_grad()
                out = local_model(X_k[batch_idx])
                loss = torch.nn.functional.cross_entropy(out, y_k[batch_idx])
                prox_term = 0.0
                for p, gp in zip(local_model.parameters(), global_params):
                    prox_term += ((p - gp) ** 2).sum()
                loss = loss + (lambd / 2.0) * prox_term
                loss.backward()
                optimizer.step()
        personal_models[k] = local_model

    # Evaluate personalized models
    all_preds, all_labels = [], []
    for k in range(K):
        personal_models[k].eval()
        with torch.no_grad():
            if client_test_X[k] is not None and len(client_test_X[k]) > 0:
                out = personal_models[k](client_test_X[k].to(DEVICE))
                preds = out.argmax(dim=1).cpu()
                all_preds.append(preds)
                all_labels.append(client_test_y[k].cpu())

    if all_preds:
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        from sklearn.metrics import accuracy_score, f1_score
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    else:
        acc, f1 = 0.0, 0.0

    return global_model, {
        'acc': acc, 'f1': f1,
        'comm_scalars': total_comm,
        'n_params': n_params,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FedTL-EEG — Transfer learning baseline for BCI
# ═══════════════════════════════════════════════════════════════════════════════
def run_fedtl_eeg(model_fn, client_train_X, client_train_y,
                  client_test_X, client_test_y,
                  num_rounds=50, participation_rate=0.5,
                  local_epochs=5, lr=1e-3, batch_size=32,
                  ft_epochs=10, ft_lr=5e-4, seed=0):
    """
    FedTL-EEG: FedAvg pre-training on source subjects, then
    fine-tune feature extractor on target subjects.
    """
    rng = np.random.RandomState(seed)
    K = len(client_train_X)

    # Phase 1: FedAvg pre-training (same as standard)
    global_model, fed_metrics = run_fedavg(
        model_fn, client_train_X, client_train_y,
        client_test_X, client_test_y,
        num_rounds=num_rounds, participation_rate=participation_rate,
        local_epochs=local_epochs, lr=lr, batch_size=batch_size,
        seed=seed)

    total_comm = fed_metrics['comm_scalars']

    # Phase 2: Per-client fine-tuning (freeze feature extractor, fine-tune classifier)
    all_preds, all_labels = [], []
    for k in range(K):
        ft_model = copy.deepcopy(global_model)
        # Fine-tune all parameters with smaller learning rate
        if len(client_train_X[k]) > 0:
            train_local(ft_model, client_train_X[k], client_train_y[k],
                        ft_epochs, ft_lr, batch_size)
        ft_model.eval()
        with torch.no_grad():
            if client_test_X[k] is not None and len(client_test_X[k]) > 0:
                out = ft_model(client_test_X[k].to(DEVICE))
                preds = out.argmax(dim=1).cpu()
                all_preds.append(preds)
                all_labels.append(client_test_y[k].cpu())

    if all_preds:
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        from sklearn.metrics import accuracy_score, f1_score
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    else:
        acc, f1 = 0.0, 0.0

    return global_model, {
        'acc': acc, 'f1': f1,
        'comm_scalars': total_comm,
        'n_params': fed_metrics['n_params'],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# LFNL-like baseline (Yang et al., Nature Comms 2022)
# Federated SNN with local learning + readout aggregation
# ═══════════════════════════════════════════════════════════════════════════════
def run_lfnl(model_fn, client_train_X, client_train_y,
             client_test_X, client_test_y,
             num_rounds=50, participation_rate=0.3,
             local_epochs=5, lr=1e-3, batch_size=32,
             seed=0):
    """
    LFNL-style: SNN with federated aggregation of only the readout layer,
    local (non-federated) update of hidden layers.
    """
    rng = np.random.RandomState(seed)
    K = len(client_train_X)
    n_select = max(1, int(participation_rate * K))
    T_all = [x.shape[0] for x in client_train_X]

    global_model = model_fn().to(DEVICE)
    n_params = global_model.count_params()
    # Only count readout layer params for communication
    readout_params = sum(p.numel() for p in global_model.fc2.parameters())
    total_comm = 0

    for r in range(num_rounds):
        selected = sorted(rng.choice(K, n_select, replace=False))
        T_sel = sum(T_all[k] for k in selected)

        client_models = []
        weights = []
        for k in selected:
            local_model = copy.deepcopy(global_model)
            # Train all layers locally
            train_local(local_model, client_train_X[k], client_train_y[k],
                        local_epochs, lr, batch_size)
            client_models.append(local_model)
            weights.append(T_all[k] / T_sel)
            total_comm += readout_params  # only readout uploaded

        # Aggregate only the readout layer (fc2)
        global_dict = global_model.state_dict()
        for key in global_dict:
            if 'fc2' in key:
                global_dict[key] = torch.zeros_like(global_dict[key], dtype=torch.float32)
                for i, cm in enumerate(client_models):
                    global_dict[key] += weights[i] * cm.state_dict()[key].float()
        global_model.load_state_dict(global_dict)
        total_comm += readout_params  # broadcast readout

    # Evaluate with per-client hidden layers (take from last local model)
    # For fair comparison, fine-tune hidden layers locally then evaluate
    all_preds, all_labels = [], []
    for k in range(K):
        local_model = copy.deepcopy(global_model)
        if len(client_train_X[k]) > 0:
            train_local(local_model, client_train_X[k], client_train_y[k],
                        local_epochs, lr, batch_size)
        local_model.eval()
        with torch.no_grad():
            if client_test_X[k] is not None and len(client_test_X[k]) > 0:
                out = local_model(client_test_X[k].to(DEVICE))
                preds = out.argmax(dim=1).cpu()
                all_preds.append(preds)
                all_labels.append(client_test_y[k].cpu())

    if all_preds:
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        from sklearn.metrics import accuracy_score, f1_score
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    else:
        acc, f1 = 0.0, 0.0

    return global_model, {
        'acc': acc, 'f1': f1,
        'comm_scalars': total_comm,
        'n_params': n_params,
        'readout_params': readout_params,
    }
