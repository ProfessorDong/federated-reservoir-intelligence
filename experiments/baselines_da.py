"""
Domain-adaptation / transfer baselines for sEMG (Reviewer 1, point 2).

Two federated baselines tailored to cross-subject surface-EMG recognition, the
deep analogues of what FRI provides (a shared global model plus subject-specific
adaptation):

  * run_fedavg_finetune -- FedAvg-pretrained recurrent model, then per-client
    fine-tuning (deep transfer learning + subject/session adaptation). This is the
    direct deep counterpart of FRI-Stats+Personalization.
  * run_feddann -- federated Domain-Adversarial Neural Network (DANN): a shared
    GRU feature extractor with a label head and a gradient-reversed domain head,
    trained with FedAvg, where each client's data carries its subject id as the
    domain label. The adversarial head encourages subject-invariant features.

Both expose the same (acc, f1, comm_scalars, n_params) interface as run_fedavg.
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import DEVICE
from baselines import train_local, evaluate_model, fedavg_aggregate


# ── Gradient reversal ────────────────────────────────────────────────────────
class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd):
    return _GradReverse.apply(x, lambd)


class DANN(nn.Module):
    """GRU feature extractor + label head + gradient-reversed domain head."""

    def __init__(self, input_dim, hidden_dim, n_classes, n_domains, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.label_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_classes))
        self.domain_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_domains))

    def features(self, x):
        _, h = self.gru(x)
        return self.dropout(h[-1])

    def forward(self, x, lambd=0.0):
        h = self.features(x)
        return self.label_head(h), self.domain_head(grad_reverse(h, lambd))

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ── FedAvg + per-client fine-tuning (deep transfer / subject adaptation) ─────
def run_fedavg_finetune(model_fn, client_train_X, client_train_y,
                        client_test_X, client_test_y,
                        num_rounds=50, participation_rate=0.3,
                        local_epochs=5, lr=1e-3, batch_size=32,
                        finetune_epochs=20, finetune_lr=5e-4, seed=0):
    """FedAvg pretraining followed by per-client fine-tuning of the whole model.

    Communication equals standard FedAvg (fine-tuning is local, no extra uplink).
    """
    from baselines import run_fedavg
    global_model, metrics = run_fedavg(
        model_fn, client_train_X, client_train_y, client_test_X, client_test_y,
        num_rounds=num_rounds, participation_rate=participation_rate,
        local_epochs=local_epochs, lr=lr, batch_size=batch_size, seed=seed)

    accs, f1s = [], []
    for k in range(len(client_train_X)):
        local = copy.deepcopy(global_model)
        train_local(local, client_train_X[k], client_train_y[k],
                    epochs=finetune_epochs, lr=finetune_lr, batch_size=batch_size)
        if client_test_X[k] is not None and len(client_test_X[k]) > 0:
            a, f = evaluate_model(local, client_test_X[k], client_test_y[k])
            accs.append(a); f1s.append(f)
    return {
        'acc': float(np.mean(accs)) if accs else 0.0,
        'f1': float(np.mean(f1s)) if f1s else 0.0,
        'comm_scalars': metrics['comm_scalars'],   # fine-tuning is communication-free
        'n_params': metrics['n_params'],
    }


# ── Federated DANN ───────────────────────────────────────────────────────────
def _train_local_dann(model, train_X, train_y, domain_id, lambd,
                      epochs, lr, batch_size):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    n = train_X.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = train_X[idx].to(DEVICE); yb = train_y[idx].to(DEVICE)
            db = torch.full((xb.shape[0],), domain_id, dtype=torch.long, device=DEVICE)
            opt.zero_grad()
            y_logits, d_logits = model(xb, lambd)
            loss = ce(y_logits, yb) + ce(d_logits, db)
            loss.backward()
            opt.step()


def run_feddann(model_fn, client_train_X, client_train_y,
                client_test_X, client_test_y,
                num_rounds=50, participation_rate=0.3,
                local_epochs=5, lr=1e-3, batch_size=32,
                lambda_max=1.0, seed=0):
    """Federated DANN. Each client uses its index as the domain label; the
    gradient-reversed domain head (aggregated via FedAvg) drives subject-invariant
    features. lambda is ramped 0->lambda_max across rounds (standard DANN schedule).
    """
    rng = np.random.RandomState(seed)
    K = len(client_train_X)
    n_select = max(1, int(participation_rate * K))
    T_all = [x.shape[0] for x in client_train_X]

    global_model = model_fn().to(DEVICE)
    n_params = global_model.count_params()
    total_comm = 0

    for r in range(num_rounds):
        p = r / max(1, num_rounds - 1)
        lambd = lambda_max * (2.0 / (1.0 + np.exp(-10 * p)) - 1.0)   # DANN schedule
        selected = sorted(rng.choice(K, n_select, replace=False))
        T_sel = sum(T_all[k] for k in selected)
        client_models, weights = [], []
        for k in selected:
            local = copy.deepcopy(global_model)
            _train_local_dann(local, client_train_X[k], client_train_y[k],
                              domain_id=k, lambd=lambd, epochs=local_epochs,
                              lr=lr, batch_size=batch_size)
            client_models.append(local); weights.append(T_all[k] / T_sel)
            total_comm += n_params
        fedavg_aggregate(global_model, client_models, weights)
        total_comm += n_params

    # Evaluate label head
    global_model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for k in range(K):
            if client_test_X[k] is not None and len(client_test_X[k]) > 0:
                y_logits, _ = global_model(client_test_X[k].to(DEVICE), 0.0)
                preds.append(y_logits.argmax(1).cpu()); labels.append(client_test_y[k].cpu())
    from sklearn.metrics import accuracy_score, f1_score
    if preds:
        preds = torch.cat(preds).numpy(); labels = torch.cat(labels).numpy()
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='macro', zero_division=0)
    else:
        acc, f1 = 0.0, 0.0
    return {'acc': acc, 'f1': f1, 'comm_scalars': total_comm, 'n_params': n_params}
