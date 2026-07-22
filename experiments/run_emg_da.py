"""
sEMG domain-adaptation baselines on Ninapro DB5 (Reviewer 1, point 2).

Compares FRI against deep transfer / domain-adaptation methods appropriate to
cross-subject EMG recognition:
  * fedavg_gru_finetune -- FedAvg + per-client fine-tuning (deep transfer +
    subject adaptation); the deep analogue of FRI-Stats+Personalization.
  * feddann -- federated Domain-Adversarial Neural Network (subject-invariant
    features via gradient reversal).

Results (n = 5 seeds) saved to ../results/emg_da_results.json.
"""
import sys, os, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from config import SEEDS, LSTM_CFG, FED_CFG, RESULTS_DIR
from baselines_da import DANN, run_fedavg_finetune, run_feddann
from baselines import GRUClassifier
from federated import comm_cost_mb


def split_train_test(client_data, client_labels, test_ratio=0.2, seed=0):
    trX, teX, trY, teY = [], [], [], []
    for X, y in zip(client_data, client_labels):
        n = len(y); rng = np.random.RandomState(seed); idx = rng.permutation(n)
        nt = max(1, int(n * test_ratio))
        trX.append(X[idx[nt:]]); teX.append(X[idx[:nt]])
        trY.append(y[idx[nt:]]); teY.append(y[idx[:nt]])
    return trX, teX, trY, teY


def main():
    from data_bci import load_ninapro_db5_all, ninapro_to_federated
    print("Loading Ninapro DB5 ...")
    cdata, clabels = ninapro_to_federated(load_ninapro_db5_all())
    all_labels = torch.cat([y if torch.is_tensor(y) else torch.tensor(y) for y in clabels])
    n_classes = int(all_labels.max().item()) + 1
    K = len(cdata)
    h = LSTM_CFG['hidden_dim']
    out = {}
    for seed in SEEDS:
        print(f"\n-- seed {seed} --")
        trX, teX, trY, teY = split_train_test(cdata, clabels, seed=seed)
        d_x = trX[0].shape[2]
        res = {}

        m = run_fedavg_finetune(
            lambda: GRUClassifier(d_x, h, n_classes, num_layers=LSTM_CFG['num_layers'],
                                  dropout=LSTM_CFG['dropout']),
            trX, trY, teX, teY, num_rounds=FED_CFG['num_rounds'],
            participation_rate=FED_CFG['participation_ninapro'],
            local_epochs=LSTM_CFG['local_epochs'], lr=LSTM_CFG['lr'],
            batch_size=LSTM_CFG['batch_size'], finetune_epochs=20, seed=seed)
        res['fedavg_gru_finetune'] = {'acc': m['acc'], 'f1': m['f1'],
                                      'comm_mb': comm_cost_mb(m['comm_scalars']),
                                      'n_params': m['n_params']}

        m = run_feddann(
            lambda: DANN(d_x, h, n_classes, n_domains=K, dropout=LSTM_CFG['dropout']),
            trX, trY, teX, teY, num_rounds=FED_CFG['num_rounds'],
            participation_rate=FED_CFG['participation_ninapro'],
            local_epochs=LSTM_CFG['local_epochs'], lr=LSTM_CFG['lr'],
            batch_size=LSTM_CFG['batch_size'], seed=seed)
        res['feddann'] = {'acc': m['acc'], 'f1': m['f1'],
                          'comm_mb': comm_cost_mb(m['comm_scalars']),
                          'n_params': m['n_params']}

        for k, v in res.items():
            print(f"    {k:>22}: acc={v['acc']:.4f}  comm={v['comm_mb']:.2f} MB")
        out[str(seed)] = res

    path = os.path.join(RESULTS_DIR, 'emg_da_results.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {path}")
    # summary
    for key in ('fedavg_gru_finetune', 'feddann'):
        accs = [out[str(s)][key]['acc'] for s in SEEDS]
        m = np.mean(accs); ci = 1.96 * np.std(accs) / np.sqrt(len(accs))
        print(f"  {key}: {m:.4f} +/- {ci:.4f}")


if __name__ == '__main__':
    main()
