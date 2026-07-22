# Federated Reservoir Intelligence (FRI) — Code

Reference implementation for the manuscript *"Federated reservoir intelligence for distributed temporal and neuromorphic learning"*, Communications Engineering, manuscript ID **COMMSENG-26-0313-T**.

This README is structured to match the Nature Code and Software Submission Checklist.

## Overview

Implements and evaluates Federated Reservoir Intelligence: a federated-learning framework in which each client runs a fixed reservoir (Echo State Network for continuous signals or Liquid State Machine for spike-driven data) and federates only a low-dimensional linear readout or its sufficient statistics. The codebase covers reservoir construction, statistics-based and readout-based federated aggregation, drift adaptation, personalization, all baselines used in the manuscript (FedAvg/FedProx LSTM, FedAvg-EEGNet, pFedMe, FedTL-EEG, FedAvg/FedProx-SNN, LFNL), and the cross-cutting ablations (reservoir-dimension sweep, participation rate, heterogeneous reservoirs, differential privacy, training-fraction data efficiency).

## License

MIT License. See top-level [`LICENSE`](../LICENSE).

## 1. System requirements

### Software dependencies and operating systems (with version numbers)

| Dependency | Version range | Purpose |
|---|---|---|
| Python | 3.10 or newer (tested on 3.10, 3.12) | Interpreter |
| numpy | ≥ 1.23, < 2.0 | Numerical arrays |
| scipy | ≥ 1.10, < 2.0 | Linear algebra, statistics |
| torch (PyTorch) | ≥ 2.0, < 3.0 | Neural-network baselines (LSTM, EEGNet, SNN) and tensor ops |
| scikit-learn | ≥ 1.2, < 2.0 | Accuracy/F1 metrics, stratified splits |
| mne | ≥ 1.4, < 2.0 | EEG data loading (BCI-IV-2a/2b GDF files) |

All listed in [`requirements.txt`](requirements.txt). All have prebuilt wheels on PyPI; no compilation step is required.

### Versions the software has been tested on

- **OS**: Ubuntu 24.04 LTS, kernel 6.11.0-29-generic
- **Python**: 3.12.x (also verified 3.10.x)
- **PyTorch**: 2.4.0 (CPU and CUDA 12.x)
- **NumPy**: 1.26.4
- **SciPy**: 1.13.1
- **MNE**: 1.7.x
- **scikit-learn**: 1.5.x

### Required non-standard hardware

**None.** A standard x86-64 desktop or laptop is sufficient. CPU-only execution is fully supported; the code automatically falls back to CPU when no GPU is available.

- **Recommended (not required)**: an NVIDIA GPU with CUDA 11.8+ (any RTX 30/40/PRO series). A GPU is not needed for the core FRI experiments (ESN + ridge readout) but accelerates the LSTM/EEGNet/SNN baselines by roughly 5–20×.
- **RAM**: 8 GB minimum. 32 GB recommended for DVS128 (event loading uses ~7.4 GB).
- **Disk**: ~15 GB for the full dataset cache (DVS128 alone is ~12 GB). The synthetic demo uses no external data.

## 2. Installation guide

### Instructions

```bash
# 1. Place the unzipped code in any working directory.
cd experiments/

# 2. Create a Python virtual environment (recommended).
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows

# 3. Install dependencies.
pip install -r requirements.txt
```

Alternative: with conda,
```bash
conda create -n fri python=3.10 && conda activate fri
pip install -r requirements.txt
```

### Typical install time on a normal desktop computer

- **5–10 minutes** on a fresh environment with broadband (PyTorch wheel is the largest at ~800 MB).
- **8–15 minutes** with conda environment creation.
- **No compilation** — all wheels are prebuilt for x86-64 Linux/macOS/Windows.

## 3. Demo

### Instructions to run the demo

The synthetic demo runs end-to-end on CPU with no external data:

```bash
cd experiments/
python demo_synthetic.py
```

### Expected output

```
[FRI synthetic demo]
  K=4 clients, T_k=240 samples each, N_r=80 reservoir, d_y=3 outputs
  Generating synthetic time-series data with shared reservoir...
[1/4] Local-only ridge readout (no federation):
  Mean test accuracy: 1.0000 ± 0.0000
[2/4] Centralized ridge (oracle, all data pooled):
  Test accuracy: 1.0000
[3/4] FRI-Stats (one-shot federated aggregation):
  Test accuracy: 1.0000
  Communication: 0.05 MiB
  Recovers centralized solution exactly: True
[4/4] FRI-Stats + personalization (mu=0.01):
  Mean test accuracy: 1.0000 ± 0.0000
Demo complete. Total runtime: 0.5 seconds on CPU.
```

The line `Recovers centralized solution exactly: True` empirically verifies the manuscript's central theoretical claim — that statistics-based FRI aggregation is bit-equivalent to centralized ridge regression under shared feature coordinates and full participation.

### Expected run time for demo on a normal desktop computer

- **CPU (any modern x86-64)**: 0.5–2 seconds
- **GPU**: comparable; the synthetic problem is too small to benefit from GPU

## 4. Instructions for use

### How to run the software on your data

#### Step 1: Place datasets

Datasets are not bundled (BCI/Ninapro require dataset-provider registration; DVS128 is ~12 GB). Place each dataset under the following paths (defined in [`config.py`](config.py)):

| Dataset | Source | Local path |
|---|---|---|
| BCI Competition IV-2a | BNCI Horizon 2020, `https://bnci-horizon-2020.eu/database/data-sets` | `../data/BCICompetition/DataSet2/extracted/` |
| BCI Competition IV-2b | BNCI Horizon 2020 (same source) | `../data/BCICompetition/DataSet4/extracted/` |
| Ninapro DB5 | `https://ninapro.hevs.ch/instructions/DB5.html` | `../data/Ninapro/DB5/extracted/` |
| DVS128 Gesture | IBM Research, `https://research.ibm.com/interactive/dvsgesture/` | `../data/DVS128/extracted/` |
| N-Caltech101 | `https://www.garrickorchard.com/datasets/n-caltech101` | `../data/N-Caltech101/extracted/` |

#### Step 2: Run an experiment

```bash
cd experiments/

# Cross-subject classification on EEG and sEMG (Tables 2, 3, S4 of the manuscript)
python run_bci_iv2a.py
python run_ninapro.py

# Event-camera gesture recognition (Table 4)
python run_dvs128.py
python run_fri_esn_events.py

# Cross-cutting ablations (Tables S5–S7, Figure 3)
python run_ablation.py

# All experiments sequentially
python run_all.py
```

Each script writes results to `../results/<dataset>_results.json`.

#### Step 3: Adapt to your own data

To run FRI on a new dataset, write a loader returning per-client `(X_train, y_train, X_test, y_test)` where each is a list of K tensors (one per client). Then build an ESN, compute reservoir states or pooled features, and call:

```python
from reservoir import ESN
from federated import FRIStatsAggregator, evaluate_readout

esn = ESN(input_dim=d_x, reservoir_dim=N_r, output_dim=d_y, ...)
client_features = [esn.run(X_k).mean(dim=1) for X_k in client_X_train]   # (n_trials_k, N_r)
client_labels_oh = [labels_to_onehot(y_k, d_y) for y_k in client_y_train]

W_global, comm_scalars = FRIStatsAggregator(ridge_lambda=0.05).run(client_features, client_labels_oh)
acc, f1 = evaluate_readout(W_global, test_features_k, test_y_k)
```

For personalization: use `FRIStatsAggregator.run_with_personalization(...)`.
For drift: see `FRIDriftAggregator` in [`federated.py`](federated.py).
For readout-based aggregation: `FRIReadoutAggregator` and `FRIReadoutProximalAggregator` in the same file.

## 5. Reproduction instructions

To reproduce the manuscript's headline numerical results:

| Manuscript table | Script | Approx. CPU runtime |
|---|---|---|
| Table 2 (BCI-IV-2a + Ninapro main results) | `run_bci_iv2a.py`, `run_ninapro.py` | 12–18 + 8–12 min |
| Table 3 (BCI-IV-2b cross-session, β sweep, held-out split) | `run_drift_heldout.py` | ~5 min |
| Table 4 (DVS128) + Tables S1, S2 | `run_dvs128.py` then `run_fri_esn_events.py` | 25–40 min CPU; 8–15 min GPU |
| Table S4 (personalization sweep) | `run_bci_iv2a.py` | included above |
| Tables S5–S7 (reservoir dim, participation, heterogeneous, generalization) | `run_ablation.py` | 5–10 min |
| Privacy scaling (Fig. 3d, Supplementary privacy table; parameter-MSE O(1/T_k²), Prop. 2) | `run_privacy_scaling.py` | ~3 min |
| Figures 2, 3 | `../generate_figures.py` | < 30 s |

The precomputed result files used in the manuscript are provided in the top-level `results/` directory (one JSON per experiment). `../generate_figures.py` reads these directly, so Figures 2 and 3 can be regenerated without re-running the experiments; re-running any `run_*.py` script overwrites the corresponding JSON.

Random seeds for the main experiments: `[0, 1, 2, 3, 4]` (5 seeds), defined in `config.SEEDS`. For ablations: `[0, 1, 2]` (3 seeds), in `config.ABLATION_SEEDS`. All hyperparameters are in [`config.py`](config.py); the values shown in the manuscript Methods correspond exactly to the constants in this file.

## File map

```
experiments/
├── config.py                 # Hyperparameters, paths, random seeds, device
├── reservoir.py              # ESN, LSM, ridge_regression, sufficient_statistics
├── federated.py              # FRIStatsAggregator, FRIReadoutAggregator (+proximal),
│                             #   FRIDriftAggregator, evaluate_readout, comm_cost_mb
├── baselines.py              # LSTMClassifier, EEGNet, SNNClassifier,
│                             #   FedAvg, FedProx, pFedMe, FedTL-EEG, LFNL
├── data_bci.py               # BCI-IV-2a (GDF/MNE), Ninapro DB5 (MAT)
├── data_bci_extra.py         # BCI-IV-2b (3-session drift)
├── data_event.py             # DVS128 (AEDAT3.1), N-Caltech101 (binary),
│                             #   event→frame and event→spike encodings
├── run_bci_iv2a.py           # BCI-IV-2a main + drift + personalization sweep
├── run_ninapro.py            # Ninapro DB5
├── run_dvs128.py             # DVS128 Gesture (FRI-LSM and baselines)
├── run_ncaltech101.py        # N-Caltech101
├── run_fri_esn_events.py     # FRI-ESN frame-based scaling on DVS128
├── run_deeper_readout.py     # MLP readout on FRI-LSM
├── run_ablation.py           # Reservoir dim, participation, heterogeneous,
│                             #   privacy, generalization
├── run_all.py                # Sequential wrapper
├── demo_synthetic.py         # Self-contained synthetic-data demo
├── requirements.txt          # Pinned Python dependencies
├── README.md                 # This file
├── INSTALL.md                # Detailed installation guide
└── DEMO.md                   # Detailed demo guide
```

## Citation and contact

Citation will be updated upon publication. During peer review, contact the corresponding author via the journal editorial office (Communications Engineering, MS ID COMMSENG-26-0313-T).
