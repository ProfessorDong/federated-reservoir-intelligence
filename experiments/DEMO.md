# Demo guide

## Self-contained synthetic demo

`demo_synthetic.py` runs the full FRI pipeline end-to-end on synthetic data with no external dataset required. It demonstrates:

1. ESN reservoir construction (fixed random recurrent weights, leaky integration)
2. Local sufficient-statistics computation (Gram matrix `G_k`, cross-correlation `H_k`)
3. Statistics-based federated aggregation (one-shot exact recovery of centralized ridge solution)
4. Readout-based federated aggregation (proximal averaging over multiple rounds)
5. Personalization with global-prior regularization
6. Communication-cost accounting in MiB

### Run it

```bash
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

The synthetic task uses class-specific sequence templates plus Gaussian noise, designed to be cleanly learnable so the four methods produce easily-comparable numbers. The two reproducible properties are:

1. **`Recovers centralized solution exactly: True`** — this is the key theoretical claim of the paper. FRI-Stats one-shot aggregation is bit-equivalent to centralized ridge regression under shared feature coordinates and full participation.
2. **All four methods reach high accuracy** on this learnable synthetic task; the relative ordering on real datasets (where local-only typically lags centralized, and personalization recovers most of the gap) is documented in the paper's main results.

### Typical runtime

- **CPU (any modern x86-64):** 15–30 seconds
- **GPU:** comparable or slightly faster; reservoir update is the bottleneck

### What the demo does not do

The synthetic demo is for verifying installation correctness and code logic. It does not:
- Use real EEG/sEMG/event-camera datasets (those require separate downloads; see `README.md`)
- Reproduce the paper's specific numerical results (e.g., the 48.1% BCI-IV-2a accuracy)
- Run baselines (LSTM/EEGNet/SNN/LFNL) — those require labeled real data and longer training

## Reproducing the manuscript's headline experiments

After installing dependencies and downloading the relevant dataset (see `README.md` § Datasets):

| Headline result | Command | Approx. runtime (CPU) |
|---|---|---|
| BCI-IV-2a Table 2 (48.1% FRI-Stats+Pers, 5 seeds) | `python run_bci_iv2a.py` | 12–18 min |
| BCI-IV-2b Table 3 (drift, β sweep) | `python run_bci_iv2a.py` (same script, drift section) | included above |
| Ninapro DB5 Table 2 (88.7% FRI-Stats+Pers) | `python run_ninapro.py` | 8–12 min |
| DVS128 Table 4 (FRI-LSM 42.4%, FRI-ESN 78.9%) | `python run_dvs128.py && python run_fri_esn_events.py` | 25–40 min on CPU; 8–15 min on RTX 4090 |
| Ablations (Fig 3, Tables S4-S7) | `python run_ablation.py` | 5–10 min |
| All experiments | `python run_all.py` | 60–90 min total on CPU |

Each script writes a JSON results file to `../results/`. The figures are then regenerated from those JSONs:

```bash
cd ..
python generate_figures.py
```
