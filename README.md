# Federated Reservoir Intelligence (FRI)

Reference implementation for the Article:

> **Federated reservoir intelligence for distributed temporal and neuromorphic learning**
> Liang Dong. *Communications Engineering* (Nature Portfolio), 2026. Accepted for publication.

FRI couples federated learning with reservoir computing: each client runs a **fixed, untrained**
recurrent reservoir (an Echo State Network for continuous signals, or a Liquid State Machine for
spike-driven data) and federates only a compact **linear readout** or its **sufficient statistics**.
This makes per-round communication independent of the recurrent-weight count and of the temporal
unrolling, and removes the backward pass through temporal and spiking dynamics.

Key properties reproduced by this code:

- **One-shot exact recovery** of the centralized ridge solution by statistics-based aggregation,
  under shared feature coordinates and full participation.
- **Closed-form personalization** via global-prior regularization, at no extra communication.
- **Closed-form drift adaptation** through exponentially discounted sufficient statistics.
- **No backpropagation** anywhere in the FRI training path.

## Repository contents

```
experiments/          methods, baselines, data loaders, and run scripts
  reservoir.py        ESN/LSM construction, ridge readout, sufficient statistics
  federated.py        statistics- and readout-based aggregation, proximal and drift variants
  baselines.py        LSTM, GRU, CNN, TCN, EEGNet, SNN, FedAvg, FedProx, pFedMe, FedTL-EEG, LFNL
  baselines_da.py     federated DANN and FedAvg + per-client fine-tuning (sEMG domain adaptation)
  data_*.py           dataset loaders (BCI-IV-2a/2b, Ninapro DB5, DVS128, N-Caltech101)
  run_*.py            one script per experiment in the paper
  demo_synthetic.py   self-contained demo, no external data, < 1 s on CPU
results/*.json        recorded outputs from which every figure and table was generated
generate_figures.py   regenerates all manuscript figures from results/
```

## Quick start (no dataset download required)

```bash
pip install -r experiments/requirements.txt
python experiments/demo_synthetic.py
```

The demo runs end-to-end on CPU in under a second and empirically confirms the one-shot
exact-recovery property.

To regenerate all figures from the recorded results:

```bash
python generate_figures.py
```

## Reproducing the full experiments

The benchmark datasets are **not redistributed here**; they remain with their original providers
and are subject to those providers' licenses and (free) registration where applicable. Download
them and place them under `data/`:

| Dataset | Source |
|---|---|
| BCI Competition IV-2a / IV-2b (EEG) | https://bnci-horizon-2020.eu/database/data-sets |
| Ninapro DB5 (sEMG) | https://ninapro.hevs.ch |
| DVS128 Gesture (events) | https://research.ibm.com/interactive/dvsgesture/ |
| N-Caltech101 (events) | https://www.garrickorchard.com/datasets/n-caltech101 |

Then run any experiment, or all of them:

```bash
python experiments/run_bci_iv2a.py     # BCI-IV-2a + personalization sweep
python experiments/run_ninapro.py      # Ninapro DB5
python experiments/run_dvs128.py       # DVS128 Gesture
python experiments/run_ablation.py     # cross-cutting ablations
python experiments/run_all.py          # everything, sequentially
```

Federated partitions and preprocessing are regenerated deterministically from the source files
using fixed random seeds, so results are reproducible without shipping intermediate arrays.

See `experiments/INSTALL.md` for detailed setup and `experiments/DEMO.md` for the demo walkthrough.

## Reference environment

Results reported in the paper were produced with **Python 3.12.13**, NumPy 1.26.4, SciPy 1.17.1,
PyTorch 2.11.0 (CUDA 13.0), scikit-learn 1.8.0, and MNE 1.12.1. The code was additionally tested
with Python 3.10. Supported dependency ranges are in `experiments/requirements.txt`. Everything
runs on a standard CPU; a GPU only accelerates the deep-learning baselines.

Core hyperparameters are set in `experiments/config.py` and match the paper's Methods: FRI-ESN uses
`N_r=500` (500/1,000/2,000 for the event-camera scaling study), spectral radius 0.95, leaking rate
0.3, washout 100, ridge `lambda=0.05`; FRI-LSM uses 500 LIF neurons, `tau_m=20 ms`, `V_th=1.0`,
`tau_ref=2 ms`, 10% connectivity, `lambda=1e-4`; personalization uses `mu=0.01` and the proximal
readout variant uses coupling `eta=0.01`.

## Citation

If you use this code, please cite the Article and the archived software release:

```bibtex
@article{dong2026fri,
  author  = {Dong, Liang},
  title   = {Federated reservoir intelligence for distributed temporal and neuromorphic learning},
  journal = {Communications Engineering},
  year    = {2026}
}
```

## License

MIT License — see [LICENSE](LICENSE). Copyright (c) 2026 Liang Dong.
