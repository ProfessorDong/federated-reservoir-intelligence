# Installation guide

## Operating system

- **Tested on:** Ubuntu 24.04 LTS (Linux 6.11), kernel `6.11.0-29-generic`
- **Should work on:** any modern Linux distribution; macOS (Apple Silicon and Intel); Windows 10/11 with WSL or native Python
- **Not tested on:** ARM Linux without CUDA, BSD variants

## Programming language

Python 3.10 or newer. Tested with Python 3.10 and 3.12.

## Hardware requirements

- **Minimum:** 8 GB RAM, any x86-64 CPU. CPU-only execution is fully supported; the code automatically falls back to CPU when no GPU is available.
- **Recommended:** 32 GB+ RAM (DVS128 event loading uses ~7.4 GB), an NVIDIA GPU with CUDA 11.8+ (any RTX 30/40/Pro series; tested on RTX 4090 and RTX PRO 5000). A GPU is not required for ESN/FRI-Stats experiments but accelerates LSTM/SNN baseline training by roughly 5–20×.
- **Disk:** ~15 GB for full dataset cache (DVS128 alone is ~12 GB).

## Software dependencies

| Package | Version range | Purpose |
|---|---|---|
| numpy | ≥1.23, <2.0 | Numerical arrays |
| scipy | ≥1.10, <2.0 | Linear algebra, stats |
| torch | ≥2.0, <3.0 | LSTM/EEGNet/SNN training and tensor ops on GPU |
| scikit-learn | ≥1.2, <2.0 | Accuracy/F1 metrics, stratified splits |
| mne | ≥1.4, <2.0 | EEG data loading (BCI-IV-2a/2b GDF files) |

All dependencies are pure-Python or have prebuilt wheels for x86-64. No system-level native dependencies beyond a working Python interpreter and (optionally) CUDA runtime for GPU.

## Installation steps

### Option A: pip with virtualenv (recommended for reviewers)

```bash
# Clone or unzip the code to a working directory
cd experiments/

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

### Option B: conda

```bash
conda create -n fri python=3.10
conda activate fri
pip install -r requirements.txt
```

### Option C: existing PyTorch environment

If you already have a Python environment with PyTorch installed:

```bash
pip install numpy scipy scikit-learn mne
```

## Typical install time

- **Fresh `pip install -r requirements.txt`:** 5–10 minutes on a typical broadband connection (PyTorch wheel is the largest at ~800 MB).
- **Conda environment creation + pip install:** 8–15 minutes.
- **No compilation step is required** — all dependencies have prebuilt wheels for the supported platforms.

## Verifying the installation

Run the synthetic demo (does not require any external data):

```bash
python demo_synthetic.py
```

Expected output: a short log showing ESN reservoir state shape, ridge readout fit, and centralized vs. statistics-based aggregation accuracy on synthetic data. Total runtime: ~30 seconds on CPU.

## Optional GPU configuration

For GPU acceleration:

```bash
# Verify CUDA is available
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

Should print `True` and a CUDA version. If you see `False`, the code will run on CPU automatically.

For Blackwell-generation GPUs (RTX PRO 5000 and newer), cuDNN may segfault on certain ops; in that case set `torch.backends.cudnn.enabled = False` (already handled in the run scripts when needed).

## Troubleshooting

- **`mne` install fails:** ensure `pip` is recent (`pip install --upgrade pip`); MNE has occasional build-isolation issues on older pip.
- **Out of memory on DVS128:** reduce `max_events_per_sample` in `config.py:DVS128_CFG` from 50,000 to 20,000.
- **Slow ESN runs:** verify NumPy is built with BLAS/LAPACK (`python -c "import numpy; numpy.show_config()"`); ESN spends most time in matrix multiplications.
