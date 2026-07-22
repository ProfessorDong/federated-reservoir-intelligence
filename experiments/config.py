"""
Configuration for all FRI experiments.
"""
import os, torch

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_ROOT = os.path.join(os.path.dirname(__file__), '..', 'data')
BCI_2A_DIR = os.path.join(DATA_ROOT, 'BCICompetition', 'DataSet2', 'extracted')
NINAPRO_DB5_DIR = os.path.join(DATA_ROOT, 'Ninapro', 'DB5', 'extracted')
DVS128_DIR = os.path.join(DATA_ROOT, 'DVS128', 'extracted',
                          'DVS spiking camera  datasets', 'DVS  Gesture dataset', 'DvsGesture')
NCALTECH_DIR = os.path.join(DATA_ROOT, 'N-Caltech101', 'extracted', 'Caltech101')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Device ───────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Random seeds ─────────────────────────────────────────────────────────────
SEEDS = [0, 1, 2, 3, 4]

# ── ESN hyperparameters ─────────────────────────────────────────────────────
ESN_CFG = dict(
    N_r=500,
    spectral_radius=0.95,
    leaking_rate=0.3,
    input_scaling=0.5,
    sparsity=0.9,
    washout=100,
    ridge_lambda=0.05,  # tuned for [mean,log-var] features (1000-dim, ~230 samples)
)

# ── LSM hyperparameters ─────────────────────────────────────────────────────
LSM_CFG = dict(
    N_s=500,
    tau_m=20e-3,       # membrane time constant (seconds)
    V_th=1.0,
    V_rest=0.0,
    V_reset=0.0,
    tau_ref=2e-3,      # refractory period (seconds)
    connectivity=0.1,
    dt=1e-3,           # simulation timestep (seconds)
    ridge_lambda=1e-4,
    trace_decay=0.95,  # kappa for exponential trace
)

# ── Federated protocol ──────────────────────────────────────────────────────
FED_CFG = dict(
    num_rounds=50,
    participation_bci2a=0.5,
    participation_ninapro=0.3,
    participation_dvs=0.3,
    participation_ncal=0.3,
    personalization_mu=0.01,
)

# ── Baseline hyperparameters ────────────────────────────────────────────────
LSTM_CFG = dict(
    hidden_dim=128,
    num_layers=1,
    dropout=0.2,
    local_epochs=5,
    lr=1e-3,
    batch_size=32,
)

EEGNET_CFG = dict(
    F1=8,
    D=2,
    F2=16,
    dropout=0.25,
    local_epochs=5,
    lr=1e-3,
    batch_size=32,
)

SNN_CFG = dict(
    hidden_dim=128,
    num_steps=50,
    beta=0.95,         # LIF decay
    local_epochs=5,
    lr=1e-3,
    batch_size=32,
)

# ── BCI-IV-2a ────────────────────────────────────────────────────────────────
BCI2A_CFG = dict(
    n_subjects=9,
    n_classes=4,
    n_channels=22,
    sfreq=250,
    lowcut=4.0,
    highcut=40.0,
    epoch_tmin=0.5,
    epoch_tmax=2.5,    # 2 seconds * 250 Hz = 500 samples per trial
)

# ── Ninapro DB5 ─────────────────────────────────────────────────────────────
NINAPRO_CFG = dict(
    n_subjects=10,
    n_channels=16,
    sfreq=200,
    window_size=400,   # 2 seconds at 200 Hz
    window_step=100,   # 0.5 second step
)

# ── DVS128 Gesture ───────────────────────────────────────────────────────────
DVS128_CFG = dict(
    n_classes=11,
    sensor_size=(128, 128),
    time_window_us=300_000,  # 300 ms windows for binning
    max_events_per_sample=50_000,
)

# ── N-Caltech101 ────────────────────────────────────────────────────────────
NCALTECH_CFG = dict(
    n_classes_subset=20,
    sensor_size=(240, 180),  # approximate
    n_clients=50,
    dirichlet_alpha=0.5,
)

# ── Ablation ─────────────────────────────────────────────────────────────────
ABLATION_CFG = dict(
    reservoir_dims=[100, 200, 500, 1000, 2000],
    participation_rates=[0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
    forgetting_factors=[1.0, 0.99, 0.95, 0.9, 0.85],
    personalization_mus=[0, 1e-3, 1e-2, 1e-1, 1.0],
    spectral_spreads=[0.0, 0.02, 0.05, 0.1, 0.2],
    dp_epsilons=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    dp_delta=1e-5,
    generalization_fracs=[0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
    privacy_scaling_epsilon=1.0,
    proximal_mu=0.01,
)
ABLATION_SEEDS = [0, 1, 2]
