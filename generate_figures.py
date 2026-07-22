#!/usr/bin/env python3
"""
Generate all publication-quality figures for the FRI NMI paper.
Nature Machine Intelligence style: Helvetica, clean, minimal, high-contrast.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ArrowStyle
from matplotlib.lines import Line2D
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# Nature MI style configuration
# ═══════════════════════════════════════════════════════════════════════════════
NATURE_W1 = 89 / 25.4   # single column (inches)
NATURE_W2 = 183 / 25.4  # double column (inches)

# Nature Portfolio palette (NEJM-style — as used by ggsci::scale_color_nejm,
# widely adopted across Nature, Nature Communications, Nature Machine Intelligence
# for multi-method comparisons).
C = {
    'blue':    '#0072B5',  # NEJM blue       — Local/Centralized ESN, Session 1
    'orange':  '#E18727',  # NEJM amber      — FRI variants, Session 2
    'red':     '#BC3C29',  # NEJM brick      — FRI-ESN 1K highlight, optimal marker
    'green':   '#20854E',  # NEJM emerald    — LFNL, FRI-Stats global, FedAvg-EEGNet
    'purple':  '#7876B1',  # NEJM violet     — FedTL-EEG, FedProx-SNN, proximal
    'slate':   '#6F99AD',  # NEJM slate      — auxiliary baselines
    'yellow':  '#FFDC91',  # NEJM yellow     — optional accent
    'pink':    '#EE4C97',  # NEJM pink       — optional accent
    'gray':    '#7F7F7F',  # neutral gray    — LSTM baselines, reference lines
    'lgray':   '#B0B0B0',  # lighter gray    — secondary baselines
    'dblue':   '#005389',  # darker NEJM blue
    'lorange': '#F4C27A',  # lighter NEJM amber — sketched FRI-LSM
    'lgreen':  '#7FBE98',  # lighter NEJM emerald
    'lpurple': '#B2B0D0',  # lighter NEJM violet
    'panelbg': '#F7F7F7',
}

def setup_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 7,
        'axes.labelsize': 8,
        'axes.titlesize': 9,
        'axes.titleweight': 'bold',
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 6.5,
        'legend.frameon': False,
        'axes.linewidth': 0.6,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.minor.size': 1.5,
        'ytick.minor.size': 1.5,
        'axes.spines.right': False,
        'axes.spines.top': False,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'figure.facecolor': 'white',
    })

setup_style()

# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════
RESULTS = Path(__file__).parent / 'results'

def load_json(name):
    with open(RESULTS / name) as f:
        return json.load(f)

def mean_ci(vals, conf=0.95):
    """Proper small-sample 95% confidence interval: Student's t with the
    sample standard deviation (ddof=1). For n seeds the half-width is
    t_{(1+conf)/2, n-1} * s / sqrt(n)."""
    from scipy import stats as _st
    vals = np.asarray(vals, dtype=float)
    n = len(vals)
    m = vals.mean()
    if n < 2:
        return m, 0.0
    se = vals.std(ddof=1) / np.sqrt(n)
    return m, float(se * _st.t.ppf(0.5 + conf / 2.0, n - 1))

def get_method_stats(data, method, seeds=range(5), keys=('acc','comm_mb')):
    """Extract mean and CI for a method across seeds."""
    out = {}
    for k in keys:
        vals = []
        for s in seeds:
            v = data[str(s)].get(method, {})
            if isinstance(v, dict) and k in v:
                vals.append(v[k])
        if vals:
            out[k], out[k+'_ci'] = mean_ci(vals)
        else:
            out[k], out[k+'_ci'] = 0, 0
    return out

def panel_label(ax, label, x=-0.08, y=1.06):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='top', ha='left')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1: Framework architecture
# ═══════════════════════════════════════════════════════════════════════════════
def fig1_framework():
    fig = plt.figure(figsize=(NATURE_W2, 4.2))

    # Three panels: a (top full), b (bottom-left), c (bottom-right)
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1.6, 1],
                           hspace=0.35, wspace=0.25,
                           left=0.02, right=0.98, top=0.97, bottom=0.03)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    for ax in [ax_a, ax_b, ax_c]:
        ax.set_xlim(0, 10)
        ax.set_aspect('equal')
        ax.axis('off')
    ax_a.set_ylim(-0.3, 6)
    ax_b.set_ylim(0, 6)
    ax_c.set_ylim(0, 6)

    # ── Panel a: Main architecture ────────────────────────────────────────
    panel_label(ax_a, 'a', x=-0.02, y=1.02)

    # Server box
    srv = FancyBboxPatch((3.5, 4.6), 3.0, 1.0, boxstyle="round,pad=0.15",
                          facecolor=C['lgreen'], edgecolor=C['green'], lw=1.5)
    ax_a.add_patch(srv)
    ax_a.text(5.0, 5.1, 'Server', ha='center', va='center',
              fontsize=9, fontweight='bold', color=C['green'])
    ax_a.text(5.0, 4.75, 'Aggregates readouts or\nsufficient statistics',
              ha='center', va='center', fontsize=6, color='#333333')

    # Three clients
    client_info = [
        ('Client 1 (EEG)', 0.3, C['blue'], 'ESN'),
        ('Client 2 (EMG)', 3.5, C['blue'], 'ESN'),
        ('Client 3 (Events)', 6.7, C['purple'], 'LSM'),
    ]

    # Layout constants for panel a
    # Data box:      y = 2.9 to 3.5 (height 0.6)
    # Reservoir box:  y = 1.9 to 2.7 (height 0.8)
    # Readout box:    y = 0.9 to 1.6 (height 0.7)  — RAISED from 0.7
    # Client container: y = 0.55 to 4.0
    # Server:         y = 4.6 to 5.6
    # Message box:    y = -0.1 to 0.4 (wider, lower)

    for name, x0, color, rtype in client_info:
        # Client container
        cl = FancyBboxPatch((x0, 0.55), 2.8, 3.45, boxstyle="round,pad=0.1",
                             facecolor='#FAFAFA', edgecolor='#CCCCCC', lw=0.8)
        ax_a.add_patch(cl)
        ax_a.text(x0+1.4, 3.75, name, ha='center', va='center',
                  fontsize=7, fontweight='bold', color='#333333')

        # Data box: y = 2.9 to 3.5
        data_box = FancyBboxPatch((x0+0.3, 2.9), 2.2, 0.6, boxstyle="round,pad=0.08",
                                   facecolor='white', edgecolor='#AAAAAA', lw=0.5)
        ax_a.add_patch(data_box)
        ax_a.text(x0+1.4, 3.2, 'Local data $\\mathcal{D}_k$',
                  ha='center', va='center', fontsize=6, color='#555555')

        # Reservoir box: y = 1.9 to 2.7
        res_box = FancyBboxPatch((x0+0.3, 1.9), 2.2, 0.8, boxstyle="round,pad=0.08",
                                  facecolor=color+'20', edgecolor=color, lw=1.0)
        ax_a.add_patch(res_box)
        ax_a.text(x0+1.4, 2.3, f'{rtype} Reservoir', ha='center', va='center',
                  fontsize=6.5, fontweight='bold', color=color)
        # FIXED badge
        badge = FancyBboxPatch((x0+1.95, 2.55), 0.5, 0.25, boxstyle="round,pad=0.05",
                                facecolor=C['red'], edgecolor='none')
        ax_a.add_patch(badge)
        ax_a.text(x0+2.2, 2.675, 'FIXED', ha='center', va='center',
                  fontsize=4.5, fontweight='bold', color='white')

        # Readout box: y = 0.9 to 1.6 (RAISED)
        ro_box = FancyBboxPatch((x0+0.3, 0.9), 2.2, 0.7, boxstyle="round,pad=0.08",
                                 facecolor=C['lorange'], edgecolor=C['orange'], lw=1.0)
        ax_a.add_patch(ro_box)
        ax_a.text(x0+1.4, 1.25, '$\\mathbf{W}_{\\mathrm{out}}$ Readout',
                  ha='center', va='center', fontsize=6.5, fontweight='bold',
                  color=C['orange'])

        # Arrows: data_box bottom (2.9) -> reservoir top (2.7)
        ax_a.annotate('', xy=(x0+1.4, 2.7), xytext=(x0+1.4, 2.9),
                      arrowprops=dict(arrowstyle='->', color='#666666', lw=0.8))
        # Arrows: reservoir bottom (1.9) -> readout top (1.6)
        ax_a.annotate('', xy=(x0+1.4, 1.6), xytext=(x0+1.4, 1.9),
                      arrowprops=dict(arrowstyle='->', color='#666666', lw=0.8))

        # Communication arrows: client container top (4.0) -> server bottom (4.6)
        ax_a.annotate('', xy=(5.0, 4.6), xytext=(x0+1.4, 4.0),
                      arrowprops=dict(arrowstyle='->', color=C['orange'],
                                      lw=1.2, connectionstyle='arc3,rad=0.0'))

    # Communication label box (raised to sit between clients and server)
    ax_a.text(2.0, 4.35, '$\\mathbf{W}_{\\mathrm{out}}^{(k)}$ or\n$(\\mathbf{G}_k, \\mathbf{H}_k)$',
              ha='center', va='center', fontsize=5.5, color=C['orange'],
              fontweight='bold',
              bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                        edgecolor=C['orange'], alpha=0.9, lw=0.5))

    # Central message box (WIDER and LOWER)
    msg = FancyBboxPatch((1.5, -0.15), 7.0, 0.55, boxstyle="round,pad=0.1",
                          facecolor='#FFF3E0', edgecolor=C['orange'], lw=0.8)
    ax_a.add_patch(msg)
    ax_a.text(5.0, 0.125, 'Reservoir weights never communicated  '
              '$\\cdot$  Communication cost: $\\mathcal{O}(N_r d_y)$ per round',
              ha='center', va='center', fontsize=6, color=C['orange'],
              fontstyle='italic')

    # ── Panel b: Readout-based aggregation ────────────────────────────────
    panel_label(ax_b, 'b', x=-0.02, y=1.02)
    ax_b.text(5.0, 5.7, 'Readout-based aggregation', ha='center',
              fontsize=8, fontweight='bold', color=C['orange'])

    for r, xoff in enumerate([0.5, 3.5, 6.5]):
        # Round box
        rbox = FancyBboxPatch((xoff, 0.5), 2.5, 4.5, boxstyle="round,pad=0.1",
                               facecolor='white', edgecolor='#DDDDDD', lw=0.5)
        ax_b.add_patch(rbox)
        ax_b.text(xoff+1.25, 4.7, f'Round {r+1}' if r < 2 else 'Round $R$',
                  ha='center', fontsize=6, fontweight='bold', color='#555555')

        # Server mini-box
        s = FancyBboxPatch((xoff+0.4, 3.5), 1.7, 0.8, boxstyle="round,pad=0.05",
                            facecolor=C['lgreen'], edgecolor=C['green'], lw=0.6)
        ax_b.add_patch(s)
        ax_b.text(xoff+1.25, 3.9, 'Server', ha='center', fontsize=5.5, color=C['green'])

        # Client mini-boxes: cy to cy+0.8
        for ci, cy in enumerate([1.0, 2.2]):
            cb = FancyBboxPatch((xoff+0.4, cy), 1.7, 0.8, boxstyle="round,pad=0.05",
                                 facecolor=C['lorange'], edgecolor=C['orange'], lw=0.6)
            ax_b.add_patch(cb)
            ax_b.text(xoff+1.25, cy+0.4, f'$\\mathbf{{W}}_{{out}}^{{({ci+1})}}$',
                      ha='center', fontsize=5.5, color=C['orange'])

            # Arrows: client top (cy+0.8) -> server bottom (3.5)
            ax_b.annotate('', xy=(xoff+1.25, 3.5), xytext=(xoff+1.25, cy+0.8),
                          arrowprops=dict(arrowstyle='->', color=C['orange'], lw=0.6))

    # Ellipsis between round 2 and R
    ax_b.text(6.2, 2.5, '...', fontsize=14, ha='center', va='center', color='#999999')

    ax_b.text(5.0, 0.15, '$R$ rounds, $\\mathcal{O}(N_r d_y)$ scalars/round',
              ha='center', fontsize=6, color=C['orange'], fontstyle='italic')

    # ── Panel c: Statistics-based aggregation ─────────────────────────────
    panel_label(ax_c, 'c', x=-0.02, y=1.02)
    ax_c.text(5.0, 5.7, 'Statistics-based aggregation', ha='center',
              fontsize=8, fontweight='bold', color=C['green'])

    # ONE SHOT badge
    badge = FancyBboxPatch((3.7, 4.8), 2.6, 0.6, boxstyle="round,pad=0.1",
                            facecolor=C['green'], edgecolor='none')
    ax_c.add_patch(badge)
    ax_c.text(5.0, 5.1, 'ONE SHOT', ha='center', va='center',
              fontsize=8, fontweight='bold', color='white')

    # Server
    s = FancyBboxPatch((2.5, 3.2), 5.0, 1.2, boxstyle="round,pad=0.1",
                        facecolor=C['lgreen'], edgecolor=C['green'], lw=1.0)
    ax_c.add_patch(s)
    ax_c.text(5.0, 3.95, 'Server computes:', ha='center', fontsize=6, color=C['green'])
    ax_c.text(5.0, 3.5, '$\\mathbf{W}^\\star = \\mathbf{H}(\\mathbf{G} + \\lambda T\\mathbf{I})^{-1}$',
              ha='center', fontsize=7, color='#333333')

    # Clients: y = 0.8 to 2.3
    for i, x0 in enumerate([0.5, 2.5, 4.5, 6.5]):
        cb = FancyBboxPatch((x0, 0.8), 2.0, 1.5, boxstyle="round,pad=0.08",
                             facecolor=C['panelbg'], edgecolor='#AAAAAA', lw=0.5)
        ax_c.add_patch(cb)
        ax_c.text(x0+1.0, 1.85, f'Client {i+1}', ha='center', fontsize=5.5,
                  fontweight='bold', color='#555555')
        ax_c.text(x0+1.0, 1.25, '$(\\mathbf{G}_k, \\mathbf{H}_k, T_k)$',
                  ha='center', fontsize=5.5, color=C['blue'])
        # Arrow: client top (2.3) -> server bottom (3.2)
        ax_c.annotate('', xy=(x0+1.0, 3.2), xytext=(x0+1.0, 2.3),
                      arrowprops=dict(arrowstyle='->', color=C['blue'], lw=0.8))

    if len(client_info) > 3:
        ax_c.text(8.8, 1.5, '...', fontsize=12, ha='center', color='#999999')

    ax_c.text(5.0, 0.4, 'Single round, exact centralized solution',
              ha='center', fontsize=6, color=C['green'], fontstyle='italic')

    fig.savefig('fig1_framework.pdf', format='pdf')
    fig.savefig('fig1_framework.png', format='png', dpi=300)
    plt.close(fig)
    print('Figure 1 saved.')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2: Main results (Pareto + per-subject)
# ═══════════════════════════════════════════════════════════════════════════════
def fig2_results():
    bci = load_json('bci_iv2a_results.json')
    dvs = load_json('dvs128_results.json')
    extra = load_json('extra_baselines_results.json')

    fig = plt.figure(figsize=(NATURE_W2, 4.2))
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 1.1],
                           wspace=0.38, left=0.06, right=0.97, top=0.90, bottom=0.52)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # ── Panel a: BCI-IV-2a Pareto ─────────────────────────────────────────
    panel_label(ax1, 'a', x=-0.18, y=1.22)
    ax1.set_title('BCI-IV-2a (EEG, 4-class)', fontsize=8)

    bci_methods = [
        ('local_only',     'Local-only ESN',  C['blue'],   'o',  7, 0.1),
        ('centralized',    'Centralized ESN',  C['blue'],   'D',  6, None),
        ('fri_readout',    'FRI-Readout',      C['orange'], '^',  7, None),
        ('fri_stats',      'FRI-Stats',        C['orange'], 's',  7, None),
        ('fri_stats_pers', 'FRI-Stats+Pers',   C['orange'], '*', 10, None),
        ('fedavg_lstm',    'FedAvg-LSTM',      C['gray'],   'x',  6, None),
        ('fedprox_lstm',   'FedProx-LSTM',     C['lgray'],  '+',  6, None),
        ('fedavg_eegnet',  'FedAvg-EEGNet',    C['green'],  'p',  6, None),
        ('per_fedavg',     'pFedMe',           C['lgray'],  'd',  5, None),
        ('fedtl_eeg',      'FedTL-EEG',        C['purple'], 'h',  7, None),
    ]

    for m, label, color, marker, ms, comm_override in bci_methods:
        st = get_method_stats(bci, m)
        x = comm_override if comm_override else st['comm_mb']
        if x == 0: x = 0.1
        ax1.errorbar(x, st['acc'], yerr=st['acc_ci'], fmt=marker,
                     color=color, markersize=ms, markeredgewidth=0.8,
                     capsize=2, capthick=0.6, elinewidth=0.6, label=label,
                     zorder=5 if 'fri' in m.lower() or 'local' in m.lower() else 3)
        sv = [bci[str(s)][m]['acc'] for s in range(5)
              if m in bci[str(s)] and 'acc' in bci[str(s)][m]]
        ax1.scatter([x] * len(sv), sv, s=3, color=color, alpha=0.30,
                    zorder=2, linewidths=0)

    # Additional deep-learning baselines (revision; from extra_baselines_results.json)
    bci_extra = [
        ('local_only_lstm', 'Local-only LSTM', '#7F7F7F', '<', 6),
        ('fedavg_gru',      'FedAvg-GRU',      '#6F99AD', 'v', 6),
        ('fedavg_cnn',      'FedAvg-CNN',      '#EE4C97', 'P', 7),
        ('fedavg_tcn',      'FedAvg-TCN',      '#8C564D', 'X', 6),
    ]
    for m, label, color, marker, ms in bci_extra:
        st = get_method_stats(extra['bci'], m)
        x = st['comm_mb'] if st['comm_mb'] > 0 else 0.1
        ax1.errorbar(x, st['acc'], yerr=st['acc_ci'], fmt=marker, color=color,
                     markersize=ms, markeredgewidth=0.8, capsize=2, capthick=0.6,
                     elinewidth=0.6, label=label, zorder=4)
        sv = [extra['bci'][str(s)][m]['acc'] for s in range(5)
              if m in extra['bci'][str(s)] and 'acc' in extra['bci'][str(s)][m]]
        ax1.scatter([x] * len(sv), sv, s=3, color=color, alpha=0.30,
                    zorder=2, linewidths=0)

    ax1.set_xscale('log')
    ax1.set_xlabel('Communication (MB)')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlim(0.05, 200)
    ax1.set_ylim(0.20, 0.65)
    ax1.axhline(0.25, color='#CCCCCC', lw=0.5, ls='--', zorder=1)
    ax1.text(0.6, 0.225, 'chance', fontsize=7, color='#AAAAAA')

    # Annotation: communication reduction
    ax1.annotate('$5\\times$ less\ncomm.', xy=(17.3, 0.485), xytext=(55, 0.58),
                 fontsize=7, ha='center', color=C['orange'],
                 arrowprops=dict(arrowstyle='->', color=C['orange'], lw=0.9))

    # ── Panel b: DVS128 Pareto ────────────────────────────────────────────
    panel_label(ax2, 'b', x=-0.16, y=1.22)
    ax2.set_title('DVS128 Gesture (Events, 11-class)', fontsize=8)

    dvs_methods = [
        ('fri_lsm_traces',      'FRI-LSM Traces',   C['orange'], 's',  6, None),
        ('fri_lsm_sketched_m100','FRI-LSM Sketch',   C['lorange'],'^',  6, None),
        ('fedavg_lstm',          'FedAvg-LSTM',      C['gray'],   'x',  6, None),
        ('fedavg_snn',           'FedAvg-SNN',       C['lgray'],  '+',  7, None),
        ('fedprox_snn',          'FedProx-SNN',      C['purple'], 'h',  7, None),
        ('lfnl',                 'LFNL',             C['green'],  'p',  7, None),
    ]

    for m, label, color, marker, ms, comm_override in dvs_methods:
        st = get_method_stats(dvs, m)
        x = comm_override if comm_override else st['comm_mb']
        if x == 0: x = 0.05
        ax2.errorbar(x, st['acc'], yerr=st['acc_ci'], fmt=marker,
                     color=color, markersize=ms, markeredgewidth=0.8,
                     capsize=2, capthick=0.6, elinewidth=0.6, label=label,
                     zorder=3)
        sv = [dvs[str(s)][m]['acc'] for s in range(5)
              if m in dvs[str(s)] and 'acc' in dvs[str(s)][m]]
        ax2.scatter([x] * len(sv), sv, s=3, color=color, alpha=0.30,
                    zorder=2, linewidths=0)

    # Add FRI-ESN data points (from fri_esn_events_results.json)
    esn_data = load_json('fri_esn_events_results.json')
    for nr, label, ms in [(500, 'FRI-ESN 500', 8), (1000, 'FRI-ESN 1K', 10)]:
        key = f'esn{nr}_mean_logvar'
        accs = [esn_data[str(s)][key]['acc'] for s in range(5)]
        comm = esn_data['0'][key]['comm_mb']
        m_acc, ci_acc = mean_ci(accs)
        ax2.errorbar(comm, m_acc, yerr=ci_acc, fmt='*',
                     color=C['red'] if nr == 1000 else C['orange'],
                     markersize=ms, markeredgewidth=0.8,
                     capsize=2, capthick=0.6, elinewidth=0.6, label=label,
                     zorder=10)
        ax2.scatter([comm] * len(accs), accs, s=3,
                    color=C['red'] if nr == 1000 else C['orange'],
                    alpha=0.30, zorder=2, linewidths=0)

    # Additional deep-learning baselines (revision); local-only LSTM (0 comm)
    # is placed at x=0.1, matching the zero-communication convention of panel a.
    dvs_extra = [
        ('local_only_lstm', 'Local-only LSTM', '#7F7F7F', '<', 6),
        ('fedavg_gru', 'FedAvg-GRU', '#6F99AD', 'v', 6),
        ('fedavg_cnn', 'FedAvg-CNN', '#EE4C97', 'P', 7),
        ('fedavg_tcn', 'FedAvg-TCN', '#8C564D', 'X', 6),
    ]
    for m, label, color, marker, ms in dvs_extra:
        st = get_method_stats(extra['dvs'], m)
        x = st['comm_mb'] if st['comm_mb'] > 0 else 0.1
        ax2.errorbar(x, st['acc'], yerr=st['acc_ci'], fmt=marker, color=color,
                     markersize=ms, markeredgewidth=0.8, capsize=2, capthick=0.6,
                     elinewidth=0.6, label=label, zorder=4)
        sv = [extra['dvs'][str(s)][m]['acc'] for s in range(5)
              if m in extra['dvs'][str(s)] and 'acc' in extra['dvs'][str(s)][m]]
        ax2.scatter([x] * len(sv), sv, s=3, color=color, alpha=0.30,
                    zorder=2, linewidths=0)

    ax2.set_xscale('log')
    ax2.set_xlabel('Communication (MB)')
    ax2.set_xlim(0.05, 5000)
    ax2.set_ylim(0.20, 0.90)

    # Annotation: FRI-ESN surpasses SNN
    ax2.annotate('Surpasses SNN\n$2\\times$ less comm.\nNo BPTT',
                 xy=(178, 0.79), xytext=(20, 0.50),
                 fontsize=7, ha='center', color=C['red'], fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color=C['red'], lw=0.8))

    # ── Panel c: Per-subject BCI accuracy ─────────────────────────────────
    panel_label(ax3, 'c', x=-0.14, y=1.22)
    ax3.set_title('Per-subject personalization', fontsize=8)

    # Average per-subject accuracy across seeds
    local_ps = np.array([bci[str(s)]['local_only']['per_subject_acc'] for s in range(5)])
    pers_ps = np.array([bci[str(s)]['fri_stats_pers']['per_subject_acc'] for s in range(5)])
    local_mean = local_ps.mean(axis=0)
    pers_mean = pers_ps.mean(axis=0)

    # Also get global FRI-Stats (single W for all subjects)
    stats_acc_vals = [bci[str(s)]['fri_stats']['acc'] for s in range(5)]
    stats_global = np.mean(stats_acc_vals)

    subjects = np.arange(1, 10)
    w = 0.3
    ax3.bar(subjects - w/2, local_mean, w, color=C['blue'], alpha=0.85,
            label='Local-only', edgecolor='white', lw=0.3)
    ax3.bar(subjects + w/2, pers_mean, w, color=C['orange'], alpha=0.85,
            label='FRI-Stats+Pers', edgecolor='white', lw=0.3)
    # Individual seed data points (editorial requirement for single-value bars)
    for si in range(local_ps.shape[0]):
        ax3.scatter(subjects - w/2, local_ps[si], s=2.5, color='#08306b',
                    alpha=0.55, zorder=6, linewidths=0)
        ax3.scatter(subjects + w/2, pers_ps[si], s=2.5, color='#7f3b08',
                    alpha=0.55, zorder=6, linewidths=0)
    ax3.axhline(stats_global, color=C['green'], lw=0.8, ls='--',
                label=f'FRI-Stats global ({stats_global:.2f})')
    ax3.axhline(0.25, color='#CCCCCC', lw=0.5, ls=':', zorder=1)

    ax3.set_xlabel('Subject')
    ax3.set_ylabel('Accuracy')
    ax3.set_xticks(subjects)
    ax3.set_ylim(0, 0.85)
    ax3.legend(loc='upper left', bbox_to_anchor=(0.0, -0.24), fontsize=7,
               ncol=1, handletextpad=0.3, borderpad=0.3, frameon=False)

    # Larger in-plot text (tick labels and axis labels) for all three panels
    for _ax in (ax1, ax2, ax3):
        _ax.tick_params(labelsize=8)
        _ax.xaxis.label.set_size(9)
        _ax.yaxis.label.set_size(9)

    # Legends for panels a and b: placed below the plot area (outside the axes)
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.24), fontsize=7,
               ncol=2, handletextpad=0.3, columnspacing=0.6, borderpad=0.3,
               frameon=False)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.24), fontsize=7,
               ncol=2, handletextpad=0.3, columnspacing=0.6, borderpad=0.3,
               frameon=False)

    fig.savefig('fig2_results.pdf', format='pdf')
    fig.savefig('fig2_results.png', format='png', dpi=300)
    plt.close(fig)
    print('Figure 2 saved.')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3: Ablation and theory verification
# ═══════════════════════════════════════════════════════════════════════════════
def fig3_ablation():
    ab = load_json('ablation_results.json')
    bci = load_json('bci_iv2a_results.json')

    fig, axes = plt.subplots(2, 2, figsize=(NATURE_W2, 3.6))
    plt.subplots_adjust(hspace=0.55, wspace=0.35, left=0.08, right=0.96,
                        top=0.92, bottom=0.10)
    ax_a, ax_b, ax_c, ax_d = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    seeds3 = range(3)

    # ── Panel a: Convergence curves ───────────────────────────────────────
    ax_a.set_title('a', loc='left', fontweight='bold', fontsize=11)

    # Average across 3 seeds
    readout_curves = np.array([ab[str(s)]['convergence']['readout_per_round'] for s in seeds3])
    prox_curves = np.array([ab[str(s)]['convergence']['proximal_per_round'] for s in seeds3])
    stats_vals = [ab[str(s)]['convergence']['stats_acc'] for s in seeds3]
    cent_vals = [ab[str(s)]['convergence']['centralized_acc'] for s in seeds3]

    rounds = np.arange(len(readout_curves[0]))

    # Smooth slightly for visual clarity
    from scipy.ndimage import uniform_filter1d
    r_mean = uniform_filter1d(readout_curves.mean(axis=0), size=3)
    p_mean = uniform_filter1d(prox_curves.mean(axis=0), size=3)
    r_std = readout_curves.std(axis=0, ddof=1)
    p_std = prox_curves.std(axis=0, ddof=1)

    ax_a.plot(rounds, r_mean, color=C['orange'], lw=1.2, label='Readout (vanilla)')
    ax_a.fill_between(rounds, r_mean-r_std, r_mean+r_std, color=C['orange'], alpha=0.15)
    ax_a.plot(rounds, p_mean, color=C['purple'], lw=1.2, ls='--', label='Readout (proximal)')
    ax_a.fill_between(rounds, p_mean-p_std, p_mean+p_std, color=C['purple'], alpha=0.15)
    # Centralized drawn first, then Stats (green, dotted) on top so the green line
    # remains visible: the two essentially coincide, illustrating one-shot exact recovery.
    ax_a.axhline(np.mean(cent_vals), color=C['blue'], lw=1.6, ls='-.',
                 label='Centralized')
    ax_a.axhline(np.mean(stats_vals), color=C['green'], lw=1.2, ls=':',
                 label='Stats (1 round)')

    ax_a.set_xlabel('Communication round')
    ax_a.set_ylabel('Accuracy')
    ax_a.set_xlim(0, 50)
    ax_a.set_ylim(0.24, 0.50)
    ax_a.legend(fontsize=6.5, loc='lower right')
    ax_a.set_title('Convergence')

    # ── Panel b: Reservoir dimension ──────────────────────────────────────
    ax_b.set_title('b', loc='left', fontweight='bold', fontsize=11)

    dims = [100, 200, 500, 1000, 2000]
    dim_accs = []
    dim_comms = []
    for nr in dims:
        accs = [ab[str(s)]['reservoir_dim'][str(nr)]['acc'] for s in seeds3]
        comms = [ab[str(s)]['reservoir_dim'][str(nr)]['comm_mb'] for s in seeds3]
        dim_accs.append(mean_ci(accs))
        dim_comms.append(np.mean(comms))

    means = [a[0] for a in dim_accs]
    cis = [a[1] for a in dim_accs]

    ax_b.errorbar(dims, means, yerr=cis, color=C['blue'], marker='o',
                  markersize=5, capsize=3, capthick=0.6, lw=1.0, label='Accuracy')
    # Individual seed data points
    for nr in dims:
        seed_accs = [ab[str(s)]['reservoir_dim'][str(nr)]['acc'] for s in seeds3]
        ax_b.scatter([nr] * len(seed_accs), seed_accs, s=4, color=C['blue'],
                     alpha=0.4, zorder=6, linewidths=0)
    ax_b.set_xlabel('Reservoir dimension $N_r$')
    ax_b.set_ylabel('Accuracy', color=C['blue'])
    ax_b.tick_params(axis='y', labelcolor=C['blue'])
    ax_b.set_xscale('log')
    ax_b.set_xticks(dims)
    ax_b.set_xticklabels([str(d) for d in dims])
    ax_b.set_ylim(0.32, 0.55)
    ax_b.set_title('Reservoir dimension')

    ax_b2 = ax_b.twinx()
    ax_b2.bar([d*1.08 for d in dims], dim_comms, width=[d*0.15 for d in dims],
              color=C['orange'], alpha=0.4, label='Comm. (MB)')
    ax_b2.set_ylabel('Communication (MB)', color=C['orange'])
    ax_b2.tick_params(axis='y', labelcolor=C['orange'])
    ax_b2.spines['right'].set_visible(True)
    ax_b2.spines['right'].set_color(C['orange'])

    # ── Panel c: Drift adaptation ─────────────────────────────────────────
    ax_c.set_title('c', loc='left', fontweight='bold', fontsize=11)

    betas = ['1.0', '0.99', '0.95', '0.9', '0.85']
    beta_labels = ['1.0', '0.99', '0.95', '0.9', '0.85']
    drift = load_json('drift_heldout_results.json')
    s1_means, s2_means, s1_cis, s2_cis = [], [], [], []
    for beta in betas:
        s1 = [drift[str(s)][f'beta_{beta}']['session1_acc'] for s in range(5)]
        s2 = [drift[str(s)][f'beta_{beta}']['session2_acc'] for s in range(5)]
        m1, c1 = mean_ci(s1)
        m2, c2 = mean_ci(s2)
        s1_means.append(m1); s2_means.append(m2)
        s1_cis.append(c1); s2_cis.append(c2)

    # LSTM baseline
    lstm_s2 = [drift[str(s)]['lstm_retrained']['session2_acc'] for s in range(5)]
    lstm_m, lstm_c = mean_ci(lstm_s2)

    x = np.arange(len(betas))
    ax_c.errorbar(x, s1_means, yerr=s1_cis, color=C['blue'], marker='o',
                  markersize=4, capsize=2, lw=1.0, label='Session 1')
    ax_c.errorbar(x, s2_means, yerr=s2_cis, color=C['orange'], marker='s',
                  markersize=4, capsize=2, lw=1.0, label='Session 2')
    # Individual seed data points
    for j, beta in enumerate(betas):
        s1v = [drift[str(s)][f'beta_{beta}']['session1_acc'] for s in range(5)]
        s2v = [drift[str(s)][f'beta_{beta}']['session2_acc'] for s in range(5)]
        ax_c.scatter([x[j]] * 5, s1v, s=3, color=C['blue'], alpha=0.4, zorder=6, linewidths=0)
        ax_c.scatter([x[j]] * 5, s2v, s=3, color=C['orange'], alpha=0.4, zorder=6, linewidths=0)
    ax_c.axhline(lstm_m, color=C['gray'], lw=0.7, ls='--',
                 label=f'LSTM retrained S2')
    ax_c.fill_between([-0.5, len(betas)-0.5], lstm_m-lstm_c, lstm_m+lstm_c,
                      color=C['gray'], alpha=0.1)

    ax_c.set_xticks(x)
    ax_c.set_xticklabels(beta_labels)
    ax_c.set_xlabel('Forgetting factor $\\beta$')
    ax_c.set_ylabel('Accuracy (held-out)')
    ax_c.set_ylim(0.45, 0.72)
    ax_c.legend(fontsize=6.5, loc='upper right')
    ax_c.set_title('Cross-session generalization')

    # ── Panel d: Privacy scaling — parameter MSE (Prop 2) ─────────────────
    ax_d.set_title('d', loc='left', fontweight='bold', fontsize=11)

    ps = load_json('privacy_scaling_results.json')
    seeds_ps = sorted(ps.keys(), key=int)
    fracs = ['frac_0.1', 'frac_0.2', 'frac_0.3', 'frac_0.5', 'frac_0.7', 'frac_1.0']

    T_ks = [np.mean([ps[s][f]['T_k_avg'] for s in seeds_ps]) for f in fracs]
    # Per-seed parameter-MSE curves, each normalized to its own smallest-T_k value
    # (dimensionless; isolates the T_k scaling). Then aggregate across seeds.
    mse_by_seed = np.array([[ps[s][f]['param_mse'] / ps[s][fracs[0]]['param_mse']
                             for f in fracs] for s in seeds_ps])
    mse_means = mse_by_seed.mean(axis=0)
    mse_cis = [mean_ci(mse_by_seed[:, j])[1] for j in range(len(fracs))]

    # O(1/T_k^2) reference (Prop 2), anchored at the smallest T_k (drawn first)
    T_arr = np.array(T_ks, dtype=float)
    ref = (T_arr[0] / T_arr) ** 2
    ax_d.plot(T_arr, ref, color=C['gray'], lw=1.1, ls='--', zorder=2,
              label='$\\mathcal{O}(1/T_k^2)$ (Prop. 2)')

    ax_d.errorbar(T_ks, mse_means, yerr=mse_cis, color=C['orange'], marker='o',
                  markersize=5, capsize=3, capthick=0.6, lw=0, ls='none', zorder=5,
                  label='Parameter MSE (empirical)')
    # Individual seed values
    for j in range(len(fracs)):
        ax_d.scatter([T_ks[j]] * len(seeds_ps), mse_by_seed[:, j], s=4,
                     color=C['orange'], alpha=0.4, zorder=6, linewidths=0)

    slope = np.polyfit(np.log(T_ks), np.log(mse_means), 1)[0]
    ax_d.set_xscale('log')
    ax_d.set_yscale('log')
    ax_d.set_xlabel('Per-client samples $T_k$')
    ax_d.set_ylabel('Parameter MSE (norm.)')
    ax_d.set_xticks([23, 46, 69, 115, 161, 231])
    ax_d.set_xticklabels(['23', '', '69', '115', '', '231'])
    ax_d.legend(fontsize=6.5, loc='upper right')
    ax_d.text(0.05, 0.07, f'fitted slope $=$ {slope:.2f}', transform=ax_d.transAxes,
              fontsize=7, color=C['orange'])
    ax_d.set_title('Privacy scaling ($\\varepsilon{=}1.0$)')

    fig.savefig('fig3_ablation.pdf', format='pdf')
    fig.savefig('fig3_ablation.png', format='png', dpi=300)
    plt.close(fig)
    print('Figure 3 saved.')


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    os.chdir(Path(__file__).parent)
    print('Generating Nature MI figures...')
    # NOTE: Figure 1 is the TikZ figure fig1_framework.tex (compile with pdflatex),
    # not this matplotlib version. The matplotlib fig1_framework() is kept only as a
    # fallback/draft and is intentionally NOT called so it does not overwrite the
    # TikZ-generated fig1_framework.pdf used by the manuscript.
    # fig1_framework()
    fig2_results()
    fig3_ablation()
    print('Figures 2 and 3 generated (Figure 1 = TikZ fig1_framework.tex).')
