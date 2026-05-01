#!/usr/bin/env python3
"""
plot_results.py — Performance plots for the follow-ahead system.

Single-run usage:
    python3 plot_results.py ~/follow_data/circle_run01_*.csv
    python3 plot_results.py ~/follow_data/           # all CSVs in dir
    python3 plot_results.py ~/follow_data/ --summary

Multi-run usage (after running with --runs N):
    python3 plot_results.py --multi-run -- ~/follow_data/circle_run*.csv
    python3 plot_results.py --multi-run --paper-compare -- ~/follow_data/*.csv

Multi-run mode:
  - Groups CSVs by test-case prefix (everything before '_run')
  - Plots one representative 4-panel figure per test case (run 01 = nominal)
  - Generates a cross-run summary table (mean ± std across N runs)
  - Optionally overlays paper baseline numbers for honest comparison

Metrics match Leisiazar et al. IEEE RA-L 2025:
    mean_dist_error  — mean |distance - 1.5 m|  (m)
    mean_alpha_rad   — mean α in radians
    mean_reward      — mean combined reward per step
"""

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm

# ── paper constants ────────────────────────────────────────────────────────────
D_TARGET   = 1.5
D_MIN      = 0.5
D_MAX      = 4.0
ALPHA_CONE = 50.0   # degrees
STARTUP_S  = 6.0    # seconds to skip at start

# ── colour palette ─────────────────────────────────────────────────────────────
C_ROBOT = '#E8520A'
C_HUMAN = '#1A7DC4'
C_DIST  = '#2E86AB'
C_ALPHA = '#A23B72'
C_RD    = '#F18F01'
C_RA    = '#C73E1D'
C_REW   = '#3B1F2B'

# ── paper baseline numbers (Leisiazar et al. IEEE RA-L 2025, Tables I–III) ────
# Format: (mean_dist_err, std_dist_err, mean_alpha_rad, std_alpha_rad)
# Mapped to our test cases as best as possible; noted where imperfect.
PAPER_BASELINE = {
    # Our scenario → closest paper scenario
    # Sudden Change = human turns > 45° in one step
    'zigzag':           ('Sudden Change 1',  0.05, 0.09, 0.03, 0.35),
    # Smooth Change = human turns ~10°/step
    'circle':           ('Smooth Change 1',  0.13, 0.06, 0.16, 0.14),
    'gentle_arc':       ('Smooth Change 2',  0.05, 0.06, 0.04, 0.20),
    'gentle_zigzag':    ('Smooth Change 3',  0.22, 0.23, 0.13, 0.11),
    # Sharp Turn = 90° corner
    'square':           ('Sharp Turn (90°)', 0.00, 0.10, 0.64, 0.70),
    # No direct paper equivalent for these cases
    'straight':         None,
    'oscillate':        None,
    'stationary':       None,
    'approach_and_hold': None,
}

# Note printed alongside comparison table
PAPER_NOTE = (
    "Paper baseline: Leisiazar et al., IEEE RA-L 2025 (proposed method, 10 runs).\n"
    "Platform differences: paper used RB1 Base (ROS1); ours uses QBot (ROS2, lower ω_max).\n"
    "Scenario mappings are approximate — see key_contributions.md for details."
)


# ── helpers ───────────────────────────────────────────────────────────────────

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {'elapsed_s', 'robot_x', 'robot_y', 'human_x', 'human_y',
                 'distance', 'alpha_deg', 'alpha_rad', 'dist_error', 'reward'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    return df


def _active(df: pd.DataFrame) -> pd.DataFrame:
    """Drop startup period so metrics only cover active following."""
    mask = df['elapsed_s'] >= STARTUP_S
    return df[mask].copy() if mask.any() else df.copy()


def _compute_metrics(df: pd.DataFrame) -> dict:
    a = _active(df)
    return {
        'n_steps':        len(a),
        'duration_s':     round(df['elapsed_s'].iloc[-1], 1),
        'mean_dist_err':  a['dist_error'].mean(),
        'std_dist_err':   a['dist_error'].std(),
        'mean_alpha_rad': a['alpha_rad'].mean(),
        'std_alpha_rad':  a['alpha_rad'].std(),
        'mean_alpha_deg': a['alpha_deg'].mean(),
        'mean_reward':    a['reward'].mean(),
        'pct_in_cone':    (a['alpha_deg'] < ALPHA_CONE).mean() * 100.0,
        'pct_in_zone':    ((a['distance'] >= D_MIN) & (a['distance'] <= D_MAX)).mean() * 100.0,
    }


def _case_from_path(path: str) -> str:
    """
    Extract test-case name from filename.
    Handles both:
      circle_20240101_120000.csv          → 'circle'
      circle_run01_20240101_120000.csv    → 'circle'
    """
    base  = os.path.splitext(os.path.basename(path))[0]
    parts = base.split('_')
    # Strip trailing run<N> and timestamp tokens
    tokens = []
    for p in parts:
        if p.startswith('run') and p[3:].isdigit():
            break
        # stop at 8-digit date token
        if len(p) == 8 and p.isdigit():
            break
        tokens.append(p)
    # Multi-word case names (approach_and_hold, gentle_arc, gentle_zigzag)
    return '_'.join(tokens) if tokens else parts[0]


def _add_note(ax, text, fontsize=8):
    ax.text(0.02, 0.02, text, transform=ax.transAxes,
            fontsize=fontsize, verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))


def _rainbow_line(ax, x, y, cmap_name):
    """Plot a line coloured by time (red=start → purple=end), matching paper Figs 3–4."""
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 2:
        return
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segs   = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segs, cmap=cmap_name, linewidth=1.8, zorder=2)
    lc.set_array(np.linspace(0, 1, len(x)))
    ax.add_collection(lc)
    ax.autoscale()


# ── single-run 4-panel figure ─────────────────────────────────────────────────

def plot_one(df: pd.DataFrame, case_label: str, out_path: str) -> dict:
    """Generate the 4-panel figure for one run. Returns metrics dict."""
    m = _compute_metrics(df)
    t = df['elapsed_s'].values

    suptitle = (
        f"{case_label}  —  "
        f"dist_err = {m['mean_dist_err']:.3f} ± {m['std_dist_err']:.3f} m  |  "
        f"α = {m['mean_alpha_rad']:.3f} ± {m['std_alpha_rad']:.3f} rad "
        f"({m['mean_alpha_deg']:.1f}°)  |  reward = {m['mean_reward']:.3f}"
    )

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(suptitle, fontsize=11, fontweight='bold', y=0.98)
    gs  = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)
    ax_traj  = fig.add_subplot(gs[0, 0])
    ax_dist  = fig.add_subplot(gs[0, 1])
    ax_alpha = fig.add_subplot(gs[1, 0])
    ax_rew   = fig.add_subplot(gs[1, 1])

    # ── Panel 1: 2-D trajectories (rainbow time colormap, matching paper Figs 3–4)
    ax = ax_traj
    _rainbow_line(ax, df['human_x'], df['human_y'], 'Blues')
    _rainbow_line(ax, df['robot_x'], df['robot_y'],  'Reds')

    # start / end markers
    ax.scatter(*df[['human_x','human_y']].iloc[0],  marker='o', s=80,
               color=C_HUMAN, zorder=5, edgecolors='white', linewidths=0.8)
    ax.scatter(*df[['robot_x','robot_y']].iloc[0],   marker='s', s=80,
               color=C_ROBOT, zorder=5, edgecolors='white', linewidths=0.8)
    ax.scatter(*df[['human_x','human_y']].iloc[-1],  marker='o', s=60,
               color=C_HUMAN, zorder=5, alpha=0.4)
    ax.scatter(*df[['robot_x','robot_y']].iloc[-1],  marker='s', s=60,
               color=C_ROBOT, zorder=5, alpha=0.4)

    # colourbar legend (time: 0 → 1)
    sm_h = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(0, 1))
    sm_r = plt.cm.ScalarMappable(cmap='Reds',  norm=plt.Normalize(0, 1))
    sm_h.set_array([]); sm_r.set_array([])
    h_patch = mpatches.Patch(color=cm.Blues(0.7),  label='Human')
    r_patch = mpatches.Patch(color=cm.Reds(0.7),   label='Robot')
    ax.legend(handles=[h_patch, r_patch], fontsize=8, loc='upper right')

    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.set_title('2-D Trajectories (colour = time)', fontsize=10)
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=0.25)
    _add_note(ax, f'in-zone {m["pct_in_zone"]:.0f}%  |  in-cone {m["pct_in_cone"]:.0f}%')

    # ── Panel 2: distance over time
    ax = ax_dist
    ax.plot(t, df['distance'], color=C_DIST, linewidth=1.2, label='distance', zorder=3)
    ax.axhline(D_TARGET, color='green',  linewidth=1.2, linestyle='--', label=f'target {D_TARGET} m')
    ax.axhline(D_MIN,    color='red',    linewidth=0.9, linestyle=':', alpha=0.7, label=f'min {D_MIN} m')
    ax.axhline(D_MAX,    color='orange', linewidth=0.9, linestyle=':', alpha=0.7, label=f'max {D_MAX} m')
    ax.axvspan(0, STARTUP_S, alpha=0.08, color='grey', label='startup delay')
    ax.set_xlabel('time (s)'); ax.set_ylabel('distance (m)')
    ax.set_title('Robot–Human Distance', fontsize=10)
    ax.legend(fontsize=7.5, loc='upper right')
    ax.grid(True, alpha=0.25); ax.set_ylim(bottom=0)

    # ── Panel 3: alpha angle over time
    ax = ax_alpha
    ax.plot(t, df['alpha_deg'], color=C_ALPHA, linewidth=1.2, label='α (°)')
    ax.axhline(ALPHA_CONE, color='red', linewidth=1.0, linestyle='--',
               label=f'cone limit {ALPHA_CONE}°')
    ax.fill_between(t, df['alpha_deg'], ALPHA_CONE,
                    where=(df['alpha_deg'] > ALPHA_CONE),
                    alpha=0.18, color='red', label='outside cone')
    ax.axvspan(0, STARTUP_S, alpha=0.08, color='grey')
    ax.set_xlabel('time (s)'); ax.set_ylabel('α (degrees)')
    ax.set_title('Alpha Angle (human forward-cone offset)', fontsize=10)
    ax.legend(fontsize=7.5, loc='upper right')
    ax.grid(True, alpha=0.25); ax.set_ylim(0, 185)
    ax2 = ax.twinx()
    ax2.set_ylim(0, np.radians(185)); ax2.set_ylabel('α (rad)', fontsize=8)
    ax2.tick_params(labelsize=7)

    # ── Panel 4: reward components over time
    ax = ax_rew
    ax.plot(t, df['r_d'],     color=C_RD,  linewidth=1.0, alpha=0.75, label='r_d (distance)')
    ax.plot(t, df['r_alpha'], color=C_RA,  linewidth=1.0, alpha=0.75, label='r_α (orientation)')
    ax.plot(t, df['reward'],  color=C_REW, linewidth=1.5, label='total reward')
    ax.axhline(0, color='grey', linewidth=0.6)
    ax.axvspan(0, STARTUP_S, alpha=0.08, color='grey')
    ax.set_xlabel('time (s)'); ax.set_ylabel('reward')
    ax.set_title('Reward Components', fontsize=10)
    ax.legend(fontsize=7.5, loc='lower right')
    ax.grid(True, alpha=0.25)

    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return {'test_case': case_label, **m}


# ── multi-run aggregation ─────────────────────────────────────────────────────

def aggregate_runs(csv_paths: list[str]) -> dict:
    """
    Group CSVs by test case, compute per-run metrics, return cross-run stats.

    Returns
    -------
    dict  {case_name: {'runs': [metrics_dict, ...],
                        'mean': metrics_dict,
                        'std':  metrics_dict,
                        'n':    int}}
    """
    groups = defaultdict(list)
    for p in csv_paths:
        case = _case_from_path(p)
        groups[case].append(p)

    results = {}
    for case, paths in sorted(groups.items()):
        run_metrics = []
        for p in sorted(paths):
            try:
                df = load_csv(p)
                run_metrics.append(_compute_metrics(df))
            except Exception as e:
                print(f'[warn] skipping {p}: {e}')

        if not run_metrics:
            continue

        keys = [k for k in run_metrics[0] if isinstance(run_metrics[0][k], float)]
        mean_m = {k: float(np.mean([r[k] for r in run_metrics])) for k in keys}
        std_m  = {k: float(np.std( [r[k] for r in run_metrics])) for k in keys}

        results[case] = {
            'runs': run_metrics,
            'mean': mean_m,
            'std':  std_m,
            'n':    len(run_metrics),
        }
    return results


# ── comparison table ──────────────────────────────────────────────────────────

def print_comparison_table(aggregated: dict):
    """
    Print a side-by-side table: our results (mean ± std across runs)
    vs. paper baseline where available.
    """
    w = 108
    print('\n' + '═' * w)
    print('  Follow-Ahead Results vs. Paper Baseline  (Leisiazar et al., IEEE RA-L 2025)')
    print('═' * w)
    hdr = (f"{'Scenario':<22} {'N':>3}  "
           f"{'── Ours ──────────────────────────────':^42}  "
           f"{'── Paper (proposed) ──────────────':^34}")
    sub = (f"{'':22} {'':3}  "
           f"{'dist_err (m)':>14} {'alpha (rad)':>14} {'reward':>8} {'in-cone%':>8}  "
           f"{'dist_err (m)':>14} {'alpha (rad)':>14}")
    print(hdr)
    print(sub)
    print('─' * w)

    for case, data in sorted(aggregated.items()):
        m, s, n = data['mean'], data['std'], data['n']

        ours_dist  = f"{m['mean_dist_err']:.3f}±{s['mean_dist_err']:.3f}"
        ours_alpha = f"{m['mean_alpha_rad']:.3f}±{s['mean_alpha_rad']:.3f}"
        ours_rew   = f"{m['mean_reward']:.3f}"
        ours_cone  = f"{m['pct_in_cone']:.1f}%"

        paper = PAPER_BASELINE.get(case)
        if paper:
            pname, pd_m, pd_s, pa_m, pa_s = paper
            paper_dist  = f"{pd_m:.3f}±{pd_s:.3f}"
            paper_alpha = f"{pa_m:.3f}±{pa_s:.3f}"
            paper_label = f'[{pname}]'
        else:
            paper_dist = paper_alpha = '—'
            paper_label = '[no mapping]'

        print(f"  {case:<20} {n:>3}  "
              f"  {ours_dist:>14} {ours_alpha:>14} {ours_rew:>8} {ours_cone:>8}  "
              f"  {paper_dist:>14} {paper_alpha:>14}  {paper_label}")

    print('═' * w)
    print(f'\n  {PAPER_NOTE.replace(chr(10), chr(10) + "  ")}\n')


def save_comparison_figure(aggregated: dict, out_path: str):
    """
    Save a bar-chart comparison figure: our mean ± std vs. paper baseline,
    one panel for distance error and one for alpha, across all test cases.
    """
    cases_with_paper = [(c, d) for c, d in sorted(aggregated.items())
                        if PAPER_BASELINE.get(c) is not None]
    if not cases_with_paper:
        return

    labels        = [c for c, _ in cases_with_paper]
    our_dist_m    = [d['mean']['mean_dist_err']  for _, d in cases_with_paper]
    our_dist_s    = [d['std']['mean_dist_err']   for _, d in cases_with_paper]
    our_alpha_m   = [d['mean']['mean_alpha_rad'] for _, d in cases_with_paper]
    our_alpha_s   = [d['std']['mean_alpha_rad']  for _, d in cases_with_paper]
    paper_dist_m  = [PAPER_BASELINE[c][1] for c in labels]
    paper_dist_s  = [PAPER_BASELINE[c][2] for c in labels]
    paper_alpha_m = [PAPER_BASELINE[c][3] for c in labels]
    paper_alpha_s = [PAPER_BASELINE[c][4] for c in labels]

    x    = np.arange(len(labels))
    w    = 0.35
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('QBot (ours) vs. Paper Baseline (Leisiazar et al. 2025)',
                 fontsize=12, fontweight='bold')

    for ax, our_m, our_s, pap_m, pap_s, ylabel, title in [
        (ax1, our_dist_m,  our_dist_s,  paper_dist_m,  paper_dist_s,
         'Mean distance error (m)', 'Distance Error'),
        (ax2, our_alpha_m, our_alpha_s, paper_alpha_m, paper_alpha_s,
         'Mean α (rad)',             'Alpha Angle'),
    ]:
        ax.bar(x - w/2, our_m, w, yerr=our_s, capsize=4,
               color='#E8520A', alpha=0.85, label='Ours (QBot, ROS2)')
        ax.bar(x + w/2, pap_m, w, yerr=pap_s, capsize=4,
               color='#2E86AB', alpha=0.85, label='Paper (RB1, ROS1)')
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[ok] comparison figure → {out_path}')


# ── single-run summary table ──────────────────────────────────────────────────

def print_summary_table(rows: list):
    if not rows:
        return
    header = (f"{'Case':<22} {'Steps':>6} {'Dur':>7} "
              f"{'DistErr(m)':>14} {'Alpha(rad)':>13} {'Alpha(°)':>9} "
              f"{'Reward':>8} {'InCone%':>8} {'InZone%':>8}")
    print('\n' + '─' * len(header))
    print('Follow-Ahead Simulation Summary')
    print('─' * len(header))
    print(header)
    print('─' * len(header))
    for r in sorted(rows, key=lambda x: x['test_case']):
        print(f"  {r['test_case']:<20} {r['n_steps']:>6} {r['duration_s']:>6.1f}s"
              f"  {r['mean_dist_err']:.3f}±{r['std_dist_err']:.3f}"
              f"  {r['mean_alpha_rad']:.3f}±{r['std_alpha_rad']:.3f}"
              f"  {r['mean_alpha_deg']:>7.1f}°"
              f"  {r['mean_reward']:>7.3f}"
              f"  {r['pct_in_cone']:>7.1f}%"
              f"  {r['pct_in_zone']:>7.1f}%")
    print('─' * len(header) + '\n')


# ── entry point ───────────────────────────────────────────────────────────────

def collect_csv_paths(inputs: list) -> list:
    paths = []
    for inp in inputs:
        if os.path.isdir(inp):
            for f in sorted(os.listdir(inp)):
                if f.endswith('.csv') and '_plot' not in f:
                    paths.append(os.path.join(inp, f))
        elif os.path.isfile(inp) and inp.endswith('.csv'):
            paths.append(inp)
        else:
            print(f'[warn] skipping {inp}')
    return paths


def main():
    parser = argparse.ArgumentParser(description='Plot follow-ahead simulation results.')
    parser.add_argument('inputs', nargs='*',
                        help='CSV file(s) or directory containing CSVs')
    parser.add_argument('--summary',      action='store_true',
                        help='Print single-run summary table')
    parser.add_argument('--multi-run',    action='store_true',
                        help='Aggregate multiple runs per test case (mean ± std across runs)')
    parser.add_argument('--paper-compare', action='store_true',
                        help='Include paper baseline in comparison output')
    args = parser.parse_args()

    paths = collect_csv_paths(args.inputs)
    if not paths:
        print('No CSV files found.')
        sys.exit(1)

    if args.multi_run:
        # ── multi-run mode ──────────────────────────────────────────────────
        aggregated = aggregate_runs(paths)

        # Plot one representative figure per test case (run 01 = nominal pose)
        for case, data in aggregated.items():
            rep_path = sorted(
                [p for p in paths if _case_from_path(p) == case]
            )[0]  # alphabetically first = run01 (lowest timestamp)
            try:
                df = load_csv(rep_path)
                out = rep_path.replace('.csv', '_plot.png')
                plot_one(df, f'{case.capitalize()} (n={data["n"]} runs)', out)
                print(f'[ok] {case:<22} representative plot → {out}')
            except Exception as e:
                print(f'[error] {rep_path}: {e}')

        print_comparison_table(aggregated)

        if args.paper_compare:
            first_csv  = paths[0]
            comp_out   = os.path.join(os.path.dirname(first_csv), 'comparison_vs_paper.png')
            save_comparison_figure(aggregated, comp_out)

    else:
        # ── single-run mode ─────────────────────────────────────────────────
        summary_rows = []
        for path in paths:
            try:
                df  = load_csv(path)
            except Exception as e:
                print(f'[error] {path}: {e}')
                continue

            case_label = _case_from_path(path).capitalize()
            out_path   = path.replace('.csv', '_plot.png')
            stats      = plot_one(df, case_label, out_path)
            summary_rows.append(stats)
            print(f'[ok] {case_label:<22}  →  {out_path}')

        if args.summary or len(paths) > 1:
            print_summary_table(summary_rows)


if __name__ == '__main__':
    main()
