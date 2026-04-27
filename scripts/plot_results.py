#!/usr/bin/env python3
"""
plot_results.py — Generate 4-panel performance plots from follow-ahead simulation CSV logs.

Usage:
    python3 plot_results.py ~/follow_data/circle_20240101_120000.csv
    python3 plot_results.py ~/follow_data/          # plots all CSVs in directory
    python3 plot_results.py ~/follow_data/ --summary  # also prints summary table

Metrics match the paper (Leisiazar et al., IEEE RA-L 2025):
    mean_dist_error  — mean |distance - 1.5 m|
    mean_alpha_rad   — mean alpha in radians (human forward-cone offset)
    mean_reward      — mean combined reward per step
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # headless — works without a display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── paper constants ────────────────────────────────────────────────────────────
D_TARGET   = 1.5    # m  — ideal robot–human distance
D_MIN      = 0.5    # m  — collision threshold
D_MAX      = 4.0    # m  — too-far threshold
ALPHA_CONE = 50.0   # °  — forward-cone half-angle for r_alpha reward

# ── colour palette ─────────────────────────────────────────────────────────────
C_ROBOT = '#E8520A'   # orange-red  (matches RViz marker)
C_HUMAN = '#1A7DC4'   # blue        (matches RViz marker)
C_DIST  = '#2E86AB'
C_ALPHA = '#A23B72'
C_RD    = '#F18F01'
C_RA    = '#C73E1D'
C_REW   = '#3B1F2B'


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {'elapsed_s', 'robot_x', 'robot_y', 'human_x', 'human_y',
                 'distance', 'alpha_deg', 'alpha_rad', 'dist_error', 'reward'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    return df


def _case_label(path: str) -> str:
    """Extract test-case name from filename, e.g. 'circle_20240101_120000.csv' → 'circle'."""
    base = os.path.splitext(os.path.basename(path))[0]
    # filename format: <test_case>_<YYYYMMDD>_<HHMMSS>
    parts = base.split('_')
    # test case names: circle, stationary, square, oscillate, zigzag
    known = {'circle', 'stationary', 'square', 'oscillate', 'zigzag',
             'gentle', 'approach', 'unknown'}
    label = parts[0]
    if label not in known and len(parts) > 1:
        # try two-word case names (none currently, but future-proof)
        label = '_'.join(parts[:2]) if '_'.join(parts[:2]) in known else parts[0]
    return label.capitalize()


def plot_one(df: pd.DataFrame, case_label: str, out_path: str):
    """Generate the 4-panel figure for a single test case and save to out_path."""

    # ── summary stats ──────────────────────────────────────────────────────────
    # Skip the first 6 s (startup delay) so metrics only cover active following
    active = df[df['elapsed_s'] >= 6.0].copy() if (df['elapsed_s'] >= 6.0).any() else df

    mean_dist_err  = active['dist_error'].mean()
    std_dist_err   = active['dist_error'].std()
    mean_alpha_rad = active['alpha_rad'].mean()
    std_alpha_rad  = active['alpha_rad'].std()
    mean_alpha_deg = active['alpha_deg'].mean()
    mean_reward    = active['reward'].mean()
    pct_in_cone    = (active['alpha_deg'] < ALPHA_CONE).mean() * 100.0
    pct_in_zone    = ((active['distance'] >= D_MIN) & (active['distance'] <= D_MAX)).mean() * 100.0

    suptitle = (
        f"{case_label}  —  "
        f"dist_err = {mean_dist_err:.3f} ± {std_dist_err:.3f} m  |  "
        f"α = {mean_alpha_rad:.3f} ± {std_alpha_rad:.3f} rad ({mean_alpha_deg:.1f}°)  |  "
        f"reward = {mean_reward:.3f}"
    )

    t = df['elapsed_s'].values

    # ── figure layout ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(suptitle, fontsize=11, fontweight='bold', y=0.98)
    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    ax_traj  = fig.add_subplot(gs[0, 0])
    ax_dist  = fig.add_subplot(gs[0, 1])
    ax_alpha = fig.add_subplot(gs[1, 0])
    ax_rew   = fig.add_subplot(gs[1, 1])

    # ── Panel 1: 2-D trajectories ──────────────────────────────────────────────
    ax = ax_traj
    ax.plot(df['human_x'], df['human_y'],
            color=C_HUMAN, linewidth=1.6, label='Human', zorder=2)
    ax.plot(df['robot_x'], df['robot_y'],
            color=C_ROBOT, linewidth=1.6, label='Robot', zorder=2)

    # start markers
    ax.scatter(df['human_x'].iloc[0], df['human_y'].iloc[0],
               marker='o', s=80, color=C_HUMAN, zorder=5, edgecolors='white', linewidths=0.8)
    ax.scatter(df['robot_x'].iloc[0], df['robot_y'].iloc[0],
               marker='s', s=80, color=C_ROBOT, zorder=5, edgecolors='white', linewidths=0.8)

    # end markers
    ax.scatter(df['human_x'].iloc[-1], df['human_y'].iloc[-1],
               marker='o', s=60, color=C_HUMAN, zorder=5, alpha=0.5)
    ax.scatter(df['robot_x'].iloc[-1], df['robot_y'].iloc[-1],
               marker='s', s=60, color=C_ROBOT, zorder=5, alpha=0.5)

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('2-D Trajectories', fontsize=10)
    ax.set_aspect('equal', adjustable='datalim')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.25)

    _add_note(ax, f'in-zone {pct_in_zone:.0f}%  |  in-cone {pct_in_cone:.0f}%', fontsize=7.5)

    # ── Panel 2: distance over time ────────────────────────────────────────────
    ax = ax_dist
    ax.plot(t, df['distance'], color=C_DIST, linewidth=1.2, label='distance', zorder=3)
    ax.axhline(D_TARGET, color='green',  linewidth=1.2, linestyle='--', label=f'target {D_TARGET} m')
    ax.axhline(D_MIN,    color='red',    linewidth=0.9, linestyle=':', alpha=0.7, label=f'min {D_MIN} m')
    ax.axhline(D_MAX,    color='orange', linewidth=0.9, linestyle=':', alpha=0.7, label=f'max {D_MAX} m')

    # shade the 6-s startup window
    ax.axvspan(0, 6, alpha=0.08, color='grey', label='startup delay')

    ax.set_xlabel('time (s)')
    ax.set_ylabel('distance (m)')
    ax.set_title('Robot–Human Distance', fontsize=10)
    ax.legend(fontsize=7.5, loc='upper right')
    ax.grid(True, alpha=0.25)
    ax.set_ylim(bottom=0)

    # ── Panel 3: alpha angle over time ─────────────────────────────────────────
    ax = ax_alpha
    ax.plot(t, df['alpha_deg'], color=C_ALPHA, linewidth=1.2, label='α (°)')
    ax.axhline(ALPHA_CONE, color='red', linewidth=1.0, linestyle='--',
               label=f'cone limit {ALPHA_CONE}°')
    ax.fill_between(t, df['alpha_deg'], ALPHA_CONE,
                    where=(df['alpha_deg'] > ALPHA_CONE),
                    alpha=0.18, color='red', label='outside cone')
    ax.axvspan(0, 6, alpha=0.08, color='grey')

    ax.set_xlabel('time (s)')
    ax.set_ylabel('α (degrees)')
    ax.set_title('Alpha Angle (human forward-cone offset)', fontsize=10)
    ax.legend(fontsize=7.5, loc='upper right')
    ax.grid(True, alpha=0.25)
    ax.set_ylim(0, 185)

    # right-hand axis in radians
    ax2 = ax.twinx()
    ax2.set_ylim(0, np.radians(185))
    ax2.set_ylabel('α (rad)', fontsize=8)
    ax2.tick_params(labelsize=7)

    # ── Panel 4: reward components over time ───────────────────────────────────
    ax = ax_rew
    ax.plot(t, df['r_d'],    color=C_RD,  linewidth=1.0, alpha=0.75, label='r_d (distance)')
    ax.plot(t, df['r_alpha'], color=C_RA,  linewidth=1.0, alpha=0.75, label='r_α (orientation)')
    ax.plot(t, df['reward'], color=C_REW, linewidth=1.5, label='total reward')
    ax.axhline(0, color='grey', linewidth=0.6, linestyle='-')
    ax.axvspan(0, 6, alpha=0.08, color='grey')

    ax.set_xlabel('time (s)')
    ax.set_ylabel('reward')
    ax.set_title('Reward Components', fontsize=10)
    ax.legend(fontsize=7.5, loc='lower right')
    ax.grid(True, alpha=0.25)

    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return {
        'test_case':       case_label,
        'n_steps':         len(active),
        'duration_s':      round(t[-1], 1),
        'mean_dist_err_m': round(mean_dist_err,  4),
        'std_dist_err_m':  round(std_dist_err,   4),
        'mean_alpha_rad':  round(mean_alpha_rad, 4),
        'std_alpha_rad':   round(std_alpha_rad,  4),
        'mean_alpha_deg':  round(mean_alpha_deg, 2),
        'mean_reward':     round(mean_reward,    4),
        'pct_in_cone':     round(pct_in_cone,    1),
        'pct_in_zone':     round(pct_in_zone,    1),
    }


def _add_note(ax, text, fontsize=8):
    ax.text(0.02, 0.02, text, transform=ax.transAxes,
            fontsize=fontsize, verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))


def print_summary_table(rows: list):
    if not rows:
        return
    header = (
        f"{'Case':<12} {'Steps':>6} {'Duration':>9} "
        f"{'DistErr(m)':>12} {'Alpha(rad)':>11} {'Alpha(°)':>9} "
        f"{'Reward':>8} {'InCone%':>8} {'InZone%':>8}"
    )
    print('\n' + '─' * len(header))
    print('Follow-Ahead Simulation Summary (paper metrics)')
    print('─' * len(header))
    print(header)
    print('─' * len(header))
    for r in sorted(rows, key=lambda x: x['test_case']):
        print(
            f"{r['test_case']:<12} {r['n_steps']:>6} {r['duration_s']:>8.1f}s "
            f"  {r['mean_dist_err_m']:.3f}±{r['std_dist_err_m']:.3f}  "
            f"  {r['mean_alpha_rad']:.3f}±{r['std_alpha_rad']:.3f}"
            f"  {r['mean_alpha_deg']:>7.1f}°"
            f"  {r['mean_reward']:>7.3f}"
            f"  {r['pct_in_cone']:>7.1f}%"
            f"  {r['pct_in_zone']:>7.1f}%"
        )
    print('─' * len(header) + '\n')


def collect_csv_paths(inputs: list) -> list:
    paths = []
    for inp in inputs:
        if os.path.isdir(inp):
            for f in sorted(os.listdir(inp)):
                if f.endswith('.csv') and not f.endswith('_plot.csv'):
                    paths.append(os.path.join(inp, f))
        elif os.path.isfile(inp) and inp.endswith('.csv'):
            paths.append(inp)
        else:
            print(f"[warn] skipping {inp} (not a .csv file or directory)")
    return paths


def main():
    parser = argparse.ArgumentParser(description='Plot follow-ahead simulation results.')
    parser.add_argument('inputs', nargs='+',
                        help='CSV file(s) or directory containing CSVs')
    parser.add_argument('--summary', action='store_true',
                        help='Print summary statistics table to stdout')
    args = parser.parse_args()

    paths = collect_csv_paths(args.inputs)
    if not paths:
        print('No CSV files found.')
        sys.exit(1)

    summary_rows = []
    for path in paths:
        try:
            df = load_csv(path)
        except Exception as e:
            print(f'[error] {path}: {e}')
            continue

        case_label = _case_label(path)
        out_path   = path.replace('.csv', '_plot.png')
        stats      = plot_one(df, case_label, out_path)
        summary_rows.append(stats)
        print(f'[ok] {case_label:12s}  →  {out_path}')

    if args.summary or len(paths) > 1:
        print_summary_table(summary_rows)


if __name__ == '__main__':
    main()
