#!/usr/bin/env python3
"""
plot_trajectories.py — Trajectory-only plots for follow-ahead CSVs.

Generates one PNG per CSV showing only the 2-D trajectory panel
(top-left panel of the standard 4-panel figure): human path in Blues,
robot path in Reds, colour encoding time progression (dark=start, light=end).

Usage:
    python3 scripts/plot_trajectories.py ~/follow_data/circle_run01_*.csv ...
    python3 scripts/plot_trajectories.py -- /path/a.csv /path/b.csv ...

Output: <csv_basename>_traj.png alongside each CSV.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from matplotlib.collections import LineCollection

C_ROBOT = '#E8520A'
C_HUMAN = '#1A7DC4'

TARGET_DIST = 1.5   # m


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {'elapsed_s', 'robot_x', 'robot_y', 'human_x', 'human_y'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'{os.path.basename(path)}: missing columns {missing}')
    return df


def _case_label(path: str) -> str:
    name = os.path.splitext(os.path.basename(path))[0]
    # strip _run01_timestamp or _timestamp suffix
    parts = name.split('_run')
    if len(parts) > 1:
        return parts[0].replace('_', ' ').title()
    # plain timestamp form: <case>_YYYYMMDD_HHMMSS
    tokens = name.rsplit('_', 2)
    return tokens[0].replace('_', ' ').title() if len(tokens) == 3 else name.replace('_', ' ').title()


def _rainbow_line(ax, x, y, cmap_name):
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 2:
        return
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segs   = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segs, cmap=cmap_name, linewidth=1.8, zorder=2)
    lc.set_array(np.linspace(0, 1, len(x)))
    ax.add_collection(lc)
    ax.autoscale()


def plot_trajectory(csv_path: str) -> str:
    df    = load_csv(csv_path)
    label = _case_label(csv_path)
    dur   = df['elapsed_s'].iloc[-1]

    # metrics for subtitle
    if 'distance' in df.columns:
        dist_err = (df['distance'] - TARGET_DIST).abs().mean()
    else:
        dist_err = np.sqrt((df['robot_x']-df['human_x'])**2 +
                           (df['robot_y']-df['human_y'])**2).sub(TARGET_DIST).abs().mean()

    if 'alpha_deg' in df.columns:
        alpha_deg = df['alpha_deg'].abs().mean()
    elif 'alpha_rad' in df.columns:
        alpha_deg = np.degrees(df['alpha_rad'].abs().mean())
    else:
        alpha_deg = float('nan')

    if 'reward' in df.columns:
        reward = df['reward'].mean()
    else:
        reward = float('nan')

    fig, ax = plt.subplots(figsize=(6, 5))

    _rainbow_line(ax, df['human_x'], df['human_y'], 'Blues')
    _rainbow_line(ax, df['robot_x'], df['robot_y'],  'Reds')

    # start markers (filled) and end markers (faded)
    ax.scatter(*df[['human_x','human_y']].iloc[0],  marker='o', s=80,
               color=C_HUMAN, zorder=5, edgecolors='white', linewidths=0.8)
    ax.scatter(*df[['robot_x','robot_y']].iloc[0],   marker='s', s=80,
               color=C_ROBOT, zorder=5, edgecolors='white', linewidths=0.8)
    ax.scatter(*df[['human_x','human_y']].iloc[-1],  marker='o', s=60,
               color=C_HUMAN, zorder=5, alpha=0.4)
    ax.scatter(*df[['robot_x','robot_y']].iloc[-1],  marker='s', s=60,
               color=C_ROBOT, zorder=5, alpha=0.4)

    h_patch = mpatches.Patch(color=cm.Blues(0.7), label='Human')
    r_patch = mpatches.Patch(color=cm.Reds(0.7),  label='Robot')
    ax.legend(handles=[h_patch, r_patch], fontsize=9, loc='upper right')

    ax.set_xlabel('x (m)', fontsize=10)
    ax.set_ylabel('y (m)', fontsize=10)
    ax.set_title(f'{label}  —  {dur:.0f} s', fontsize=11, fontweight='bold')
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=0.25)

    # stats note in bottom-left corner
    note = f'dist_err={dist_err:.2f} m  |  α={alpha_deg:.1f}°  |  r={reward:.3f}'
    ax.text(0.02, 0.02, note, transform=ax.transAxes,
            fontsize=7.5, color='#444444', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7, ec='none'))

    fig.tight_layout()

    out = os.path.splitext(csv_path)[0] + '_traj.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out


def main():
    paths = [a for a in sys.argv[1:] if a != '--']
    if not paths:
        print(f'Usage: python3 {os.path.basename(__file__)} <csv> [<csv> ...]')
        sys.exit(1)

    ok, fail = 0, 0
    for p in paths:
        try:
            out = plot_trajectory(p)
            print(f'[ok] {os.path.basename(out)}')
            ok += 1
        except Exception as e:
            print(f'[err] {os.path.basename(p)}: {e}')
            fail += 1

    print(f'\n{ok} plot(s) saved, {fail} failed.')


if __name__ == '__main__':
    main()
