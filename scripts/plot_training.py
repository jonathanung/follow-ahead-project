#!/usr/bin/env python3
"""
plot_training.py — Training visualizations for the Follow-Ahead system.

Generates two figures:
  1. RL (A2C) training reward curve with best-model marker
     → RL_sim/training_curve.png
  2. LSTM human predictor train/test loss curve
     → lstm-fc/lstm_fc/training/final_v3/training_curve.png

Usage:
    python3 scripts/plot_training.py
"""

import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.join(SCRIPT_DIR, '..')

EVAL_NPZ  = os.path.join(REPO_ROOT, 'RL_sim', 'logs', 'evaluations.npz')
LSTM_LOG  = os.path.join(REPO_ROOT, 'lstm-fc', 'outputs', 'final_v3', 'training.log')


def plot_rl() -> str:
    if not os.path.exists(EVAL_NPZ):
        print(f'[skip] RL eval data not found: {EVAL_NPZ}')
        return ''

    data    = np.load(EVAL_NPZ)
    steps   = data['timesteps'] / 1e6        # convert to millions
    results = data['results']                # shape (N_evals, N_episodes)
    mean_r  = results.mean(axis=1)
    std_r   = results.std(axis=1)
    best_idx = int(np.argmax(mean_r))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(steps, mean_r, color='steelblue', lw=1.8, label='Mean eval reward (10 episodes)')
    ax.fill_between(steps, mean_r - std_r, mean_r + std_r,
                    alpha=0.20, color='steelblue', label='±1 std')
    ax.axvline(steps[best_idx], color='crimson', linestyle='--', lw=1.5,
               label=f'Best model deployed  ({steps[best_idx]:.1f} M steps, reward {mean_r[best_idx]:.2f})')
    ax.set_xlabel('Training timesteps (millions)', fontsize=11)
    ax.set_ylabel('Episode reward', fontsize=11)
    ax.set_title('A2C Follow-Ahead: Evaluation Reward During Training', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = os.path.join(REPO_ROOT, 'RL_sim', 'training_curve.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[ok] RL training curve  → {out}')
    print(f'     Best model @ {steps[best_idx]:.2f} M steps, mean reward = {mean_r[best_idx]:.3f}')
    return out


def plot_lstm() -> str:
    if not os.path.exists(LSTM_LOG):
        print(f'[skip] LSTM training log not found: {LSTM_LOG}')
        return ''

    steps, train_loss, test_loss = [], [], []
    # Handles both log formats:
    #   "Step N/M | Train Loss: X | Test Loss: Y"
    #   "Step N/M | Train: X | Test MSE: Y"
    pat = re.compile(
        r'Step\s+(\d+)/\d+.*?Train(?:\s+Loss)?:\s*([\d.eE+\-]+).*?Test(?:\s+MSE)?:\s*([\d.eE+\-]+)'
    )
    with open(LSTM_LOG) as f:
        for line in f:
            m = pat.search(line)
            if m:
                steps.append(int(m.group(1)))
                train_loss.append(float(m.group(2)))
                test_loss.append(float(m.group(3)))

    if not steps:
        print(f'[skip] no matching lines found in {LSTM_LOG}')
        return ''

    steps      = np.array(steps)
    train_loss = np.array(train_loss)
    test_loss  = np.array(test_loss)
    best_idx   = int(np.argmin(test_loss))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.semilogy(steps, train_loss, color='steelblue',  lw=1.5, label='Train loss (MSE)')
    ax.semilogy(steps, test_loss,  color='darkorange', lw=1.5, label='Test loss (MSE)')
    ax.axvline(steps[best_idx], color='crimson', linestyle='--', lw=1.5,
               label=f'Best test loss {test_loss[best_idx]:.5f} @ step {steps[best_idx]:,}')
    ax.set_xlabel('Training step', fontsize=11)
    ax.set_ylabel('MSE loss (log scale)', fontsize=11)
    ax.set_title('LSTM Human Predictor: Training Loss Curve (200 k steps)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    fig.tight_layout()

    out = os.path.join(REPO_ROOT, 'lstm-fc', 'outputs', 'final_v3', 'training_curve.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[ok] LSTM training curve → {out}')
    print(f'     Best test loss {test_loss[best_idx]:.5f} @ step {steps[best_idx]:,}')
    return out


if __name__ == '__main__':
    rl_out   = plot_rl()
    lstm_out = plot_lstm()
    if not rl_out and not lstm_out:
        print('[error] no outputs generated — check file paths above')
        sys.exit(1)
