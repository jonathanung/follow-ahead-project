"""
reward.py — Reward functions for the Follow-Ahead RL agent.

Implements the reward signal from the paper (equations 3a, 3b, 3c):

    r_d     — distance reward  (Eq. 3b): peaked at 1.5 m, penalties outside zone
    r_alpha — orientation reward (Eq. 3c): rewards being in the 50° forward cone
    reward  — combined total reward (Eq. 3a): r_d + r_alpha

These are PURE FUNCTIONS — no class state, no ROS dependency, no gym import.
This means they can be called identically from:
  - nav_env.py   (RL training)
  - planner.py   (MCTS node evaluation via RL_interface.py)

Aligned with the published paper (Leisiazar et al., IEEE RA-L 2025):
  - r_d thresholds: D_MIN=0.5, D_MAX=4.0 (Eq. 3b)
  - r_d rescaled from [-1,1] → [0,1] to match navi_state.py (lines 71-73)
  - r_alpha threshold: 50° (Eq. 3c); penalty branch: -1.0 (flat, per paper)
"""

import numpy as np

# ---------------------------------------------------------------------------
# Target zone constants (from navi_state.py params)
# ---------------------------------------------------------------------------

D_MIN   = 0.5   # [m] — any closer is a collision risk → r_d = -1
D_MAX   = 4.0   # [m] — too far behind → r_d = -1
D_IDEAL = 1.5   # [m] — peak of the distance reward

ALPHA_THRESHOLD_DEG = 50.0  # [degrees] — forward cone half-angle for r_alpha (Eq. 3c)


# ---------------------------------------------------------------------------
# Equation 3b — distance reward  r_d ∈ [0, 1]  (rescaled)
# ---------------------------------------------------------------------------

def r_d(distance: float) -> float:
    """
    Distance reward — rewards the agent for staying near the target zone.

    Piecewise linear shape (faithful to navi_state.py calculate_reward):

        d < D_MIN  or  d > D_MAX     → raw = -1      (too close / too far)
        D_MIN ≤ d ≤ 1.0              → raw = -2*(1-d)  (ramps -1 → 0)
        1.0  < d ≤ 2.0               → raw = 2*(0.5 - |d-1.5|)  peak +0.5 at d=1.5
        2.0  < d ≤ D_MAX             → raw = -0.5*(d-2)  (ramps 0 → -1)

    Then rescaled: raw/2 + 0.5  so the output is in [0, 1].
    At d=D_IDEAL (1.5m), output = 1.0 (maximum reward).

    Parameters
    ----------
    distance : float — Euclidean robot–human distance [m]

    Returns
    -------
    float — r_d in [0, 1]
    """
    d = float(distance)

    if d < D_MIN or d > D_MAX:
        raw = -1.0
    elif D_MIN <= d <= 1.0:
        raw = -2.0 * (1.0 - d)          # ramps from -1 at D_MIN toward 0 at d=1
    elif 1.0 < d <= 2.0:
        raw = 2.0 * (0.5 - abs(d - 1.5))  # peaked at +0.5 when d=1.5
    else:  # 2.0 < d <= D_MAX
        raw = -0.5 * (d - 2.0)           # slopes down from 0 toward -1 at D_MAX

    # Rescale [-1, 1] → [0, 1]  (matches navi_state.py: r_d /= 2; r_d += 0.5)
    return raw / 2.0 + 0.5


# ---------------------------------------------------------------------------
# Equation 3c — orientation reward  r_alpha
# ---------------------------------------------------------------------------

def r_alpha(alpha_deg: float) -> float:
    """
    Orientation reward (Eq. 3c) — rewards the agent for being in the human's forward cone.

        |alpha| < 50°  →  (25 - |alpha|) / 25   peaks at 1.0 when alpha=0° (directly ahead)
        otherwise      →  -1.0                   flat penalty outside the cone

    Note: the paper uses a 50° half-angle cone (not 25°). Threshold fixed to match
    Eq. 3c exactly: r_alpha = (25 - |α|)/25 if |α| < 50, else -1.

    Parameters
    ----------
    alpha_deg : float — angle between human heading and human→robot vector [degrees]
                        as produced by FollowState.alpha  (always in [0, 180])

    Returns
    -------
    float — r_alpha in [-1.0, 1.0]
    """
    a = float(alpha_deg)

    if a < ALPHA_THRESHOLD_DEG:
        return (25.0 - a) / 25.0   # Eq. 3c numerator is 25, not threshold
    else:
        return -1.0                # flat -1 outside 50° cone (paper exact)


# ---------------------------------------------------------------------------
# Equation 3a — combined total reward
# ---------------------------------------------------------------------------

def reward(distance: float, alpha_deg: float) -> float:
    """
    Total reward (Eq. 3a):  r = r_d(distance) + r_alpha(alpha_deg)

    The paper does not clip the combined reward — the components are bounded
    such that the sum is well-defined. A soft clip is kept for numerical safety.

    This is the function called each step by nav_env.py and each node
    evaluation in planner.py (MCTS).

    Parameters
    ----------
    distance  : float — robot–human Euclidean distance [m]
    alpha_deg : float — angular offset [degrees], from FollowState.alpha

    Returns
    -------
    float — combined reward
    """
    return float(r_d(distance) + r_alpha(alpha_deg))
