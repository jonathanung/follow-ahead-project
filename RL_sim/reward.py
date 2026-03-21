"""
reward.py — Reward functions for the Follow-Ahead RL agent.

Implements the reward signal from the paper (equations 3a, 3b, 3c):

    r_d     — distance reward  (Eq. 3b): peaked at 1.5 m, penalties outside zone
    r_alpha — orientation reward (Eq. 3c): rewards being in the forward cone
    reward  — combined total reward (Eq. 3a): clipped to [-1, 1]

These are PURE FUNCTIONS — no class state, no ROS dependency, no gym import.
This means they can be called identically from:
  - nav_env.py   (RL training)
  - search.py    (MCTS node evaluation via RL_interface.py)

Faithfully reproduces navi_state.py's calculate_reward() from the ROS1 repo.
Key difference vs. the old nav_env.py inline code:
  - r_d thresholds match navi_state.py exactly (D_MIN=0.5, D_MAX=4.0, ramp at 2.0)
  - r_d is rescaled from [-1,1] → [0,1] before combining with r_alpha
    (matches navi_state.py lines 71-73: r_d /= 2; r_d += 0.5)
"""

import numpy as np

# ---------------------------------------------------------------------------
# Target zone constants (from navi_state.py params)
# ---------------------------------------------------------------------------

D_MIN   = 0.5   # [m] — any closer is a collision risk → r_d = -1
D_MAX   = 4.0   # [m] — too far behind → r_d = -1
D_IDEAL = 1.5   # [m] — peak of the distance reward

ALPHA_THRESHOLD_DEG = 25.0  # [degrees] — forward cone half-angle for r_alpha


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
# Equation 3c — orientation reward  r_alpha ∈ [-0.25, 1.0]
# ---------------------------------------------------------------------------

def r_alpha(alpha_deg: float) -> float:
    """
    Orientation reward — rewards the agent for being in the human's forward cone.

        alpha < ALPHA_THRESHOLD_DEG  →  (threshold - alpha) / threshold
                                         peaks at 1.0 when alpha=0° (directly ahead)
        alpha ≥ ALPHA_THRESHOLD_DEG  →  -0.25 * alpha / 180
                                         small growing penalty for being to the side/behind

    Parameters
    ----------
    alpha_deg : float — angle between human heading and human→robot vector [degrees]
                        as produced by FollowState.alpha  (always in [0, 180])

    Returns
    -------
    float — r_alpha in roughly [-0.25, 1.0]
    """
    a = float(alpha_deg)

    if a < ALPHA_THRESHOLD_DEG:
        return (ALPHA_THRESHOLD_DEG - a) / ALPHA_THRESHOLD_DEG
    else:
        return -0.25 * a / 180.0


# ---------------------------------------------------------------------------
# Equation 3a — combined total reward  r ∈ [-1, 1]
# ---------------------------------------------------------------------------

def reward(distance: float, alpha_deg: float) -> float:
    """
    Total reward (Eq. 3a):  r = clip( r_d(distance) + r_alpha(alpha_deg), -1, 1 )

    This is the function called each step by nav_env.py and each node
    evaluation by search.py.

    Parameters
    ----------
    distance  : float — robot–human Euclidean distance [m]
    alpha_deg : float — angular offset [degrees], from FollowState.alpha

    Returns
    -------
    float — clipped combined reward in [-1, 1]
    """
    total = r_d(distance) + r_alpha(alpha_deg)
    return float(np.clip(total, -1.0, 1.0))
