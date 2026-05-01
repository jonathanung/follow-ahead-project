"""
nav_env.py — 2D Gymnasium environment for Follow-Ahead A2C training.
================================================================

CHANGES FROM ORIGINAL
---------------------
1. Action space: 16 (arbitrary compass) → 6 (left/right/straight × 2 speeds)
   Matches the MCTS tree vocabulary in navi_state.py exactly so V(s) is
   consistent between training and tree evaluation.

2. Robot motion model: absolute compass angles → relative turn from heading
   Matches navi_state.py's calculate_new_state() kinematics.

3. Human walk: uniform random → biased (mostly straight, p=0.5/0.25/0.25)
   More realistic pedestrian motion.

4. Gymnasium semantics:
   - terminated = False  (no explicit goal/fail state in this task)
   - truncated  = True   when step limit is hit (correct for time limits)

5. Reward: inline formula replaced by import from reward.py (pure functions
   shared with the MCTS node evaluator in search.py).

6. State: raw arrays replaced by FollowState dataclass from state.py.
   Exposes a .state property so MCTS nodes can read distance/alpha directly.

7. Observation space: (-inf,inf) → bounded Box for better normalisation.
"""

import math

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Optional

from state import FollowState
from reward import reward as calc_reward


# ---------------------------------------------------------------------------
# Action vocabulary — MUST match navi_state.py's robot_acts list
# ---------------------------------------------------------------------------
#
#  idx   name            turn (rel. to current heading)   speed
#   0    left            +turn_angle                      normal
#   1    right           -turn_angle                      normal
#   2    straight         0                               normal
#   3    fast_left       +turn_angle                      fast (×lambda)
#   4    fast_right      -turn_angle                      fast (×lambda)
#   5    fast_straight    0                               fast (×lambda)
#
NUM_ACTIONS  = 6
ACTION_NAMES = ["left", "right", "straight", "fast_left", "fast_right", "fast_straight"]


class Environment(gym.Env):
    """
    Autonomous Frontal-Following 2D Simulation Environment.

    Designed for Gymnasium 1.x + Stable-Baselines3 v2.7+.
    Action space matches the MCTS tree in navi_state.py (6 discrete actions).

    Parameters
    ----------
    robot_vel      : float  — normal robot speed [m/step]
    robot_vel_fast : float  — fast robot speed [m/step]  (lambda × normal)
    human_vel      : float  — human walking speed [m/step]
    turn_angle_deg : float  — turn angle per step [degrees]
    max_steps      : int    — maximum episode length
    world_size     : float  — world is world_size × world_size [m]
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        robot_vel: float = 0.10,       # 0.5 m/s × 0.2 s — matches MCTS ROBOT_VEL
        robot_vel_fast: float = 0.12,  # 0.6 m/s × 0.2 s — matches MCTS ROBOT_VEL_FAST
        human_vel: float = 0.08,       # 0.4 m/s × 0.2 s — matches MCTS HUMAN_VEL
        turn_angle_deg: float = 6.0,   # 0.5 rad/s × 0.2 s — matches MCTS ROBOT_TURN
        max_steps: int = 500,          # longer episodes at slow step size
        world_size: float = 10.0,      # 10m box: robot travels ~0.10×500=50m max
        # Legacy aliases so old make_env(target_distance=...) calls don't crash
        target_distance: Optional[float] = None,   # unused — kept for back-compat
    ):
        super().__init__()

        # --- simulation parameters ---
        self.robot_vel       = robot_vel
        self.robot_vel_fast  = robot_vel_fast
        self.human_vel       = human_vel
        self.turn_angle      = math.radians(turn_angle_deg)
        self.max_steps       = max_steps
        self.world_size      = world_size

        # --- action space: 6 discrete actions (MCTS vocabulary) ---
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        # --- observation space: [dx, dy, h_theta, r_theta], bounded ---
        obs_high = np.array(
            [world_size, world_size, np.pi, np.pi], dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-obs_high, high=obs_high, dtype=np.float32
        )

        # internal state (set in reset)
        self._state: Optional[FollowState] = None
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Reset to a random start. Returns (obs, info)."""
        super().reset(seed=seed)

        lo, hi = 2.0, self.world_size - 2.0   # keep agents away from walls

        hx, hy = self.np_random.uniform(lo, hi, size=2)
        rx, ry = self.np_random.uniform(lo, hi, size=2)

        h_theta = float(self.np_random.uniform(-np.pi, np.pi))
        r_theta = float(self.np_random.uniform(-np.pi, np.pi))

        self._state = FollowState(
            human_x=float(hx), human_y=float(hy), human_theta=h_theta,
            robot_x=float(rx), robot_y=float(ry), robot_theta=r_theta,
        )
        self._step_count = 0
        return self._state.to_numpy(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Advance one timestep.

        Returns
        -------
        obs        : np.ndarray  shape (4,) float32
        reward     : float       in [-1, 1]
        terminated : bool        always False (no explicit success/fail state)
        truncated  : bool        True when episode step limit is reached
        info       : dict
        """
        assert self._state is not None, "Call reset() before step()."
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # 1. Move robot (relative turn model matching navi_state.py)
        self._apply_robot_action(action)

        # 2. Move human (biased random walk — mostly straight)
        self._step_human()

        # 3. Clamp both agents inside the world boundary
        self._clamp_to_world()

        # 4. Reward from shared reward module (same function as MCTS uses)
        r = calc_reward(self._state.distance, self._state.alpha)

        # 5. Termination
        self._step_count += 1
        terminated = False                                # no terminal goal state
        truncated  = self._step_count >= self.max_steps  # time limit → truncated

        return self._state.to_numpy(), r, terminated, truncated, {}

    # ------------------------------------------------------------------
    # Internal dynamics helpers
    # ------------------------------------------------------------------

    def _apply_robot_action(self, action: int) -> None:
        """
        Update robot pose based on a discrete action.

        Relative turn model: the robot turns by ±turn_angle relative to its
        *current* heading, then moves forward — exactly as navi_state.py
        calculate_new_state() does.
        """
        s = self._state

        # Turn delta and speed for each action index
        if action == 0:    # left
            delta, speed = +self.turn_angle, self.robot_vel
        elif action == 1:  # right
            delta, speed = -self.turn_angle, self.robot_vel
        elif action == 2:  # straight
            delta, speed = 0.0, self.robot_vel
        elif action == 3:  # fast_left
            delta, speed = +self.turn_angle, self.robot_vel_fast
        elif action == 4:  # fast_right
            delta, speed = -self.turn_angle, self.robot_vel_fast
        else:              # fast_straight (action == 5)
            delta, speed = 0.0, self.robot_vel_fast

        # New heading — wrap to (-pi, pi]
        new_theta = (s.robot_theta + delta + np.pi) % (2 * np.pi) - np.pi

        self._state = FollowState(
            human_x=s.human_x,
            human_y=s.human_y,
            human_theta=s.human_theta,
            robot_x=s.robot_x + speed * math.cos(new_theta),
            robot_y=s.robot_y + speed * math.sin(new_theta),
            robot_theta=new_theta,
        )

    def _step_human(self) -> None:
        """
        Advance the human one step with a biased random heading change.

        Probabilities: 50% straight, 25% slight left, 25% slight right.
        More realistic than the original uniform distribution and creates
        a better training signal for anticipatory following.
        """
        s = self._state
        delta = float(self.np_random.choice(
            [0.0, self.turn_angle, -self.turn_angle],
            p=[0.5, 0.25, 0.25],
        ))
        new_h_theta = (s.human_theta + delta + np.pi) % (2 * np.pi) - np.pi

        self._state = FollowState(
            human_x=s.human_x + self.human_vel * math.cos(new_h_theta),
            human_y=s.human_y + self.human_vel * math.sin(new_h_theta),
            human_theta=new_h_theta,
            robot_x=s.robot_x,
            robot_y=s.robot_y,
            robot_theta=s.robot_theta,
        )

    def _clamp_to_world(self) -> None:
        """Keep both agents inside [0, world_size] × [0, world_size]."""
        s = self._state
        lo, hi = 0.0, self.world_size
        self._state = FollowState(
            human_x=float(np.clip(s.human_x, lo, hi)),
            human_y=float(np.clip(s.human_y, lo, hi)),
            human_theta=s.human_theta,
            robot_x=float(np.clip(s.robot_x, lo, hi)),
            robot_y=float(np.clip(s.robot_y, lo, hi)),
            robot_theta=s.robot_theta,
        )

    # ------------------------------------------------------------------
    # Public accessors for MCTS / logging
    # ------------------------------------------------------------------

    @property
    def state(self) -> FollowState:
        """Read-only access to the current FollowState.

        Used by the MCTS node evaluator to compute distance and alpha
        without re-deriving them from the raw observation vector.
        """
        assert self._state is not None, "Call reset() before accessing state."
        return self._state