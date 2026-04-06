"""
state.py — Shared state definition for the Follow-Ahead RL + MCTS system.

This dataclass is the single contract between:
  - The 2D simulation environment (nav_env.py)
  - The A2C training loop (train_a2c.py)
  - The MCTS evaluator (search.py / RL_interface.py)

By centralising the state schema here, every part of the system talks
about positions/orientations in the same way, and the observation vector
handed to the RL policy is identical whether it comes from the sim or the
MCTS tree expansion.

Mirrors Gemini_RL_sim/utils/state.py — kept in sync as a shared source of truth.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class FollowState:
    """
    Full pose of the human and the robot in 2D space.

    All positions are in metres relative to the same world frame.
    All angles (theta) are in radians, measured counter-clockwise from
    the positive x-axis.  Wrapped to (-pi, pi].

    Attributes
    ----------
    human_x     : float  — human x position [m]
    human_y     : float  — human y position [m]
    human_theta : float  — human heading angle [rad]
    robot_x     : float  — robot x position [m]
    robot_y     : float  — robot y position [m]
    robot_theta : float  — robot heading angle [rad]
    """
    human_x: float
    human_y: float
    human_theta: float
    robot_x: float
    robot_y: float
    robot_theta: float

    # ------------------------------------------------------------------
    # Observation vector
    # ------------------------------------------------------------------

    def to_numpy(self) -> np.ndarray:
        """Return state as a flat (4,) float32 array: [dx, dy, h_theta, r_theta].

        dx, dy are the robot position *relative to the human* so the RL agent
        sees a translation-invariant observation — it only needs to know where
        it is relative to the human, not absolute world coordinates.
        """
        dx = self.robot_x - self.human_x
        dy = self.robot_y - self.human_y
        return np.array([dx, dy, self.human_theta, self.robot_theta],
                        dtype=np.float32)

    # ------------------------------------------------------------------
    # Derived quantities used by the reward function and MCTS
    # ------------------------------------------------------------------

    @property
    def distance(self) -> float:
        """Euclidean distance between robot and human [m]."""
        return float(np.linalg.norm(
            [self.robot_x - self.human_x, self.robot_y - self.human_y]
        ))

    @property
    def bearing_angle(self) -> float:
        """
        Angle from the human to the robot, in the world frame [rad].

        beta = atan2(robot_y - human_y,  robot_x - human_x)

        Matches navi_state.py line 51:
            beta = np.arctan2(state[0,1]-state[1,1], state[0,0]-state[1,0])
        """
        return float(np.arctan2(
            self.robot_y - self.human_y,
            self.robot_x - self.human_x,
        ))

    @property
    def alpha(self) -> float:
        """
        Angular difference between the human's heading and the
        human→robot bearing vector [degrees, range [0, 180]].

        This drives the orientation reward r_alpha.
        alpha = 0  → robot is directly ahead of the human (ideal).
        alpha = 180 → robot is directly behind the human (worst).

        Matches navi_state.py line 52:
            diff = |human_theta - beta| * 180/pi, wrapped to [0,180]
        """
        diff = abs(self.human_theta - self.bearing_angle) * 180.0 / np.pi
        if diff > 180.0:
            diff = 360.0 - diff
        return float(diff)
