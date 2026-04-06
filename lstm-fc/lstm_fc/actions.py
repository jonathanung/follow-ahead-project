"""Shared action definitions for the follow-ahead system.

Both the LSTM predictor and MCTS planner import from here so action
indices, names, and mappings stay in sync.
"""

from enum import IntEnum
from typing import Dict


class HumanAction(IntEnum):
    """Discrete human actions predicted by the LSTM.

    The LSTM outputs a softmax probability distribution over these
    three actions.  The MCTS uses these probabilities to weight the
    UCB score when expanding human-turn nodes.

    Index order matches the LSTM output tensor: [left, straight, right].
    """

    LEFT = 0
    STRAIGHT = 1
    RIGHT = 2


class RobotAction(IntEnum):
    """Discrete robot actions available to the MCTS planner.

    Fast variants multiply base velocity by ``FAST_LAMBDA``.
    Angular displacement per step is ±45 degrees for turn actions.
    """

    FAST_LEFT = 0
    FAST_RIGHT = 1
    FAST_STRAIGHT = 2
    LEFT = 3
    RIGHT = 4
    STRAIGHT = 5


# String names matching legacy dict keys used throughout the system.
HUMAN_ACTION_NAMES = ("left", "straight", "right")
ROBOT_ACTION_NAMES = (
    "fast_left",
    "fast_right",
    "fast_straight",
    "left",
    "right",
    "straight",
)

# Kinematic parameters shared between forward model and MCTS.
DECISION_HZ = 5  # Control loop frequency
DECISION_DT = 1.0 / DECISION_HZ  # 0.2 s per step
HUMAN_VEL = 0.6  # m/s base human walking speed
ROBOT_VEL = 0.6  # m/s base robot speed
FAST_LAMBDA = 1.5  # Multiplier for fast robot actions
TURN_ANGLE_DEG = 45.0  # Degrees per turn action
INPUT_LENGTH = 14  # Trajectory points the LSTM expects
