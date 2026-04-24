"""
planner.py — MCTS Planner for Follow-Ahead Robot Navigation.

Implements the tree search loop (select → expand → evaluate → backprop)
from Algorithm 1 in Leisiazar et al., IEEE RA-L 2025.

Key changes from the original stub:
  1. rl_value() is now wired to the trained A2C value function via RL_interface.
  2. Node evaluation uses the paper reward (Eq 3a/3b/3c) from RL_sim/reward.py
     instead of the grid's -(dist²) proxy.
  3. State conversion helper (_grid_state_to_obs) maps the grid dict →
     the [dx, dy, h_theta, r_theta] obs vector the RL model expects.

Model is loaded ONCE at import time (module-level singleton) so ROS2 nodes
that import MCTSPlanner don't re-load the model on every planning call.
"""

import os
import sys
import math
import time
import random
import copy
import numpy as np

from node import Node
# from human_model import human_probabilities  # Removed as it's missing in algo-main

# ---------------------------------------------------------------------------
# Path setup — add RL_sim to sys.path so we can import reward, state, and RL_interface
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_RL_SIM_DIR = os.path.join(_SCRIPT_DIR, '..', '..', 'RL_sim')
sys.path.insert(0, os.path.abspath(_RL_SIM_DIR))

from reward import reward as paper_reward
from RL_interface import RL_model
from state import FollowState

# ---------------------------------------------------------------------------
# MCTS & Kinematics hyperparameters (Aligned with ROS1 & RA-L 2025)
# ---------------------------------------------------------------------------
MAX_DEPTH        = 20
GAMMA            = 0.95
STAY_DISTANCE    = 1.5   # [m] - aligned with ROS1 stay() logic
ROBOT_VEL        = 0.5   # [m/step] normal
ROBOT_VEL_FAST   = 0.75  # [m/step] (0.5 * 1.5)
HUMAN_VEL        = 0.5   # [m/step]
ROBOT_TURN       = math.radians(30.0)  # [rad]
HUMAN_TURN       = math.radians(8.0)   # [rad]
EXPANSION_TIME   = 0.15  # [s] (5 Hz search budget)

SAFETY_R = 0.5   # [m] keep-out radius
SAFETY_A = 0.25  # [m] human-displaced center

ROBOT_ACTIONS = ["left", "right", "straight", "fast_left", "fast_right", "fast_straight"]
HUMAN_ACTIONS = ["left", "right", "straight"]

# ---------------------------------------------------------------------------
# RL model singleton — loaded once at module import time
# ---------------------------------------------------------------------------
_rl = RL_model()
_MODEL_PATH = os.path.join(_RL_SIM_DIR, 'models', 'a2c_follow_ahead')

try:
    _rl.load_model(_MODEL_PATH)
    print(f"[planner] RL model loaded from: {_MODEL_PATH}.zip")
except Exception as e:
    import logging
    logging.getLogger(__name__).error(
        f"[planner] FATAL: could not load RL model from {_MODEL_PATH} — {e}. "
        "MCTS will use zero value function, which severely degrades performance."
    )
    _rl = None


# ---------------------------------------------------------------------------
# State Dynamics: Relative turn kinematics (matches navi_state.py)
# ---------------------------------------------------------------------------

def calculate_new_state(state: FollowState, action: str, is_robot: bool) -> FollowState:
    """
    Apply a relative turn and step.
    Robot: uses ROBOT_TURN (45 deg)
    Human: uses HUMAN_TURN (10 deg)
    """
    if is_robot:
        angle_delta = 0.0
        speed = ROBOT_VEL
        if "left" in action:
            angle_delta = ROBOT_TURN
        elif "right" in action:
            angle_delta = -ROBOT_TURN
        
        if "fast" in action:
            speed = ROBOT_VEL_FAST
            
        new_theta = (state.robot_theta + angle_delta + math.pi) % (2 * math.pi) - math.pi
        return FollowState(
            human_x=state.human_x,
            human_y=state.human_y,
            human_theta=state.human_theta,
            robot_x=state.robot_x + speed * math.cos(new_theta),
            robot_y=state.robot_y + speed * math.sin(new_theta),
            robot_theta=new_theta
        )
    else:
        # Human movement
        angle_delta = 0.0
        if action == "left":
            angle_delta = HUMAN_TURN
        elif action == "right":
            angle_delta = -HUMAN_TURN
            
        new_h_theta = (state.human_theta + angle_delta + math.pi) % (2 * math.pi) - math.pi
        return FollowState(
            human_x=state.human_x + HUMAN_VEL * math.cos(new_h_theta),
            human_y=state.human_y + HUMAN_VEL * math.sin(new_h_theta),
            human_theta=new_h_theta,
            robot_x=state.robot_x,
            robot_y=state.robot_y,
            robot_theta=state.robot_theta
        )


def is_safe(state: FollowState) -> bool:
    """
    Collision avoidance using the Displaced Circle model from the paper.
    Calculates if the robot is outside the safety boundary around the person.
    """
    # alpha: angle between human heading and human-to-robot vector
    # Using human-centric alpha from FollowState
    alpha_rad = math.radians(state.alpha)
    
    # Boundary d_circle = a*cos(alpha) + sqrt(r^2 - a^2*sin^2(alpha))
    # This is the positive root of: d^2 - 2*d*a*cos(alpha) + a^2 - r^2 = 0
    a = SAFETY_A
    r = SAFETY_R
    
    d_circle = a * math.cos(alpha_rad) + math.sqrt(r**2 - (a * math.sin(alpha_rad))**2)
    
    return state.distance >= d_circle


# ---------------------------------------------------------------------------
# Node value = paper reward + discounted RL value estimate
# ---------------------------------------------------------------------------

def rl_value(state: FollowState) -> float:
    """
    Query the trained A2C critic to get V(s) for a FollowState.
    """
    if _rl is None:
        return 0.0
    obs = state.to_numpy()
    return _rl.evaluate_state(obs)


def _node_value(state: FollowState) -> float:
    """Combined node value: immediate paper reward + discounted RL estimate."""
    imm_reward = paper_reward(state.distance, state.alpha)
    return imm_reward + (rl_value(state) / 10.0) * GAMMA


# ---------------------------------------------------------------------------
# MCTS helpers
# ---------------------------------------------------------------------------

STAY_ALPHA_MAX = 70.0   # [deg] — only STAY when inside human's forward cone

def _stay_bool(state: FollowState, use_stay=False):
    if not use_stay:
        return False
    # Only stop when the robot is already ahead of the human (low alpha)
    # AND far enough that it doesn't need to keep moving.
    return state.distance > STAY_DISTANCE and state.alpha < STAY_ALPHA_MAX


def _expand_robot(node: Node):
    """
    Expand the robot's turn. Skip actions that lead to collision.
    """
    p = 1.0 / len(ROBOT_ACTIONS)
    for action in ROBOT_ACTIONS:
        next_s = calculate_new_state(node.state, action, is_robot=True)
        
        # --- Safety Pruning (Deviation alignment) ---
        if not is_safe(next_s):
            continue
            
        child = Node(next_s, parent=node,
                     action=action, prior=p, node_type='human')
        node.children.append(child)


def _expand_human(node, human_probs=None):
    if human_probs is not None:
        probs = human_probs
    else:
        # Default fallback (from nav_env.py)
        probs = {'straight': 0.5, 'left': 0.25, 'right': 0.25}

    for action, p in probs.items():
        # node.action is the robot's chosen action
        # First confirm the robot action was safe
        new_state = calculate_new_state(node.state, action, is_robot=False)
        
        child = Node(new_state, parent=node, action=action,
                     prior=float(p), node_type='robot')
        node.children.append(child)


def _select(node):
    depth = 0
    while not node.is_leaf():
        if depth >= MAX_DEPTH:
            break
        node = node.best_child()
        depth += 1
    return node


def _backprop(node, value):
    while node is not None:
        node.visits += 1
        node.total_value += value
        node = node.parent


def _run_tree(root, budget_fn, human_probs=None):
    while budget_fn():
        leaf = _select(root)
        if leaf.visits == 0:
            value = _node_value(leaf.state)
            _backprop(leaf, value)
            continue
            
        if leaf.node_type == 'robot':
            _expand_robot(leaf)
        else:
            _expand_human(leaf, human_probs)
            
        if leaf.children:
            leaf = random.choice(leaf.children)
            value = _node_value(leaf.state)
            _backprop(leaf, value)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class MCTSPlanner:
    """
    Monte Carlo Tree Search planner for Follow-Ahead navigation.

    Parameters
    ----------
    n_simulations : int  — fixed number of simulations per plan() call
                           (used if time_budget is None)
    time_budget   : float — wall-clock seconds per plan() call (default: 0.15s)
    verbose       : bool  — print root visit counts after each plan
    use_stay      : bool  — allow STAY action when robot is far from human
    """

    def __init__(self, n_simulations=None, time_budget=EXPANSION_TIME,
                 verbose=False, use_stay=False):
        self.n_simulations = n_simulations
        self.time_budget   = time_budget
        self.verbose       = verbose
        self.use_stay      = use_stay

    def plan(self, state: FollowState, human_probs: dict = None) -> str:
        """
        Run MCTS from the current state and return the best action.

        Parameters
        ----------
        state       : FollowState object
        human_probs : dict, optional — output from lstm_fc HumanActionPredictor
                                       {'left': float, 'straight': float, 'right': float}

        Returns
        -------
        str — best action: one of ROBOT_ACTIONS or 'STAY'
        """
        if _stay_bool(state, use_stay=self.use_stay):
            return 'STAY'

        root = Node(copy.deepcopy(state), node_type='robot')

        if self.time_budget is not None:
            deadline  = time.time() + self.time_budget
            budget_fn = lambda: time.time() < deadline
        else:
            counter = [0]
            limit   = self.n_simulations
            def budget_fn():
                counter[0] += 1
                return counter[0] <= limit

        _run_tree(root, budget_fn, human_probs)

        if self.verbose:
            print(f"[MCTSPlanner] root visits: {root.visits}")
            print(f"[MCTSPlanner] children: "
                  f"{[(c.action, c.visits) for c in root.children]}")

        return root.best_action_child().action