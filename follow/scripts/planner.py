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
from simple_grid import transition, ROBOT_ACTIONS, desired_robot_pos, move_in_dir, GRID_SIZE
from human_model import human_probabilities
from rollout import rollout

# ---------------------------------------------------------------------------
# Path setup — add RL_sim to sys.path so we can import reward and RL_interface
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_RL_SIM_DIR = os.path.join(_SCRIPT_DIR, '..', '..', 'RL_sim')
sys.path.insert(0, os.path.abspath(_RL_SIM_DIR))

from reward import reward as paper_reward   # Eq 3a — uses fixed 50° threshold
from RL_interface import RL_model

# ---------------------------------------------------------------------------
# MCTS hyperparameters (from paper)
# ---------------------------------------------------------------------------
MAX_DEPTH     = 20
GAMMA         = 0.95
STAY_DISTANCE = 2.5   # grid cells; if robot is very far, just STAY
CELL_SIZE     = 0.5   # metres per grid cell  (matches mcts_node.py)

# ---------------------------------------------------------------------------
# Heading → radians mapping  (matches mcts_node.py's ACTION_TO_YAW)
# ---------------------------------------------------------------------------
_HEADING_TO_RAD = {
    'N':  math.pi / 2,
    'E':  0.0,
    'S': -math.pi / 2,
    'W':  math.pi,
}

# ---------------------------------------------------------------------------
# RL model singleton — loaded once at module import time
# ---------------------------------------------------------------------------
_rl = RL_model()
_MODEL_PATH = os.path.join(_RL_SIM_DIR, 'models', 'a2c_follow_ahead')

try:
    _rl.load_model(_MODEL_PATH)
    print(f"[planner] RL model loaded from: {_MODEL_PATH}.zip")
except Exception as e:
    print(f"[planner] WARNING: could not load RL model ({e}). rl_value() will return 0.")
    _rl = None


# ---------------------------------------------------------------------------
# State conversion: grid dict → RL observation vector
# ---------------------------------------------------------------------------

def _grid_state_to_obs(state: dict) -> np.ndarray:
    """
    Convert a grid state dict to the 4-D RL observation vector.

        grid state: {
            'robot_pos':     (cx, cy),          # integer grid cells
            'human_pos':     (hx, hy),          # integer grid cells
            'human_heading': 'N'|'E'|'S'|'W',
        }

        RL obs: [dx, dy, human_theta, robot_theta]  — float32, metres + radians

    robot_theta is approximated as 0.0 because the grid world does not track
    the robot's heading independently. This is a known limitation.

    Parameters
    ----------
    state : dict — grid state from simple_grid.transition()

    Returns
    -------
    np.ndarray  shape (4,) float32
    """
    rx, ry = state['robot_pos']
    hx, hy = state['human_pos']
    h_theta = _HEADING_TO_RAD.get(state['human_heading'], 0.0)

    dx = (rx - hx) * CELL_SIZE   # metres, robot relative to human
    dy = (ry - hy) * CELL_SIZE

    return np.array([dx, dy, h_theta, 0.0], dtype=np.float32)


# ---------------------------------------------------------------------------
# Node value = paper reward + discounted RL value estimate
# ---------------------------------------------------------------------------

def rl_value(state: dict) -> float:
    """
    Query the trained A2C critic to get V(s) for a grid state.

    Returns 0.0 if the model failed to load (graceful degradation).

    Parameters
    ----------
    state : dict — grid state dict from simple_grid

    Returns
    -------
    float — V(s) from the A2C critic
    """
    if _rl is None:
        return 0.0
    obs = _grid_state_to_obs(state)
    return _rl.evaluate_state(obs)


def _paper_reward(state: dict) -> float:
    """
    Compute the paper reward (Eq 3a) for a grid state.

    Converts grid cell positions to metres, computes Euclidean distance and
    alpha angle, then calls reward.reward(distance_m, alpha_deg).

    Parameters
    ----------
    state : dict — grid state dict

    Returns
    -------
    float — r_d + r_alpha  (paper Eq 3a)
    """
    rx, ry = state['robot_pos']
    hx, hy = state['human_pos']

    # Euclidean distance in metres
    distance = math.sqrt(((rx - hx) * CELL_SIZE) ** 2 +
                         ((ry - hy) * CELL_SIZE) ** 2)

    # Alpha: angle between human heading and human→robot vector [degrees]
    h_theta = _HEADING_TO_RAD.get(state['human_heading'], 0.0)
    bearing  = math.atan2((ry - hy), (rx - hx))   # human→robot bearing (grid units, no CELL_SIZE needed)
    diff_rad = abs(h_theta - bearing)
    if diff_rad > math.pi:
        diff_rad = 2 * math.pi - diff_rad
    alpha_deg = math.degrees(diff_rad)

    return paper_reward(distance, alpha_deg)


def _node_value(state: dict) -> float:
    """Combined node value: immediate paper reward + discounted RL estimate."""
    return _paper_reward(state) + (rl_value(state) / 10.0) * GAMMA


# ---------------------------------------------------------------------------
# MCTS helpers
# ---------------------------------------------------------------------------

def _dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _stay_bool(state, use_stay=False):
    if not use_stay:
        return False
    return _dist(state['robot_pos'], state['human_pos']) > STAY_DISTANCE


def _expand_robot(node):
    p = 1.0 / len(ROBOT_ACTIONS)
    for action in ROBOT_ACTIONS:
        child = Node(copy.deepcopy(node.state), parent=node,
                     action=action, prior=p, node_type='human')
        node.children.append(child)


def _expand_human(node, human_probs=None):
    if human_probs is not None:
        # map LSTM output keys to simple_grid HUMAN_ACTIONS
        probs = {
            'forward': human_probs.get('straight', 0.85),
            'left': human_probs.get('left', 0.075),
            'right': human_probs.get('right', 0.075)
        }
    else:
        probs = human_probabilities(node.state)

    for action, p in probs.items():
        new_state = transition(node.state, node.action, action)
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
            value = _node_value(leaf.state) + rollout(leaf.state)
            _backprop(leaf, value)
            continue
        if leaf.node_type == 'robot':
            _expand_robot(leaf)
        else:
            _expand_human(leaf, human_probs)
        if leaf.children:
            leaf = random.choice(leaf.children)
        value = _node_value(leaf.state) + rollout(leaf.state)
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
    time_budget   : float — wall-clock seconds per plan() call (default: 0.15s,
                            matching the paper's 5 Hz decision rate)
    verbose       : bool  — print root visit counts after each plan
    use_stay      : bool  — allow STAY action when robot is far from human
    """

    def __init__(self, n_simulations=None, time_budget=None,
                 verbose=False, use_stay=False):
        self.n_simulations = n_simulations
        self.time_budget   = time_budget
        self.verbose       = verbose
        self.use_stay      = use_stay
        if n_simulations is None and time_budget is None:
            self.n_simulations = 1000

    def plan(self, state: dict, human_probs: dict = None) -> str:
        """
        Run MCTS from the current state and return the best action.

        Parameters
        ----------
        state       : dict — {'robot_pos': (cx,cy), 'human_pos': (hx,hy),
                             'human_heading': 'N'|'E'|'S'|'W'}
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