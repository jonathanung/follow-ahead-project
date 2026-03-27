# LSTM-FC API Reference

Human action probability predictor for the follow-ahead MCTS planner.

## Overview

```
Vicon Mocap (100+ Hz)
       │
       ▼
 TrajectoryBuffer.push(x, y)       ← called at 5 Hz
       │
       ▼
 HumanActionPredictor.predict()    ← returns {"left": p, "straight": p, "right": p}
       │
       ▼
 MCTS UCB weighting                ← UCB *= human_prob[action]
       │
       ▼
 Robot action selection
```

## Installation

```python
pip install -e /path/to/lstm-fc
```

Or just add the directory to `PYTHONPATH`.

## Quick Start

```python
from lstm_fc import HumanActionPredictor, TrajectoryBuffer, INPUT_LENGTH, DECISION_HZ

# Load model (once at startup)
predictor = HumanActionPredictor("path/to/model_final.pt")

# Create trajectory buffer
buf = TrajectoryBuffer(length=INPUT_LENGTH)  # 14 points

# In your 5 Hz control loop:
def on_human_position(x: float, y: float):
    buf.push(x, y)

    if buf.ready:
        probs = predictor.predict(buf.get())
        # probs = {"left": 0.12, "straight": 0.76, "right": 0.12}
        run_mcts(human_probs=probs)
```

---

## Core Classes

### `HumanActionPredictor`

Loads a trained LSTM-FC checkpoint and returns action probabilities.

```python
class HumanActionPredictor:
    ACTION_NAMES = ("left", "straight", "right")

    def __init__(self, model_path: str | Path, device: str = "auto")
    def predict(self, history: ndarray[N, 2]) -> dict[str, float]
    def predict_batch(self, histories: ndarray[B, N, 2]) -> ndarray[B, 3]
    def predict_raw(self, histories_normalized: ndarray[B, N, 2]) -> ndarray[B, 3]
```

**`__init__(model_path, device="auto")`**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_path` | `str \| Path` | Path to `.pt` checkpoint |
| `device` | `str` | `"auto"` (default), `"cuda"`, or `"cpu"` |

The checkpoint must contain `model_state_dict`. If it also contains
`model_config`, that config is used; otherwise defaults apply.

**`predict(history) → dict`**

Single-trajectory prediction. Handles normalization automatically.

| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| `history` | `ndarray` or `list` | `(N, 2)` | 2D positions `[x, y]` in meters. `N` = 14 (INPUT_LENGTH). |

Returns `{"left": float, "straight": float, "right": float}` summing to ~1.0.

Normalization: the last point is subtracted from all points, so the model
sees displacement patterns relative to the current position.

**`predict_batch(histories) → ndarray`**

Batch prediction with automatic normalization.

| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| `histories` | `ndarray` | `(B, N, 2)` | B trajectories |

Returns `ndarray` of shape `(B, 3)` — `[p_left, p_straight, p_right]` per row.

**`predict_raw(histories_normalized) → ndarray`**

Batch prediction **without** normalization. Use only if you already subtracted
the last point yourself.

---

### `TrajectoryBuffer`

Fixed-length sliding window for accumulating human positions at 5 Hz.

```python
class TrajectoryBuffer:
    def __init__(self, length: int = 14)
    def push(self, x: float, y: float) -> None
    def clear(self) -> None
    @property ready -> bool
    @property count -> int
    def get(self) -> ndarray[length, 2]
```

**`push(x, y)`** — Append a new position. Oldest point is discarded when full.

**`ready`** — `True` once `length` points have been pushed.

**`get()`** — Returns `(length, 2)` float32 array. Raises `RuntimeError` if not ready.

**`clear()`** — Empty the buffer (call on tracking loss or reinitialization).

---

## Shared Constants (`lstm_fc.actions`)

These constants are shared between the LSTM predictor and MCTS planner to
keep action definitions in sync.

```python
from lstm_fc.actions import (
    HumanAction,         # IntEnum: LEFT=0, STRAIGHT=1, RIGHT=2
    RobotAction,         # IntEnum: FAST_LEFT=0 .. STRAIGHT=5
    HUMAN_ACTION_NAMES,  # ("left", "straight", "right")
    ROBOT_ACTION_NAMES,  # ("fast_left", "fast_right", ..., "straight")
    DECISION_HZ,         # 5 Hz control frequency
    DECISION_DT,         # 0.2 s per step
    INPUT_LENGTH,        # 14 trajectory points
    HUMAN_VEL,           # 0.6 m/s
    ROBOT_VEL,           # 0.6 m/s
    FAST_LAMBDA,         # 1.5x speed multiplier
    TURN_ANGLE_DEG,      # 45° per turn action
)
```

### `HumanAction` (IntEnum)

| Value | Name | LSTM Output Index |
|-------|------|-------------------|
| 0 | `LEFT` | `output[0]` |
| 1 | `STRAIGHT` | `output[1]` |
| 2 | `RIGHT` | `output[2]` |

### `RobotAction` (IntEnum)

| Value | Name | Velocity | Turn |
|-------|------|----------|------|
| 0 | `FAST_LEFT` | 0.9 m/s | +45° |
| 1 | `FAST_RIGHT` | 0.9 m/s | -45° |
| 2 | `FAST_STRAIGHT` | 0.9 m/s | 0° |
| 3 | `LEFT` | 0.6 m/s | +45° |
| 4 | `RIGHT` | 0.6 m/s | -45° |
| 5 | `STRAIGHT` | 0.6 m/s | 0° |

---

## MCTS Integration Guide

### How MCTS Should Use the Predictor

The MCTS tree alternates between robot turns and human turns.
Human action probabilities from the LSTM weight the UCB score:

```python
from lstm_fc import HumanActionPredictor, TrajectoryBuffer, INPUT_LENGTH
from lstm_fc.actions import HUMAN_ACTION_NAMES

predictor = HumanActionPredictor("model.pt")
buf = TrajectoryBuffer(length=INPUT_LENGTH)


def mcts_decision_step(human_x, human_y, state):
    """Called at 5 Hz from the control loop."""
    buf.push(human_x, human_y)

    if not buf.ready:
        return default_action()

    human_probs = predictor.predict(buf.get())
    # human_probs = {"left": 0.1, "straight": 0.8, "right": 0.1}

    best_action = expand_tree(
        root_state=state,
        human_probs=human_probs,
        time_budget=0.15,  # 150 ms
    )
    return best_action
```

### UCB Formula

```python
def compute_ucb(child, parent, human_probs, c_param=2.0):
    exploitation = child.value / child.n
    exploration = c_param * sqrt(log(parent.n) / child.n)
    ucb = exploitation + exploration

    # Weight by human action probability on human-turn nodes
    if child.is_human_turn:
        ucb *= human_probs[child.action_name]

    return ucb
```

### Node Evaluation

Each MCTS leaf node should be evaluated as:

```python
node.value = immediate_reward + (rl_value / 10.0) * gamma
```

Where:
- `immediate_reward` = distance reward + orientation reward (from forward model)
- `rl_value` = V(state) from A2C value function
- `gamma` = 0.9 (discount factor)
- `/10.0` scaling prevents RL value from dominating

### Forward Model

The MCTS needs a forward model to simulate state transitions:

```python
from lstm_fc.actions import DECISION_DT, HUMAN_VEL, ROBOT_VEL, FAST_LAMBDA, TURN_ANGLE_DEG
import numpy as np

def forward_step(state, action, is_robot):
    """Simulate one agent taking one action.

    Args:
        state: [agent_x, agent_y, agent_yaw_rad]
        action: action name string
        is_robot: True for robot, False for human

    Returns:
        New [x, y, yaw] after the step.
    """
    x, y, yaw = state
    vel = ROBOT_VEL if is_robot else HUMAN_VEL
    turn_rad = np.radians(TURN_ANGLE_DEG)

    if "fast" in action:
        vel *= FAST_LAMBDA

    if "left" in action:
        yaw += turn_rad
    elif "right" in action:
        yaw -= turn_rad

    x += vel * DECISION_DT * np.cos(yaw)
    y += vel * DECISION_DT * np.sin(yaw)

    return np.array([x, y, yaw])
```

---

## Data Flow Summary

```
                        ┌──────────────────────────┐
                        │     Vicon / Sensor        │
                        │  human_x, human_y @ 5 Hz  │
                        └────────────┬─────────────┘
                                     │
                                     ▼
                        ┌──────────────────────────┐
                        │    TrajectoryBuffer       │
                        │  push(x, y)               │
                        │  14-point sliding window   │
                        └────────────┬─────────────┘
                                     │ .get() → (14, 2)
                                     ▼
                        ┌──────────────────────────┐
                        │  HumanActionPredictor     │
                        │  .predict(history)        │
                        │                           │
                        │  Normalize (subtract last)│
                        │  LSTM → Dropout → FC      │
                        │  Softmax → [p_l, p_s, p_r]│
                        └────────────┬─────────────┘
                                     │ {"left": .1, "straight": .8, "right": .1}
                                     ▼
┌─────────────┐         ┌──────────────────────────┐
│  RL Value   │────────▶│      MCTS Planner        │
│  V(state)   │         │                           │
└─────────────┘         │  UCB *= human_prob[action]│
                        │  150ms time budget         │
┌─────────────┐         │  Forward model simulation  │
│  Costmap    │────────▶│  Safety zone checking      │
│  (optional) │         └────────────┬─────────────┘
└─────────────┘                      │ best_action ∈ RobotAction
                                     ▼
                        ┌──────────────────────────┐
                        │   Robot Motor Control     │
                        │   cmd_vel @ 50+ Hz        │
                        └──────────────────────────┘
```

## Model Details

| Property | Value |
|----------|-------|
| Architecture | LSTM → Dropout → FC → Softmax |
| Input | `(batch, 14, 2)` — 14 points of `[x, y]` |
| Output | `(batch, 3)` — `[p_left, p_straight, p_right]` |
| Loss | MSE on soft probability labels |
| Normalization | Subtract last input point (automatic in `predict()`) |
| Labels | `tanh^0.2` soft encoding of angular heading change |
| Training data | Walking trajectories at 5 Hz with 50x augmentation |
