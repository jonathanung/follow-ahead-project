# Algo-Main: Python Component Documentation

This document explains each Python (`.py`) file in the `algo-main` directory, grouped by their role in the **Follow-Ahead Robot Navigation** system.

---

## 1. MCTS Planning Engine (`follow/scripts/`)

These files implement the Monte Carlo Tree Search (MCTS) algorithm and its interface with the robot system.

| File | Description |
| :--- | :--- |
| [`planner.py`](file:///Users/ankushsingh/Desktop/CMPT%20720/Follow_ahead_reaction-master/algo-main/follow/scripts/planner.py) | **The MCTS Engine.** Orchestrates the search loop (Select, Expand, Evaluate, Backprop). It blends the RL value function with human prediction priors using relative kinematics. |
| [`node.py`](file:///Users/ankushsingh/Desktop/CMPT%20720/Follow_ahead_reaction-master/algo-main/follow/scripts/node.py) | Defines the MCTS `Node` class. Manages visit counts, values, priors, and UCB scores. |
| [`mcts_node.py`](file:///Users/ankushsingh/Desktop/CMPT%20720/Follow_ahead_reaction-master/algo-main/follow/scripts/mcts_node.py) | **ROS2 Wrapper.** Subscribes to Odometry and Vicon topics, runs the `MCTSPlanner` at 5Hz, and publishes `cmd_vel` (Twist) commands. |
| [`simple_grid.py`](file:///Users/ankushsingh/Desktop/CMPT%20720/Follow_ahead_reaction-master/algo-main/follow/scripts/simple_grid.py) | Base 2D grid world. Defines the discrete state space for initial development and testing. |
| [`rollout.py`](file:///Users/ankushsingh/Desktop/CMPT%20720/Follow_ahead_reaction-master/algo-main/follow/scripts/rollout.py) | Implements greedy rollout simulation strategies used to evaluate nodes at the leaf level. |
| [`test_integration.py`](file:///Users/ankushsingh/Desktop/CMPT%20720/Follow_ahead_reaction-master/algo-main/follow/scripts/test_integration.py) | **Integration Tests.** Validates state conversions, reward math, and planner-model connectivity. |
| [`experiment_sharp_turn.py`](file:///Users/ankushsingh/Desktop/CMPT%20720/Follow_ahead_reaction-master/algo-main/follow/scripts/experiment_sharp_turn.py) | Standalone simulation script representing the "Sharp Turn" scenario for terminal-based validation. |
| [`test_mcts_6actions.py`](file:///Users/ankushsingh/Desktop/CMPT%20720/Follow_ahead_reaction-master/algo-main/follow/scripts/test_mcts_6actions.py) | Testing suite for the six discrete 2D actions (`left`, `right`, `straight`, `fast_left`, `fast_right`, `fast_straight`). |
| [`test_mcts_alignment.py`](file:///Users/ankushsingh/Desktop/CMPT%20720/Follow_ahead_reaction-master/algo-main/follow/scripts/test_mcts_alignment.py) | Ensures the tree search logic strictly adheres to the mathematical requirements defined in the RA-L paper. |

---

## 2. Human Action Prediction (`lstm-fc/`)

Predicts the human's next intent (left, right, or straight) based on movement history.

### Core Package (`lstm_fc/`)
| File | Role | Details |
| :--- | :--- | :--- |
| [`inference.py`](file:///Users/ankushsingh/Desktop/CMPT%20720/Follow_ahead_reaction-master/algo-main/lstm-fc/lstm_fc/inference.py) | **Primary API** | Contains `HumanActionPredictor` (model inference) and `TrajectoryBuffer` (sliding window of human positions). |
| [`model.py`](file:///Users/ankushsingh/Desktop/CMPT%20720/Follow_ahead_reaction-master/algo-main/lstm-fc/lstm_fc/model.py) | **Architecture** | Defines the LSTM-FC neural network architecture. |
| [`config.py`](file:///Users/ankushsingh/Desktop/CMPT%20720/Follow_ahead_reaction-master/algo-main/lstm-fc/lstm_fc/config.py) | **Parameters** | Houses model and training hyperparameters. |
| [`actions.py`](file:///Users/ankushsingh/Desktop/CMPT%20720/Follow_ahead_reaction-master/algo-main/lstm-fc/lstm_fc/actions.py) | **Definitions** | Defines human intents and constants like `INPUT_LENGTH=14`. |
| [`data/dataset.py`](file:///Users/ankushsingh/Desktop/CMPT%20720/Follow_ahead_reaction-master/algo-main/lstm-fc/lstm_fc/data/dataset.py) | **Data Loader** | PyTorch dataset implementation for training trajectories. |
| [`training/train.py`](file:///Users/ankushsingh/Desktop/CMPT%20720/Follow_ahead_reaction-master/algo-main/lstm-fc/lstm_fc/training/train.py) | **Trainer** | Contains the model fitting and optimization logic. |

---

## 3. RL Value Function & Sim (`RL_sim/`)

Provides the trained critic used to score MCTS nodes.

| File | Category | Role |
| :--- | :--- | :--- |
| [`reward.py`](file:///Users/ankushsingh/Desktop/CMPT%20720/Follow_ahead_reaction-master/algo-main/RL_sim/reward.py) | **Math Core** | Implements the paper's reward equations (Eq 3a, 3b, 3c) for distance and orientation. |
| [`state.py`](file:///Users/ankushsingh/Desktop/CMPT%20720/Follow_ahead_reaction-master/algo-main/RL_sim/state.py) | **State Model** | The `FollowState` dataclass—unified state schema for RL training and MCTS search. |
| [`nav_env.py`](file:///Users/ankushsingh/Desktop/CMPT%20720/Follow_ahead_reaction-master/algo-main/RL_sim/nav_env.py) | **Simulation** | A 2D Gymnasium environment for training the Advantage Actor-Critic (A2C) model. |
| [`RL_interface.py`](file:///Users/ankushsingh/Desktop/CMPT%20720/Follow_ahead_reaction-master/algo-main/RL_sim/RL_interface.py) | **Wrapper** | Bridges Stable-Baselines3 models with the MCTS planner to provide scalar $V(s)$ estimates. |
| [`train_a2c.py`](file:///Users/ankushsingh/Desktop/CMPT%20720/Follow_ahead_reaction-master/algo-main/RL_sim/train_a2c.py) | **Training** | Script for training the A2C policy under the navigation environment. |
| [`export_value_fn.py`](file:///Users/ankushsingh/Desktop/CMPT%20720/Follow_ahead_reaction-master/algo-main/RL_sim/export_value_fn.py) | **Export** | Extracts the trained value network for standalone use in search. |
| [`util.py`](file:///Users/ankushsingh/Desktop/CMPT%20720/Follow_ahead_reaction-master/algo-main/RL_sim/util.py) | **Utilities** | Geometry helpers (e.g., angle wrapping, distance calculations). |

---

## Performance Notes & Integration Logic

When the 5Hz control loop is active:
1.  **Input Sensing**: `mcts_node.py` receives human and robot poses.
2.  **Intent Prediction**: Human coordinates are pushed to the `TrajectoryBuffer`. The `HumanActionPredictor` returns probabilities (`straight`, `left`, `right`).
3.  **MCTS Tree Branching**: The `MCTSPlanner` uses these probabilities as tree priors. Nodes are expanded using relative kinematics.
4.  **Value Backpropagation**: `RL_interface` queries the A2C critic to evaluate node goodness, which is then propagated back to the root.
5.  **Action Selection**: The robot executes the action with the highest visit count.
