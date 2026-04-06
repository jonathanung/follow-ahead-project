# Algorithmic Integration Handoff

This document traces the origin of the core algorithmic files, explains how the Machine Learning models integrate with the MCTS planner, and provides guidance for the final ROS2 integration testing strategy.

---

## 1. Branch Origins & Merged Files

The current codebase is a synthesis of three separate implementation branches.

### From branch: `mcts`
This branch provided the base Python implementation of the MCTS tree search and simple grid world.
- `follow/scripts/node.py` — The core MCTS tree node (handles UCB, child selection, expansion).
- `follow/scripts/rollout.py` — Greedy rollout simulation for leaf nodes.
- `follow/scripts/simple_grid.py` — The mathematical 7x7 grid world, basic kinematics (`move_in_dir`, `transition`), and action spaces.

### From branch: `lstm-train`
This branch provided the offline training pipeline and the active Python package for human action prediction.
- `lstm-fc/` — The complete package. 
- **Critical Files:** 
  - `lstm-fc/outputs/final_v3/model_final.pt` (The trained weights)
  - `lstm-fc/lstm_fc/inference.py` (The highly-refined `HumanActionPredictor` and `TrajectoryBuffer` API)

### From branch: `add-RL-sim-dependencies` (Current Branch)
This branch introduced the completed continuous Reinforcement Learning model and the final MCTS integration logic.
- `RL_sim/` — Defines the Gymnasium environment and training scripts.
- **Critical Files:**
  - `RL_sim/models/a2c_follow_ahead.zip` (The trained A2C Value function acting as the critic).
  - `RL_sim/reward.py` (The corrected Reward Function strictly complying with Paper Equations 3a, 3b, and 3c).
  - `follow/scripts/RL_interface.py` (The wrapper that makes the SB3 stable_baseline model compatible with the planner).
  - `follow/scripts/planner.py` (The final integrated MCTS Engine).

---

## 2. Component Integration Architecture

The **`MCTSPlanner`** (in `follow/scripts/planner.py`) marries the MCTS search logic with the two ML models.

### How the Models are Loaded
The `RL_model` is instantiated and loaded globally *once* upon importing `planner.py`. This prevents overhead when querying the value function. The LSTM model is instantiated outside the planner and passed in during the 5Hz control loop.

### How the LSTM Injects Priors
At each 5 Hz tick, a ROS node (e.g., `main_node.py` or `mcts_node.py`) should collect the human's root coordinates `(x, y)` and push them to the `TrajectoryBuffer`.
When the buffer is full (14 points), it predicts a dictionary:
`human_probs = {'left': 0.1, 'straight': 0.8, 'right': 0.1}`

The ROS node calls `MCTSPlanner.plan(state, human_probs=human_probs)`. 
Inside `planner.py`, `_expand_human()` takes these probabilities and assigns them directly as the mathematically rigorous **`prior`** parameters on the newly expanded Human Nodes in the tree. The tree search dynamically scales the UCB exploration bounds by these probabilities.

### How the RL Value Evaluates Nodes
When the tree algorithm wants to score a leaf node (`_node_value()`), it computes:
`Value = Paper_Reward(Immediate_State) + Discounted_RL_Estimation(State)`
The `rl_value(state)` function internally translates the discrete grid state into a dense 4D observation vector (`[dx, dy, human_theta, robot_theta]`) and feeds it to the `a2c_follow_ahead` critic network.

---

## 3. How Integration Was Tested

### Native Validation (Without ROS)
To guarantee that the math and algorithm loops are bug-free before dealing with difficult ROS2 middleware or Gazebo constraints, we wrote:
- **`follow/scripts/test_integration.py`**: A 7/7 test suite evaluating isolated value functions, node limits, correct conversions, and mock/real LSTM inference tests. *(You can run this via pytest or standard python)*.
- **`follow/scripts/experiment_sharp_turn.py`**: A fully native, terminal-rendered version of the "Sharp Turn Experiment" (Exp 2 in the paper). It creates fake trajectories and loops MCTS + LSTM over time, proving the robot successfully wraps around a human turning 180° on the grid.

### Why this is important:
If the robot fails to follow the human in ROS2/Gazebo, **you can confidently rule out algorithmic/ML bugs.** The problem will strictly reside in costmap configurations, robot max velocity limits (Turtlebot vs Robotnik), or tf tree/odometry issues.

---

## 4. Better Ways to Test It (Next Steps)

While native terminal scripts prove algorithmic correctness, they lack geometric/spatial realism. The final validation *must* occur in ROS2 simulation before real-world deployment. 

**Untouched Simulation & Visualization Tasks:**
1. **Gazebo Simulation Loop (`fake_vicon.py`):** The MCTS currently produces discrete grid commands (`N`, `S`, `E`, `W`). The ROS wrapper must generate native `<geometry_msgs/Twist>` messages from these discrete actions, and funnel simulated Vicon data back into the `planner.py` state dictionaries. Writing `fake_vicon.py` to drive a modeled human in Gazebo will allow end-to-end continuous validation.
2. **RViz Tree Branch Debugging:** MCTS is famously hard to debug visually because it explores thousands of bad imaginary paths. You should port the original `rviz.rviz` configuration to output `<visualization_msgs/MarkerArray>`. Rendering the MCTS `children` branches as faint lines projecting out of the robot's current pose will let you instantly see if the RL value function is exploring unsafe territory correctly before making its final 5Hz decision.
