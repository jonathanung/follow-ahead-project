# Follow-Ahead ROS2 Project Status & To-Do List

Based on the `proj-proposal-720.pdf`, recent milestones, and the ongoing integration efforts, here is the current state of the project.

## ✅ What We Have Done (Completed)
These tasks form the algorithmic core of the project and are fully validated offline/natively.

**1. Machine Learning Models Trained (Interim Deliverables)**
- **RL Value Function:** The A2C agent was trained natively in Gymnasium, perfectly replicating paper Equations 3a, 3b, 3c. Models exported successfully to `.zip`, `.pt`, and `.onnx`.
- **LSTM Probability Model:** The `lstm-fc` predictor was trained and refined (`model_final.pt`), with an elegant `HumanActionPredictor` API wrapper created.

**2. MCTS Algorithmic Integration**
- The core MCTS logic (`planner.py`) has been fully re-written to consume both the LSTM and RL models.
- **LSTM Prior Weighting:** The planner cleanly injects the `{"left", "straight", "right"}` LSTM probabilities into the MCTS UCB bounds.
- **RL Node Evaluation:** `rl_value()` is seamlessly wired in, replacing the `-(dist²)` proxy with the true Paper Reward.

**3. Native Algorithm Validation**
- Wrote and passed robust integration tests (`test_integration.py`).
- Completed a native version of the **Sharp Turn Experiment** (`experiment_sharp_turn.py`), proving the algorithm adapts to 180° human turns dynamically *without* needing ROS.

---

## 🟡 What Needs Polishing / More Work
These items have partial implementations but need to be finalized and connected.

**1. MCTS ROS2 Node Integration**
- The `mcts_node.py` ROS wrapper exists, but it needs to be updated to import the newly completed `MCTSPlanner` from `planner.py` and correctly funnel ROS `TransformStamped` topics into the grid state.
- **Safety Module / Collision Checker:** Currently `simple_grid.py` has an empty `OBSTACLES` set. The collision logic from the costmap needs to be wired into the `is_valid()` checks to prune unsafe tree expansions.

**2. Simulation Infrastructure Setup**
- `fake_vicon.py`: Needs to be finalized to publish mock human walking trajectories for the Gazebo sim.

---

## 🔴 What is NOT Done At All
Major deliverables remaining for the final deadline (Apr. 17).

**1. Simulation Experiments**
- Port remaining experiments from the paper to the Gazebo repo.
- Run Gazebo and RViz experiments (Generate visualizations + Document results). Consider building an automation script to speed this up.

**2. PID Hardware Control**
- Write and test the Custom PID Vision Control logic to replace the baseline camera tracking.
- Test in isolation on hardware.

**3. Lab Experiments (Physical Hardware)**
- Full MCTS/AI stack integrated and running on the physical Robot in the lab.
- Real-world evaluations using the Vicon system (Obstacle Avoidance and Sudden Trajectory Tasks).
- Document lab results and statistically compare the new PID camera tracking against the baseline.

**4. Final Deliverables**
- Stretch Goal: Online learning mechanism (optional).
- Write Final Report.
- Create Poster Presentation.

---

## 🚫 Blockers Preventing ROS2 Progress
These issues must be resolved immediately to proceed with integration.

**1. ROS 1 to ROS 2 Bridge Issue**
- **Action Required:** Email/meet with Kurtis to resolve the ROS bridge constraints. Without this, sensor data (like Vicon) cannot cleanly communicate with the ROS2 stack, preventing full-loop simulation and hardware testing.

**2. API Handshake**
- Need to get the repository's simulation working with ROS2, which requires understanding the new API constraints of `planner.py` and `lstm_fc` (e.g., how the `TrajectoryBuffer` expects 5Hz coordinate inputs). A brief sync is required here.
