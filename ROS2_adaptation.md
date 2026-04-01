# ROS2 Algorithmic Adaptation & Justification

This document justifies the parity between the revamped `algo-main` codebase, the legacy ROS1 implementation, and the **IEEE RA-L 2025** paper by Leisiazar et al. It also highlights modern architectural improvements made to facilitate the ROS2 migration.

---

## ⚖️ Algorithmic Parity Justification

The current `algo-main` implementation is a mathematically identical match to the core algorithm described in the research paper and the previous ROS1 repository.

| Feature | Paper / ROS1 Spec | algo-main Implementation | Match? |
| :--- | :--- | :--- | :---: |
| **Action Space** | 6 Discrete Actions (L/R/S × 2 speeds) | `ROBOT_ACTIONS` (6 indexes) | ✅ |
| **Kinematics** | Relative turn model (45° robot / 10° human) | `calculate_new_state` (Relative kinematics) | ✅ |
| **Safety Pruning** | Displaced Circle (r=0.5m, a=0.25m) | `is_safe()` quadratic boundary pruning | ✅ |
| **Reward Function** | Equations 3a, 3b, 3c (r_d + r_alpha) | `reward.py` (Rescaled from [-1, 1] to [0, 1]) | ✅ |
| **Search Logic** | MCTS with UCB + LSTM Priors | `planner.py` (Multiplicative UCB priors) | ✅ |
| **Node Eval** | A2C Critic Value Function V(s) | `RL_model.evaluate_state()` | ✅ |
| **Decision Rate** | 5 Hz (0.2s steps / 0.15s expansion) | `time_budget = 0.15` in MCTSPlanner | ✅ |

### 🛠️ Key Alignment Evidence
- **Relative Turns:** The robot no longer moves on a global grid. It calculates its next `x, y, theta` based on its current heading, exactly as implemented in ROS1's `navi_state.py`.
- **Safety Pruning:** The MCTS expansion now explicitly filters out any robot branch that enters the "keep-out" zone around the human, ensuring the tree only considers safe paths.
- **Value Integration:** The A2C critic is loaded as a singleton and provides the leaf node evaluation, providing the same "look-ahead" benefit as the original RL-integrated search.

---

## 🚀 Improvements over the ROS1 Implementation

While maintaining algorithmic parity, we have significantly hardened the codebase for professional ROS2 deployment.

### 1. Robust Modular Architecture
*   **Original:** Monolithic scripts with high coupling (e.g., `LSTM_classification.py`).
*   **Improvement:** Decoupled packages (`lstm-fc/` for inference, `RL_sim/` for training/reward logic, `follow/scripts/` for planning). This allows for cleaner ROS2 node wrappers and easier dependency management.

### 2. Device-Agnostic Execution (CPU/GPU)
*   **Original:** Hardcoded `device='cuda'` calls caused crashes on non-NVIDIA hardware or Mac/NUC systems.
*   **Improvement:** Full `torch.device` detection. The models now seamlessly load on CPU (typical for robot onboard computers) or GPU (for simulation/training).

### 3. Automated Trajectory Normalization
*   **Original:** Manual normalization logic was scattered across scripts, leading to "silent" prediction errors if inputs weren't perfectly scaled.
*   **Improvement:** The `TrajectoryBuffer` class handles 5 Hz sliding-window recording and **automatic normalization** using the mean/std parameters from training, ensuring the LSTM always sees the correct distribution.

### 4. Centralized State Schema (`FollowState`)
*   **Original:** State was passed around as raw NumPy arrays or dictionaries, making it difficult to debug coordinate frame mismatches.
*   **Improvement:** A dedicated `FollowState` dataclass manages world coordinates, relative bearings, and the `alpha` angle. This acts as a single source of truth for the Planner, the RL model, and future ROS2 messages.

### 5. Integration Guardrails (Unit Testing)
*   **Original:** No unit tests for the core MCTS or LSTM logic. Validation required running the full ROS1 simulation.
*   **Improvement:** Added standalone integration tests (`test_mcts_alignment.py`) that verify the planner can:
    - Prune dangerous actions.
    - Pick straight paths when ideal.
    - Adapt to sharp turns using LSTM priors.

---

## 📌 ROS2 Handoff Note
The current `MCTSPlanner` class in `planner.py` is now a pure-Python engine. To integrate it into ROS2:
1.  Wrap it in a `rclpy.node.Node` subclass.
2.  Map `vicon` subscriber data to `FollowState`.
3.  Publish the resulting `action` string directly to your robot bridge.
