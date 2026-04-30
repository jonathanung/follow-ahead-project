## Follow-Ahead Robot — Quick Start

### Requirements
- ROS2 Humble
- QBot 2e with `qbot_driver`
- VICON system with `vicon_ros2_node`

### Installation
Clone the repo into your ROS2 workspace `src` 
folder and build:
```bash
colcon build
```

### Running on Hardware

> Ensure `ROS_DOMAIN_ID` is set to the same value 
> on both the QBot and your machine.

**On the QBot:**
```bash
# Terminal 1
python3 vicon_ros2_node_new.py

# Terminal 2
ros2 launch qbot_driver bringup.launch.py
```

**On your machine:**
```bash
# Terminal 1
ros2 run follow vicon_bridge

# Terminal 2
ros2 run follow main --ros-args -p sim:=false
```

---

## Project Overview

Replication and ROS2 extension of the MCTS-DRL framework for proactive follow-ahead navigation on a physical QBot 2e. The system integrates Monte Carlo Tree Search with a Deep Reinforcement Learning value function and an LSTM-based human action predictor.

### Key Features & Contributions
- **ROS2 Architecture**: Fully rebuilt and modularized from the original ROS1 codebase.
- **Sim-to-Real Kinematics**: Enforces strict hardware acceleration and velocity limits via `fake_odom.py` ($V_{\max}=0.6$ m/s, $a_{\max}=0.5$ m/s$^2$) to ensure simulations transfer to the QBot.
- **Hardware Integration**: Custom VICON Bridge (`vicon_bridge.py` and `bringup_vicon.launch.py`) for robust, ground-truth map-level EKF localization.
- **Algorithmic Fixes**: Corrected the reference MCTS implementation to prevent duplicate leaf node re-expansion and UCB corruption.
- **Perception**: Replaces default camera drivers with the official RealSense ROS2 wrapper, processing depth pointclouds via `rtabmap_util` for reliable obstacle detection.

---

## Codebase & Workflow Guide

### 1. Training the RL Value Function
The A2C agent evaluates robot-human configurations to determine the "follow-ahead quality" for the MCTS planner. It is trained offline in an obstacle-free simulation.
```bash
cd src/follow-ahead-project/RL_sim
python3 train_a2c.py
```

### 2. Training the LSTM Predictor
The LSTM takes a rolling buffer of human poses and predicts the human's next action (`straight`, `left`, `right`) to bias the tree expansion.
```bash
cd src/follow-ahead-project/lstm-fc
python3 train_final_v3.py
```

### 3. Running Simulations
Test the algorithm in closed-loop simulation with kinematics constraints. RViz will open automatically.
```bash
# From workspace root (~/Desktop/qbot_ws)
source /opt/ros/humble/setup.bash && source install/setup.bash

# Run using ROS2 launch explicitly:
ros2 launch follow sim.launch.py test_case:=circle
```
*(Available test cases: `circle`, `stationary`, `square`, `oscillate`, `zigzag`, `gentle_arc`, `gentle_zigzag`, `approach_and_hold`, `straight`. Kill with `Ctrl+C` to flush the data log to `~/follow_data/`.)*

### 4. Data Visualization
Process the flushed simulation logs to generate quantitative metrics and 2D trajectory plots.
```bash
python3 src/follow-ahead-project/scripts/plot_results.py ~/follow_data/ --summary
```
