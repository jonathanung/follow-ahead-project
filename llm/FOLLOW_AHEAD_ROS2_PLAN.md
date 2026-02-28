# Follow-Ahead Robot: ROS2 Implementation Plan

## Executive Summary

This document describes the step-by-step plan to port the **Follow_ahead_reaction** project (ROS1) to the **follow-ahead-project** (ROS2 Humble), plus extension goals including online LSTM learning and PID-aided motor control. The original system enables a robot to follow a person from the front while adapting to direction changes, using MCTS planning, LSTM human prediction, and RL value estimation.

---

## 1. Source System Architecture (Follow_ahead_reaction)

### 1.1 System Overview

The original system is a ROS1 catkin project combining three AI/ML techniques:

| Component | Purpose | Key File |
|-----------|---------|----------|
| **MCTS** | Decision-making (action selection) | `follow/scripts/search.py` |
| **LSTM-FC** | Human action prediction (left/straight/right) | `follow/scripts/human_prob_dist.py` |
| **A2C RL** | Node value estimation in MCTS | `follow/scripts/RL_interface.py` |

### 1.2 End-to-End Data Flow

```
SENSORS                          PREDICTION                    PLANNING                  CONTROL
────────                         ──────────                    ────────                  ───────
Vicon Mocap ──┐                  LSTM Model                    MCTS Engine               Twist Cmd
  helmet/root ├──► State ──────► P(left,straight,right) ──┐   ┌──────────┐              ┌────────┐
  robot/root  ┘   Assembly       (15-pt history, 5Hz)     ├──►│ UCB      │──► Best ───► │ V, W   │
                                                          │   │ Selection│    Action     │ cmd_vel│
Costmap ─────────────────────────────────────────────────►│   │ Safety   │              └────────┘
                                 RL Model (A2C)           │   │ Checks   │
                                 V(state) ────────────────┘   └──────────┘
                                                              (150ms budget, 5Hz)
```

### 1.3 Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Control frequency | 5 Hz | Main decision loop rate |
| Robot velocity | 0.6 m/s (0.9 fast) | Base linear velocity |
| Robot turn angle | 45 deg | Discrete turning angle |
| Human turn angle | 10 deg | Expected human turn angle |
| MCTS expansion time | 150 ms | Tree search time budget |
| LSTM history | 15 points (3 sec) | Input sequence length |
| LSTM hidden size | 64 | Single LSTM layer |
| Safety radius | 0.5 m | Minimum distance to human |
| Discount factor | 0.9 | RL gamma |

### 1.4 Action Spaces

- **Robot**: 6 discrete actions: `{fast_left, fast_right, fast_straight, left, right, straight}`
- **Human**: 3 discrete actions: `{left, straight, right}`

### 1.5 Reward Function

```
r_d (distance):  Optimal at 1-2m; penalized <0.5m or >4m
r_o (orientation): Rewards robot staying within 25-degree cone in front of human
Total: r = r_d + r_o, normalized to [0, 1]
```

---

## 2. Implementation Steps

### Phase 1: ROS2 Package Scaffolding

**Goal**: Create proper ROS2 Python package structure in `follow-ahead-project/`.

#### Step 1.1: Create `follow` ROS2 Package

Create the following structure:

```
follow-ahead-project/follow/
  package.xml            # ROS2 format (ament_python buildtool)
  setup.py               # Python setuptools entry points
  setup.cfg              # Build config
  resource/follow        # Ament index marker (empty file)
  follow/
    __init__.py
    main_node.py          # Main decision loop (port of main.py)
    mcts_search.py        # MCTS engine (port of search.py)
    mcts_node.py          # Tree node class (port of nodes.py)
    navi_state.py         # State representation (port of navi_state.py)
    human_prob_dist.py    # LSTM inference wrapper (port of human_prob_dist.py)
    rl_interface.py       # RL value function (port of RL_interface.py)
    util.py               # NN utilities (port of util.py)
  config/
    params.yaml           # All parameters (replaces hardcoded values)
    costmap_common.yaml   # Costmap parameters
  launch/
    main.launch.py        # ROS2 Python launch file
  include/
    human_prob.pth        # Pre-trained LSTM model weights
    multiply_rewards_1.zip # Pre-trained RL model
    cropped.yaml          # Map file
```

**Key changes from ROS1**:
- `CMakeLists.txt` is **removed** (Python-only package uses setuptools)
- `package.xml`: `<buildtool_depend>ament_python</buildtool_depend>` replaces catkin
- `setup.py` with `entry_points` for node executables
- Launch files are Python (`.launch.py`) not XML (`.launch`)

#### Step 1.2: Create `rotate_motor` ROS2 Package (if needed)

Same structure as above, containing the camera servo tracking node. This may be deferred if using a different sensor setup.

#### Step 1.3: Create `lstm_fc` Training Package

```
follow-ahead-project/lstm-fc/
  train_lstm.py           # LSTM training script (port of LSTM_classification.py)
  requirements.txt        # torch, matplotlib, scipy
  data/                   # Training data or download scripts
  models/                 # Saved model checkpoints
```

This is a **standalone Python package** (not ROS2), used offline for model training.

**Pain points**:
- Human3.6M dataset access may require institutional license
- Training requires CUDA GPU (100K epochs, ~hours on single GPU)
- Model architecture is small (64 hidden, 1 layer) so training is fast per-epoch

---

### Phase 2: Core ROS2 Node Migration

**Goal**: Port `main.py` and all its dependencies to ROS2.

#### Step 2.1: Port `navi_state.py` (No ROS dependency)

This module is **pure Python** with no ROS imports. Changes needed:
- None functionally; just copy and organize
- Consider adding type hints for clarity

This is the state representation with:
- `calculate_new_state(action)`: Kinematic forward model
- `calculate_reward()`: Distance + orientation reward
- Alternating robot/human turns

**Risk**: None. This is pure math/numpy.

#### Step 2.2: Port `human_prob_dist.py` (No ROS dependency)

This LSTM inference wrapper is **pure PyTorch**. Changes needed:
- None functionally; just copy
- Ensure model path is configurable (not hardcoded)

Architecture: `LSTMModel2D` (input=2, hidden=64, output=3, softmax)

**Risk**: PyTorch version compatibility. The `.pth` file was saved with a specific torch version. May need to re-save or retrain if torch versions differ significantly.

#### Step 2.3: Port `rl_interface.py` (No ROS dependency)

The RL model loader uses **stable-baselines3**. Changes needed:
- None functionally
- Verify stable-baselines3 compatibility with current Python/torch versions
- The model (`multiply_rewards_1.zip`) was trained with a specific SB3 version

**Risk**: SB3 API changes between versions. The `A2C.load()` and `model.policy.evaluate_actions()` API must match. If SB3 version differs, may need to retrain or use compatibility shims.

#### Step 2.4: Port `nodes.py` (Minimal ROS dependency)

The MCTS node class has one ROS-adjacent dependency: it accesses costmap data passed via `params`. Changes needed:
- No ROS imports to change
- Safety checking (`is_safe_to_pass`, `close_to_human`, `any_obs`) uses raw costmap data (numpy array), not ROS messages directly

**Risk**: None. The costmap data is pre-extracted in the main node.

#### Step 2.5: Port `search.py` (No ROS dependency)

The MCTS engine is **pure Python**. Changes needed:
- None functionally
- It uses `time.time()` for expansion timing (not ROS time)

**Risk**: None. This is the core algorithm with no ROS coupling.

#### Step 2.6: Port `main.py` → `main_node.py` (Heavy ROS migration)

This is the **critical migration step**. All ROS1 patterns must change:

| ROS1 Pattern | ROS2 Replacement |
|---|---|
| `import rospy` | `import rclpy; from rclpy.node import Node` |
| `rospy.init_node('main')` | `class MainNode(Node): super().__init__('main')` |
| `rospy.Subscriber(topic, Type, cb, buff_size=1)` | `self.create_subscription(Type, topic, cb, qos_profile)` |
| `rospy.Publisher(topic, Type, queue_size=1)` | `self.create_publisher(Type, topic, 1)` |
| `rospy.Time.now()` | `self.get_clock().now().to_msg()` |
| `rospy.spin()` | `rclpy.spin(node)` |
| Manual `time.time()` throttling | `self.create_timer(1.0/freq, self.decision_callback)` |
| Hardcoded params | `self.declare_parameter()` + YAML config |

**Specific changes**:

1. **Class structure**: Inherit from `rclpy.node.Node`
2. **Subscriptions**:
   - `/move_base/global_costmap/costmap` (OccupancyGrid) → same topic, QoS depth=1
   - `vicon/helmet/root` (TransformStamped) → same or adapt to new sensor
   - `vicon/robot/root` (TransformStamped) → same or adapt to new sensor
3. **Publishers**:
   - `/robot/robotnik_base_control/cmd_vel` (Twist) → topic name may change for new robot
   - Marker publishers for visualization remain the same
4. **Timer**: Replace manual `time.time()` frequency control with `create_timer(0.2, self.decision_loop)` for 5 Hz
5. **Parameters**: Move all hardcoded values to `config/params.yaml`
6. **Entry point**: Add `main()` function for ROS2 lifecycle

**Pain points**:
- Topic names will differ based on actual robot hardware
- QoS profiles need tuning (especially for costmap - may need `RELIABLE` + `TRANSIENT_LOCAL`)
- Vicon bridge must have a ROS2 version (or use `ros1_bridge`)
- Navigation2 replaces `move_base` (different costmap topics)

#### Step 2.7: Create ROS2 Launch File

Convert XML launch to Python:

```python
# launch/main.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    return LaunchDescription([
        # Map server (Nav2 version)
        Node(package='nav2_map_server', executable='map_server', ...),
        # Main follow node
        Node(package='follow', executable='main_node',
             parameters=['config/params.yaml']),
        # Vicon bridge (if ROS2 version exists)
        # ...
    ])
```

**Pain points**:
- Nav2 launch patterns differ significantly from ROS1 navigation stack
- Need to verify all dependent packages have ROS2 Humble versions

---

### Phase 3: Sensor Abstraction Layer

**Goal**: Decouple the system from specific sensors (Vicon, ZED2) to support simulation and different hardware.

#### Step 3.1: Define Sensor Interface

Create an abstract interface so the main node doesn't care whether input comes from Vicon, simulation, or another tracking system:

```python
# Expected input: Two poses at regular intervals
# - Human pose: (x, y, yaw) in map frame
# - Robot pose: (x, y, yaw) in map frame
# - Costmap: OccupancyGrid
```

For **simulation** (TurtleBot3 in Gazebo):
- Robot pose from `/odom` or `/tf` (map→base_link)
- Human pose from simulated human (scripted waypoints or teleop)
- Costmap from Nav2

For **real hardware**:
- Robot pose from odometry/SLAM
- Human pose from ZED2 object detection + motor tracking (existing approach)
- Costmap from Nav2

#### Step 3.2: Simulation Environment Setup

The Docker container already includes TurtleBot3 + Nav2. Set up:

```
follow-ahead-project/env/sim/
  world/                    # Gazebo world files
  launch/
    sim.launch.py           # Launch Gazebo + TurtleBot3 + Nav2
  config/
    nav2_params.yaml        # Nav2 configuration
  scripts/
    fake_human_publisher.py # Publishes simulated human poses
```

**Pain points**:
- Simulating a walking human in Gazebo requires either a scripted actor or manual teleop
- The reward function assumes precise pose estimation; simulation introduces different noise characteristics
- TurtleBot3 has different kinematics than Robotnik base

---

### Phase 4: LSTM Model Integration

**Goal**: Get LSTM inference working in the ROS2 pipeline.

#### Step 4.1: Verify Model Loading

Test that the existing `human_prob.pth` loads correctly in the Docker environment:

```python
import torch
model = torch.load('human_prob.pth', map_location='cpu')
# Verify: model architecture matches LSTMModel2D
```

If the model fails to load (torch version mismatch), retrain from `LSTM_classification.py`.

#### Step 4.2: Integrate with Main Node

The LSTM inference is already decoupled - just instantiate `prob_dist` in the main node and call `forward(history)` when 15 points are accumulated.

**Pain points**:
- GPU availability in Docker/deployment target
- If running on CPU, inference latency increases (but LSTM is small, should be <5ms even on CPU)
- The model expects 5Hz input; must match timer frequency

#### Step 4.3: Retrain Model (Optional)

If the pre-trained model doesn't transfer well, retrain using `LSTM_classification.py`:

1. Download Human3.6M walking data
2. Run training (100K epochs, ~2-4 hours on GPU)
3. Export new `.pth` file
4. Verify accuracy matches original (check against test set)

---

### Phase 5: RL Model Integration

**Goal**: Get A2C value function working for MCTS node evaluation.

#### Step 5.1: Verify Model Loading

```python
from stable_baselines3 import A2C
model = A2C.load('multiply_rewards_1.zip')
```

If SB3 version mismatch, need to retrain.

#### Step 5.2: Retrain RL Model (If Needed)

The training environment is defined in `nav_env.py` (Gymnasium-compatible). If retraining:

1. Port `nav_env.py` to follow-ahead-project
2. Train A2C with `stable_baselines3`:
   ```python
   from stable_baselines3 import A2C
   env = Environment()
   model = A2C('MlpPolicy', env, verbose=1)
   model.learn(total_timesteps=500_000)
   model.save('model')
   ```

**Pain points**:
- Reward function tuning is critical - the multiply_rewards name suggests reward shaping was iterative
- Training stability depends on hyperparameters
- Observation normalization (stored mean/std) must be consistent between training and inference

---

### Phase 6: Integration Testing

**Goal**: End-to-end system working in simulation.

#### Step 6.1: Unit Tests

Test each module independently:
- `navi_state.py`: State transitions, reward calculation
- `human_prob_dist.py`: LSTM inference with known inputs
- `rl_interface.py`: Value estimation with known states
- `mcts_node.py`: Node expansion, safety checks
- `mcts_search.py`: Tree search produces valid actions

#### Step 6.2: Integration Test in Simulation

1. Launch Gazebo with TurtleBot3
2. Launch Nav2 for costmap
3. Publish fake human trajectory
4. Run main node
5. Verify: robot follows human, avoids obstacles, stays in front

#### Step 6.3: Performance Validation

- MCTS completes within 150ms budget
- Control loop maintains 5Hz
- LSTM inference < 5ms
- Robot behavior matches original system qualitatively

---

### Phase 7: Real Hardware Integration

**Goal**: Deploy on physical robot.

#### Step 7.1: Sensor Integration

- Configure Vicon bridge for ROS2 (or alternative tracking)
- Configure ZED2 wrapper for ROS2 (if using camera tracking)
- Configure motor control (Dynamixel ROS2 driver)

#### Step 7.2: Robot-Specific Tuning

- Adjust velocity limits for actual robot
- Tune safety parameters for real environment
- Calibrate coordinate frame transforms

---

## 3. Extension Goals

### Extension A: Online LSTM Learning

**Goal**: Update the LSTM model in real-time based on observed human behavior.

#### Architecture

```
Main Thread (ROS2 callbacks, MCTS, control @ 5Hz)
     │
     ├──► Inference: human_prob.forward(history) → P(left,straight,right)
     │
     └──► Data Collection: (trajectory, actual_action) pairs
              │
              ▼
     Background Thread (Training @ ~0.1Hz)
         ├── Collect batch (e.g., 64 samples)
         ├── Compute loss (MSE between predicted and actual)
         ├── Backprop + optimizer step
         ├── Optional: experience replay buffer
         └── Atomic model swap (double-buffering)
```

#### Implementation Steps

1. **Ground truth labeling**: After observing 15+1 points, compute the actual human action using the same `generate_target()` function from training. Compare the future point (point 16) against current heading to get actual left/straight/right label.

2. **Online training loop** (background thread):
   ```python
   class OnlineLSTMTrainer:
       def __init__(self, model, buffer_size=1000, batch_size=64, lr=0.001):
           self.model = copy.deepcopy(model)  # Shadow copy for training
           self.buffer = deque(maxlen=buffer_size)
           self.optimizer = Adam(self.model.parameters(), lr=lr)
           self.criterion = nn.MSELoss()

       def add_sample(self, trajectory, actual_label):
           self.buffer.append((trajectory, actual_label))

       def train_step(self):
           if len(self.buffer) < self.batch_size:
               return
           batch = random.sample(self.buffer, self.batch_size)
           # ... standard training loop
           # When done, signal main thread to swap model
   ```

3. **Model swapping**: Use `threading.Lock` or atomic reference swap to update the inference model without blocking the control loop.

4. **Catastrophic forgetting prevention**:
   - Keep experience replay buffer with old data (from Human3.6M)
   - Mix old and new samples (e.g., 50/50)
   - Use small learning rate (0.001 vs 0.01 for offline training)
   - Optional: Elastic Weight Consolidation (EWC)

#### Pain Points

- **Labeling delay**: Need to wait for future point to compute actual action (200ms at 5Hz)
- **Distribution shift**: Online data may be very different from Human3.6M (different person, speed, environment)
- **Stability**: Rapid updates could degrade predictions temporarily
- **Computational cost**: Training while inferring on same GPU requires careful memory management
- **Evaluation**: Need online metrics to detect if model is improving or degrading

#### Mitigation Strategies

- Start with frozen model, only enable online learning after sufficient data collection (~500 samples)
- Implement rollback: if prediction accuracy drops below threshold, revert to last known good model
- Log all predictions vs actuals for offline analysis
- Use learning rate warmup and decay

---

### Extension B: PID Controller for Motor Control

**Goal**: Replace open-loop velocity commands with closed-loop PID control for smoother, more accurate tracking.

#### Current System (Open-Loop)

```
MCTS Action (discrete) → Fixed velocity/angle → Twist publish → Robot moves
No feedback on actual velocity achieved
```

#### Proposed System (Closed-Loop)

```
MCTS Action → Desired (v, w) → PID Controller → Adjusted Twist → Robot moves
                                     ▲                                │
                                     └── Odometry feedback ◄──────────┘
```

#### Implementation: Two PID Controllers

**A. Linear Velocity PID**:
```python
class LinearPID:
    def __init__(self, Kp=0.5, Ki=0.1, Kd=0.2):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, desired_v, actual_v, dt):
        error = desired_v - actual_v
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative
```

**B. Angular Velocity PID**:
```python
class AngularPID:
    def __init__(self, Kp=1.0, Ki=0.05, Kd=0.3):
        # Same structure, different gains
```

#### Integration with Main Node

```python
# In main_node.py
self.linear_pid = LinearPID()
self.angular_pid = AngularPID()

# Subscribe to odometry for feedback
self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

def move(self):
    desired_v, desired_w = self.action_to_velocity(self.best_action)

    # PID correction
    v_cmd = desired_v + self.linear_pid.update(desired_v, self.actual_v, self.dt)
    w_cmd = desired_w + self.angular_pid.update(desired_w, self.actual_w, self.dt)

    # Clamp to safe limits
    v_cmd = np.clip(v_cmd, 0.0, 1.0)
    w_cmd = np.clip(w_cmd, -1.5, 1.5)

    twist = Twist()
    twist.linear.x = v_cmd
    twist.angular.z = w_cmd
    self.cmd_vel_pub.publish(twist)
```

#### PID for Camera Servo (rotate_motor)

Replace the threshold-based centering with smooth PID:

```python
# Current: if error > threshold, rotate by fixed amount
# Proposed:
class ServoPID:
    def __init__(self, Kp=0.05, Ki=0.01, Kd=0.02):
        ...

    def update(self, target_pixel, current_pixel, dt):
        # target_pixel = image center (640)
        # current_pixel = human bounding box center
        error = target_pixel - current_pixel
        return pid_output  # Degrees to rotate
```

Benefits: smoother tracking, less oscillation, faster convergence.

#### Pain Points

- **Gain tuning**: Requires real-world testing (simulation gains won't transfer directly)
- **Integral windup**: Need anti-windup mechanism (clamp integral term)
- **Actuator saturation**: Robot motors have physical limits
- **Noise in odometry**: May need low-pass filtering on velocity feedback
- **Interaction with MCTS**: PID operates at higher frequency than MCTS decisions; need to handle action transitions smoothly

---

### Extension C: Motor-Data Synthesis with PID

**Goal**: Use PID controller data to improve trajectory simulation in MCTS.

#### Concept

The MCTS forward model (`navi_state.calculate_new_state()`) currently uses an idealized kinematic model:
```python
new_x = x + v * dt * cos(theta)  # Assumes perfect execution
```

In reality, the robot doesn't achieve exact commanded velocities. PID data reveals the **actual** velocity response, which can be used to:

1. **Calibrate forward model**: Learn actual velocity as function of commanded velocity
   ```python
   actual_v = f(commanded_v)  # Linear model from PID data
   ```

2. **Add noise model**: Learn typical velocity variance from PID history
   ```python
   actual_v ~ N(commanded_v * scale, sigma)  # Gaussian noise
   ```

3. **Improve MCTS accuracy**: Use calibrated model for tree expansion, leading to better action selection

#### Implementation

```python
class CalibratedKinematicModel:
    def __init__(self):
        self.v_scale = 1.0   # Updated from PID data
        self.v_offset = 0.0
        self.w_scale = 1.0
        self.w_offset = 0.0

    def update_from_pid(self, commanded_vs, actual_vs):
        # Linear regression: actual = scale * commanded + offset
        self.v_scale, self.v_offset = np.polyfit(commanded_vs, actual_vs, 1)

    def predict_next_state(self, state, action, params):
        v_cmd, w_cmd = action_to_velocity(action, params)
        v_actual = self.v_scale * v_cmd + self.v_offset
        w_actual = self.w_scale * w_cmd + self.w_offset
        # Use calibrated velocities in kinematic model
        ...
```

#### Integration with CasADi

The project already includes CasADi as a dependency (in `requirements.txt`). CasADi can be used for:
- **Trajectory optimization**: Compute optimal trajectories using calibrated dynamics
- **Model Predictive Control (MPC)**: Replace or augment MCTS with optimization-based planning
- **Constraint handling**: Encode safety constraints as optimization constraints

```python
import casadi as ca

# Define dynamics symbolically
x = ca.MX.sym('x', 3)  # [x, y, theta]
u = ca.MX.sym('u', 2)  # [v, w]

# Calibrated dynamics
x_next = ca.vertcat(
    x[0] + (v_scale * u[0] + v_offset) * dt * ca.cos(x[2]),
    x[1] + (v_scale * u[0] + v_offset) * dt * ca.sin(x[2]),
    x[2] + (w_scale * u[1] + w_offset) * dt
)
```

---

## 4. Implementation Order and Dependencies

```
Phase 1: Package Scaffolding
  ├── Step 1.1: follow package structure          ← START HERE
  ├── Step 1.2: rotate_motor package (deferred)
  └── Step 1.3: lstm-fc training package

Phase 2: Core Migration (can partially parallelize)
  ├── Step 2.1: navi_state.py     ← No deps, copy directly
  ├── Step 2.2: human_prob_dist.py ← No deps, copy directly
  ├── Step 2.3: rl_interface.py    ← No deps, copy + verify SB3
  ├── Step 2.4: nodes.py          ← Depends on 2.1
  ├── Step 2.5: search.py         ← Depends on 2.2, 2.3, 2.4
  └── Step 2.6: main_node.py      ← Depends on ALL above (critical path)

Phase 3: Sensor Abstraction
  ├── Step 3.1: Interface definition
  └── Step 3.2: Simulation setup  ← Needed for testing

Phase 4: LSTM Integration          ← Depends on 2.2, 2.6
Phase 5: RL Integration            ← Depends on 2.3, 2.6
Phase 6: Integration Testing       ← Depends on ALL above

Phase 7: Hardware Integration       ← Final step

Extensions (can start after Phase 6):
  A: Online LSTM Learning           ← Depends on Phase 4
  B: PID Controller                 ← Independent, can start in Phase 2
  C: Motor-Data Synthesis           ← Depends on Extension B
```

### Critical Path

```
1.1 → 2.1 → 2.4 → 2.5 → 2.6 → 3.2 → 6.2 → 7.1
         ↗ 2.2 ↗
         ↗ 2.3 ↗
```

Estimated minimum steps on critical path: 8 major steps.

---

## 5. Known Pain Points and Mitigations

### 5.1 ROS2 Ecosystem Gaps

| Dependency | ROS1 Package | ROS2 Status | Mitigation |
|------------|-------------|-------------|------------|
| Vicon | `vicon_bridge` | Partial ROS2 ports exist | Use `ros1_bridge` or find ROS2 fork |
| ZED Camera | `zed_wrapper` | Official ROS2 wrapper available | Use `zed-ros2-wrapper` |
| Dynamixel | `dynamixel_workbench` | `dynamixel_sdk` ROS2 available | Use `dynamixel_sdk` |
| Navigation | `move_base` | **Nav2** (full replacement) | Migrate to Nav2 APIs |
| Map Server | `map_server` | `nav2_map_server` | Direct replacement |

### 5.2 Nav2 Migration

The biggest ROS ecosystem change. Nav2 differs significantly from ROS1 navigation:

- Costmap topic: `/global_costmap/costmap` → may be `/map` or custom
- Lifecycle nodes: Nav2 uses managed lifecycle (configure → activate)
- Behavior trees: Nav2 uses BT instead of FSM
- For our use case, we mainly need the **costmap** — can use Nav2 costmap node standalone

### 5.3 Model Compatibility

- **PyTorch**: The LSTM model (`.pth`) may not load across major torch versions. If torch 1.x model is loaded in torch 2.x, may need `weights_only=False` flag or re-export
- **Stable-Baselines3**: The A2C model (`.zip`) format changed between SB3 1.x and 2.x. Check version compatibility
- **Mitigation**: Include model retraining scripts and document exact versions

### 5.4 Real-Time Performance

- MCTS with 150ms budget is tight; Python GIL may cause issues with concurrent callbacks
- **Mitigation**: Use `MultiThreadedExecutor` in ROS2 or `rclpy.executors.SingleThreadedExecutor` with careful callback design
- Consider moving MCTS to a separate thread with `threading.Thread`

### 5.5 Coordinate Frame Conventions

The original code applies a 90-degree rotation to convert Vicon frames to map frames. This is **hardware-specific** and must be recalibrated for any new setup.

```python
# Original (main.py lines 70-80):
theta_rotation = 90 * (np.pi / 180)
rot_matrix = [[cos(theta), -sin(theta)], [sin(theta), cos(theta)]]
```

### 5.6 Simulation Fidelity

- The MCTS forward model assumes instantaneous velocity changes (no acceleration limits)
- Real robots have momentum and motor response lag
- In simulation, TurtleBot3 has different dynamics than Robotnik
- **Mitigation**: PID controller (Extension B) helps, plus calibrated kinematic model (Extension C)

---

## 6. File-by-File Migration Reference

| Original File | Target File | ROS Changes | Effort |
|---|---|---|---|
| `follow/scripts/main.py` | `follow/follow/main_node.py` | Heavy (all ROS APIs) | High |
| `follow/scripts/search.py` | `follow/follow/mcts_search.py` | None (pure Python) | Low |
| `follow/scripts/nodes.py` | `follow/follow/mcts_node.py` | None (uses params dict) | Low |
| `follow/scripts/navi_state.py` | `follow/follow/navi_state.py` | None (pure numpy) | Low |
| `follow/scripts/human_prob_dist.py` | `follow/follow/human_prob_dist.py` | None (pure torch) | Low |
| `follow/scripts/RL_interface.py` | `follow/follow/rl_interface.py` | None (pure SB3) | Low |
| `follow/scripts/nav_env.py` | `lstm-fc/nav_env.py` | None (pure gymnasium) | Low |
| `follow/scripts/util.py` | `follow/follow/util.py` | None (pure torch) | Low |
| `follow/launch/main.launch` | `follow/launch/main.launch.py` | XML → Python | Medium |
| `follow/package.xml` | `follow/package.xml` | catkin → ament | Medium |
| `follow/CMakeLists.txt` | `follow/setup.py` | CMake → setuptools | Medium |
| `rotate_motor/scripts/track_human.py` | `rotate_motor/.../track_human_node.py` | Heavy (ROS + services) | High |
| `hmn_traj_prob_dest/LSTM_classification.py` | `lstm-fc/train_lstm.py` | None (standalone) | Low |

**Key insight**: Only 2 files out of 13 require heavy ROS migration. Most of the codebase is pure Python/PyTorch.

---

## 7. Testing Strategy

### Unit Tests (per module)

```
tests/
  test_navi_state.py      # State transitions, rewards
  test_human_prob.py      # LSTM inference accuracy
  test_rl_interface.py    # Value estimation
  test_mcts_node.py       # Node expansion, safety
  test_mcts_search.py     # Action selection quality
  test_pid.py             # PID convergence (Extension B)
```

### Integration Tests

```
tests/
  test_main_node.py       # Full pipeline with mocked sensors
  test_sim_launch.py      # Launch in Gazebo, verify behavior
```

### Acceptance Criteria

1. Robot maintains 1-2m distance from human (distance reward > 0.5)
2. Robot stays within 25-degree frontal cone (orientation reward > 0)
3. No collisions (safety check always passes)
4. Control loop maintains 5Hz (< 200ms per cycle)
5. MCTS completes within 150ms budget
6. System runs stably for > 10 minutes

---

## 8. Summary: What to Build, In What Order

| # | Task | Depends On | Deliverable |
|---|------|-----------|-------------|
| 1 | ROS2 package scaffolding | Nothing | Package structure, setup.py, package.xml |
| 2 | Copy pure Python modules | #1 | navi_state, search, nodes, util, human_prob, rl_interface |
| 3 | Port main_node.py to ROS2 | #2 | Working ROS2 node with subs/pubs/timer |
| 4 | Create launch file | #3 | main.launch.py |
| 5 | Create params.yaml | #3 | Externalized configuration |
| 6 | Simulation environment | #1 | Gazebo world + fake human publisher |
| 7 | Verify LSTM model loading | #2 | Confirmed inference works |
| 8 | Verify RL model loading | #2 | Confirmed value estimation works |
| 9 | Integration test (sim) | #3-8 | End-to-end behavior in Gazebo |
| 10 | PID controller (Ext B) | #3 | Smoother velocity tracking |
| 11 | Online LSTM learning (Ext A) | #7, #9 | Adaptive human prediction |
| 12 | Calibrated dynamics (Ext C) | #10 | Better MCTS forward model |
| 13 | Hardware integration | #9 | Deploy on physical robot |
