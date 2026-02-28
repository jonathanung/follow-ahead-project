# Follow-Ahead Robot: ROS2 Implementation Plan

## Executive Summary

**Motivation**: A "follow-ahead" robot stays *in front of* a walking human (e.g., for filming, guiding, or social interaction), anticipating direction changes rather than simply trailing behind. This is harder than follow-behind because the robot must predict where the human will go.

This document describes the step-by-step plan to port the **Follow_ahead_reaction** project (ROS1, located at `/workspace/src/Follow_ahead_reaction/`) to the **follow-ahead-project** (ROS2, located at `/workspace/src/follow-ahead-project/`). The target project is currently an empty skeleton with `.gitkeep` placeholder files. All code must be written from scratch.

The target ROS2 distribution is determined by the development container (check `echo $ROS_DISTRO` -- the Dockerfile references Humble, but the running environment may be Jazzy). This plan uses generic ROS2 patterns compatible with both.

Extension goals include online LSTM learning, PID-aided motor control, and motor-data synthesis.

---

## Table of Contents

1. [Glossary](#glossary)
2. [Prerequisites](#prerequisites)
3. [Source System Architecture](#1-source-system-architecture-follow_ahead_reaction)
4. [Implementation Steps (Phases 1-7)](#2-implementation-steps)
5. [Extension Goals (A, B, C)](#3-extension-goals)
6. [Implementation Order and Dependencies](#4-implementation-order-and-dependencies)
7. [Known Pain Points and Mitigations](#5-known-pain-points-and-mitigations)
8. [File-by-File Migration Reference](#6-file-by-file-migration-reference)
9. [Configuration Files Migration](#7-configuration-files-migration)
10. [Testing Strategy](#8-testing-strategy)
11. [Summary: What to Build, In What Order](#9-summary-what-to-build-in-what-order)

---

## Glossary

| Term | Full Name | Description |
|------|-----------|-------------|
| **MCTS** | Monte Carlo Tree Search | Planning algorithm that builds a decision tree by random sampling |
| **LSTM** | Long Short-Term Memory | Recurrent neural network for sequence prediction |
| **LSTM-FC** | LSTM + Fully Connected | LSTM layer followed by a fully-connected output layer (class: `LSTMModel2D`) |
| **A2C** | Advantage Actor-Critic | RL algorithm (from stable-baselines3) used to estimate state values |
| **UCB** | Upper Confidence Bound | Selection formula in MCTS that balances exploration vs exploitation |
| **SB3** | Stable-Baselines3 | Python library for RL algorithms (`pip install stable-baselines3`) |
| **EWC** | Elastic Weight Consolidation | Technique to prevent catastrophic forgetting during online learning |
| **MPC** | Model Predictive Control | Optimization-based control over a receding horizon |
| **Nav2** | Navigation2 | ROS2 navigation framework (replaces ROS1 `move_base`) |
| **QoS** | Quality of Service | ROS2 communication policy (reliability, durability, depth) |
| **GIL** | Global Interpreter Lock | Python threading limitation -- only one thread executes Python at a time |
| **BT** | Behavior Tree | Decision structure used by Nav2 (replaces ROS1 finite state machines) |
| **DDS** | Data Distribution Service | Middleware layer for ROS2 communication (e.g., CycloneDDS) |

---

## Prerequisites

### P.1: Verify Model Compatibility (Do This FIRST)

Before any code migration, verify that pre-trained models load in the target environment. These are binary go/no-go checks that could add days if they fail late.

**LSTM model** (`human_prob.pth`, 73 KB):
```python
import torch
from human_prob_dist import LSTMModel2D
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('Follow_ahead_reaction/follow/include/human_prob.pth',
                    map_location=device, weights_only=False)
# Verify: should be LSTMModel2D(input=2, hidden=64, output=3, layers=1)
```

**RL model** (`multiply_rewards_1.zip`, 107 KB):
```python
from stable_baselines3 import A2C
model = A2C.load('Follow_ahead_reaction/follow/include/multiply_rewards_1.zip')
# Verify: model.policy.predict_values() exists and returns a tensor
```

If either fails, retraining is required before proceeding.

### P.2: Update Docker Dependencies

The current `requirements.txt` only has `matplotlib` and `casadi`. Add all required packages:

```
torch
stable-baselines3
gymnasium
scipy
matplotlib
casadi
treelib          # Optional: used in search.py debug visualization
```

Note: PyTorch with CUDA support requires `pip install torch --index-url https://download.pytorch.org/whl/cu121` (or appropriate CUDA version). CPU-only works but is slower.

### P.3: Verify ROS2 Distribution

```bash
echo $ROS_DISTRO  # Expected: humble or jazzy
```

Key differences:
- **Humble** (Ubuntu 22.04, Python 3.10): Uses Gazebo Classic 11, `ros-humble-*` packages
- **Jazzy** (Ubuntu 24.04, Python 3.12): Uses Gz Sim 8 (new Gazebo), `ros-jazzy-*` packages, `ros1_bridge` is NOT available

### P.4: CycloneDDS Configuration

The workspace has DDS configs at `setup/cyclonedds*.xml`. For Docker simulation-only setups, set:
```bash
export CYCLONEDDS_URI=file:///workspace/setup/cyclonedds_lo.xml  # loopback
```

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
                  (extract x,y,                           │   │ Selection│    Action     │ cmd_vel│
Costmap ──────── yaw from ────────────────────────────────►   │ Safety   │              └────────┘
                  quaternions)   RL Model (A2C)            │   │ Checks   │
                                 V(state) / 10 ───────────┘   └──────────┘
                                                              (150ms budget, 5Hz)
```

**Dual-rate architecture**: In the original code, MCTS runs at 5Hz (every 200ms), but `move()` is called on *every* Vicon callback (~100-200Hz), republishing the last best action. The robot receives continuous velocity commands, not just 5Hz bursts.

### 1.3 Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Control frequency | 5 Hz | MCTS decision rate |
| Cmd_vel publish rate | ~100-200 Hz | Republishes best action on every sensor callback |
| Robot velocity | 0.6 m/s | Base linear velocity (`robot_vel`) |
| Robot fast multiplier | 1.5x | Yields 0.9 m/s (`robot_vel_fast_lamda`) |
| Human velocity | 0.6 m/s | Used in MCTS forward model (`human_vel`) |
| Robot turn angle | 45 deg | Discrete turning angle |
| Human turn angle | 10 deg | Expected human turn angle |
| MCTS expansion time | 150 ms | Tree search time budget |
| LSTM history | 15 points (3 sec) | Collected at 5Hz; see note below on 14 vs 15 |
| LSTM hidden size | 64 | Single LSTM layer |
| Safety radius | 0.5 m (r), 0.25 m (a) | Elliptical safety zone, not a simple circle |
| Reaction zone | 0.8 m (r), 0.3 m (a) | Larger zone for planning triggers |
| Discount factor (gamma) | 0.9 | RL discount factor |
| Heading discretization | 20 deg | Human heading quantized to 20-degree bins |
| Sim mode flag | `False` | When `True`, skips obstacle checking |
| Stay threshold | 1.5 m | Robot waits until human is within 1.5m before starting |

### 1.4 Action Spaces

- **Robot**: 6 discrete actions: `{fast_left, fast_right, fast_straight, left, right, straight}`
- **Human**: 3 discrete actions: `{left, straight, right}`

### 1.5 Reward Function

The reward has two components. **Note**: only `r_d` is normalized; the total is NOT bounded to [0,1].

```python
# Distance reward r_d (piecewise, then rescaled to ~[0,1]):
if D < 0.5 or D > 4:     r_d = -1
elif 0.5 <= D <= 1:       r_d = -2 * (1 - D)
elif 1 < D <= 2:          r_d = 2 * (0.5 - abs(D - 1.5))   # Peak at D=1.5
elif 2 < D <= 4:          r_d = -0.5 * (D - 2)
r_d = r_d / 2 + 0.5      # Rescale from [-1,1] to [0,1]

# Orientation reward r_o (unbounded below):
diff = angle between (robot→human vector) and (human heading)
if diff < 25:  r_o = 1.0 * ((25 - diff) / 25)    # Up to +1.0
else:          r_o = -0.25 * diff / 180            # Can be very negative

# Total (NOT normalized to [0,1]):
r = r_d + r_o
```

### 1.6 MCTS Details

**UCB with human probability weighting** (non-standard modification):
```python
UCB = (child.value / child.n + 2.0 * sqrt(log(parent.n) / child.n)) * human_prob
```
Human action probabilities from the LSTM directly multiply the UCB score.

**Node evaluation** (note the `/10` scaling factor):
```python
value = immediate_reward + (RL_value.item() / 10) * gamma
```
The RL value is divided by 10 before combining -- this is a critical tuning parameter.

**Tie-breaking**: `best_child_node()` biases toward "straight" actions when visit counts are equal.

**Stay behavior**: The robot waits (publishes zero velocity) until the human is within 1.5m, controlled by `stay_bool` flag.

---

## 2. Implementation Steps

### Phase 1: ROS2 Package Scaffolding

**Goal**: Create proper ROS2 Python package structure in `follow-ahead-project/`.

#### Step 1.1: Create `follow` ROS2 Package

Create the following structure:

```
follow-ahead-project/follow/
  package.xml            # ROS2 format (see template below)
  setup.py               # Python setuptools (see template below)
  setup.cfg              # Build config (see template below)
  resource/follow        # Ament index marker (empty file)
  follow/
    __init__.py
    main_node.py          # Main decision loop (port of main.py)
    mcts_search.py        # MCTS engine (port of search.py)
    mcts_node.py          # Tree node class (port of nodes.py)
    navi_state.py         # State representation (port of navi_state.py)
    human_prob_dist.py    # LSTM inference wrapper (port of human_prob_dist.py)
    rl_interface.py       # RL value function (port of RL_interface.py)
    nav_env.py            # RL training env (needed by rl_interface at import time)
    util.py               # NN utilities (port of util.py)
  config/
    params.yaml           # All parameters (replaces hardcoded values)
  launch/
    main.launch.py        # ROS2 Python launch file
  models/
    human_prob.pth        # Pre-trained LSTM model weights
    multiply_rewards_1.zip # Pre-trained RL model
  maps/
    cropped.yaml          # Map file
    cropped.pgm           # Map image
```

**`package.xml`** (ROS2 ament_python format):
```xml
<?xml version="1.0"?>
<package format="3">
  <name>follow</name>
  <version>0.0.0</version>
  <description>Follow-ahead robot navigation with MCTS + LSTM + RL</description>
  <maintainer email="user@todo.todo">user</maintainer>
  <license>MIT</license>

  <exec_depend>rclpy</exec_depend>
  <exec_depend>geometry_msgs</exec_depend>
  <exec_depend>nav_msgs</exec_depend>
  <exec_depend>visualization_msgs</exec_depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

**`setup.py`**:
```python
import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'follow'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'models'), glob('models/*')),
        (os.path.join('share', package_name, 'maps'), glob('maps/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'main_node = follow.main_node:main',
        ],
    },
)
```

**`setup.cfg`**:
```ini
[develop]
script_dir=$base/lib/follow
[install]
install_scripts=$base/lib/follow
```

**`config/params.yaml`**:
```yaml
follow:
  ros__parameters:
    # Velocities
    robot_vel: 0.6
    robot_vel_fast_lambda: 1.5
    human_vel: 0.6
    dt: 0.2

    # Angles (degrees)
    robot_angle: 45.0
    human_angle: 10.0
    theta_discretization: 20.0

    # Safety
    safety_r: 0.5
    safety_a: 0.25
    reaction_r: 0.8
    reaction_a: 0.3
    stay_distance: 1.5

    # Planning
    gamma: 0.9
    expansion_time: 0.15
    control_freq: 5.0

    # Model paths (relative to package share)
    lstm_model: "models/human_prob.pth"
    rl_model: "models/multiply_rewards_1.zip"

    # Simulation mode
    sim: false
```

**Build and run**:
```bash
cd /workspace
colcon build --packages-select follow --symlink-install
source install/setup.bash
ros2 launch follow main.launch.py
```

#### Step 1.2: Create `rotate_motor` ROS2 Package (deferred)

Same structure pattern. Contains the camera servo tracking node. Deferred unless using ZED2 + Dynamixel hardware.

**When needed, note these topics/services to migrate**:
- Subscribes: `/zed2/zed_node/obj_det/objects` (ObjectsStamped), `/odom` (Odometry)
- Publishes: `/person_pose` (PoseStamped), `/person_pose_pred_all` (PoseArray), `/pub_glob_coords` (PointStamped)
- Service: `/dynamixel_workbench/dynamixel_command` (DynamixelCommand) -- must use async client in ROS2

**Known bugs in original `track_human.py` to fix during port**:
- Line 101: `if len(self.human_x > 10)` -- compares boolean, should be `if len(self.human_x) > 10`
- Line 107: `y = np.array(self.human_x)` -- should be `y = np.array(self.human_y)`

#### Step 1.3: Create `lstm_fc` Training Package

```
follow-ahead-project/lstm-fc/
  train_lstm.py           # LSTM training script (port of LSTM_classification.py)
  requirements.txt        # torch, matplotlib, scipy
  data/                   # Training data or download scripts
  models/                 # Saved model checkpoints
```

This is a **standalone Python package** (not ROS2), used offline for model training.

**Training data**: The script expects `data_3d_h36m.npz` with structure:
```python
{'S1': {'Walking': array(N, joints, 3)}, 'S5': {...}, 'S6': {...}, ...}
```
This preprocessed file extracts 3D joint positions from the Human3.6M dataset. It can be generated using [VideoPose3D data preparation scripts](https://github.com/facebookresearch/VideoPose3D). Only the hip joint (index 0) 2D coordinates are used.

**Pain points**:
- Human3.6M dataset access may require institutional license
- Training requires CUDA GPU (100K epochs, ~hours on single GPU)
- Model architecture is small (64 hidden, 1 layer) so training is fast per-epoch

---

### Phase 2: Core ROS2 Node Migration

**Goal**: Port `main.py` and all its dependencies to ROS2.

#### Steps 2.1-2.5: Pure Python Modules

These modules have **no ROS imports** and can be ported with minimal changes. However, several require non-trivial fixes (not pure copies).

| Step | Original File | Changes Required | Effort |
|------|--------------|-----------------|--------|
| 2.1 | `navi_state.py` | Remove unused `import matplotlib.pyplot` | Low |
| 2.2 | `human_prob_dist.py` | Fix hardcoded `.cuda()` calls; fix `torch.load()` | Low-Medium |
| 2.3 | `rl_interface.py` | Fix CUDA, fix `nav_env` import coupling, fix default arg | **Medium** |
| 2.4 | `nodes.py` | None; uses params dict, no imports to change | Low |
| 2.5 | `search.py` | Remove hardcoded path in `draw_tree()`, add `treelib` to deps | Low-Medium |

**Step 2.1 detail** (`navi_state.py`): Pure numpy/math. Contains `calculate_new_state(action)` (kinematic forward model), `calculate_reward()` (distance + orientation reward), and alternating robot/human turns. Remove unused `import matplotlib.pyplot as plt`.

**Step 2.2 detail** (`human_prob_dist.py`): Replace hardcoded `.cuda()` with device-agnostic code:
```python
# BEFORE (will crash on CPU-only Docker):
self.model = torch.load(model_dir).cuda()
history = torch.tensor(history).float().cuda()

# AFTER:
self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
self.model = torch.load(model_dir, map_location=self.device, weights_only=False)
history = torch.tensor(history).float().to(self.device)
```

**LSTM input length note**: The training script uses `input_length=14` (14 points as input, 15th as target label), but `main.py` passes 15 points to inference. The `prob_dist.forward()` method normalizes all points relative to the last one (making it `[0,0]`), so the model receives a 15-point sequence where the final point is zero. Verify whether this mismatch affects prediction quality and decide whether to align training/inference to both use 14 or 15 points.

**Step 2.3 detail** (`rl_interface.py`): Three fixes required:

1. **Fix CUDA hardcoding**: Replace `DEVICE = 'cuda'` with device detection
2. **Fix import coupling**: The file imports `from nav_env import Environment` at module level, and the `load_model` method has `env=Environment()` as a default argument -- this instantiates a full gymnasium environment at import time. Fix:
   ```python
   # BEFORE:
   from nav_env import Environment
   def load_model(self, model_path='', policy='a2c', env=Environment()):

   # AFTER:
   def load_model(self, model_path='', policy='a2c', env=None):
       if env is None:
           from nav_env import Environment
           env = Environment()
   ```
3. **Correct API name**: The code uses `model.policy.predict_values(state)` (NOT `evaluate_actions()`). Ensure this API exists in the installed SB3 version.

**Note**: `nav_env.py` must be included in the `follow` package (not just `lstm-fc/`) because `rl_interface.py` imports it at runtime.

**Step 2.5 detail** (`search.py`): The `draw_tree()` debug method contains a hardcoded path `/home/sahar/catkin_ws/src/Follow_ahead_reaction/follow/scripts/tree.txt` -- parameterize or remove. The `treelib` package must be installed (`pip install treelib`). Also note: `search.py` imports `torch` and calls `self.params['RL_model'].evaluate_state()` directly, so it has a runtime dependency on both PyTorch and the RL model.

**Unused files**: `replayBuffer.py` exists in the source but is not actively imported (the import in `nodes.py` is commented out). It contains `state_to_obs()` and `normalize()` functions that may be needed if RL model retraining uses normalized observations. Include for reference but mark as optional.

`util.py` is not imported by any other module in the current system -- likely a leftover from prior RL training. Copy but mark as potentially dead code.

#### Step 2.6: Port `main.py` → `main_node.py` (Heavy ROS migration)

This is the **critical migration step**. All ROS1 patterns must change.

**ROS1 → ROS2 pattern mapping**:

| ROS1 Pattern | ROS2 Replacement |
|---|---|
| `import rospy` | `import rclpy; from rclpy.node import Node` |
| `rospy.init_node('main')` | `class MainNode(Node): super().__init__('main')` |
| `rospy.Subscriber(topic, Type, cb, buff_size=1)` | `self.create_subscription(Type, topic, cb, qos_profile)` |
| `rospy.Publisher(topic, Type, queue_size=1)` | `self.create_publisher(Type, topic, 1)` |
| `rospy.Time.now()` | `self.get_clock().now().to_msg()` |
| `rospy.spin()` | `rclpy.spin(node)` or `executor.spin()` |
| Manual `time.time()` throttling | `self.create_timer(period, callback)` |
| Hardcoded params | `self.declare_parameter()` + YAML config |

**MainNode skeleton** (the minimum viable structure):

```python
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker

from .human_prob_dist import prob_dist
from .rl_interface import RL_model
from .mcts_search import MCTS
from .navi_state import navState

class MainNode(Node):
    def __init__(self):
        super().__init__('follow_ahead_main')

        # Callback groups (required for MultiThreadedExecutor)
        self.sensor_cb_group = MutuallyExclusiveCallbackGroup()
        self.timer_cb_group = MutuallyExclusiveCallbackGroup()

        # Parameters
        self.declare_parameter('robot_vel', 0.6)
        self.declare_parameter('control_freq', 5.0)
        # ... declare all params from params.yaml

        freq = self.get_parameter('control_freq').value

        # QoS profiles
        sensor_qos = QoSProfile(depth=1)
        costmap_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1
        )

        # Subscriptions
        self.create_subscription(
            TransformStamped, 'vicon/helmet/root',
            self.helmet_callback, sensor_qos,
            callback_group=self.sensor_cb_group)
        self.create_subscription(
            TransformStamped, 'vicon/robot/root',
            self.robot_callback, sensor_qos,
            callback_group=self.sensor_cb_group)
        self.create_subscription(
            OccupancyGrid, '/global_costmap/costmap',
            self.costmap_callback, costmap_qos,
            callback_group=self.sensor_cb_group)

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.robot_traj_pub = self.create_publisher(Marker, '/robot_traj', 1)
        self.human_traj_pub = self.create_publisher(Marker, '/human_traj', 1)

        # MCTS decision timer (5Hz)
        self.create_timer(1.0 / freq, self.decision_loop,
                          callback_group=self.timer_cb_group)

        # Optional: fast republish timer for smooth motion (~50Hz)
        self.create_timer(0.02, self.republish_cmd_vel,
                          callback_group=self.timer_cb_group)

        # Load models
        self.human_prob = prob_dist(model_dir='...')
        self.rl_model = RL_model()
        self.rl_model.load_model(model_path='...')

        # State
        self.human_history = []
        self.best_action = None
        self.stay_bool = True  # Wait until human is close enough

    def helmet_callback(self, msg):
        """Update human state from Vicon/sensor."""
        # Extract position and orientation from msg
        # WARNING: Original code passes quaternion as [w,x,y,z] to
        # scipy.Rotation.from_quat() which expects [x,y,z,w].
        # Use the correct ordering for your sensor.
        pass

    def robot_callback(self, msg):
        """Update robot state from Vicon/sensor."""
        pass

    def costmap_callback(self, msg):
        """Cache costmap data for safety checking."""
        pass

    def decision_loop(self):
        """Run MCTS at 5Hz to select best action."""
        if len(self.human_history) < 15:
            return
        if self.stay_bool:
            # Wait until human within 1.5m
            return
        human_prob = self.human_prob.forward(self.human_history)
        # ... run MCTS tree expansion ...
        # ... select best action ...

    def republish_cmd_vel(self):
        """Republish last best action at high rate for smooth motion."""
        if self.best_action is not None:
            self.cmd_vel_pub.publish(self._action_to_twist(self.best_action))

    def _action_to_twist(self, action):
        """Convert discrete action to Twist message."""
        t = Twist()
        # ... map action to linear.x and angular.z ...
        return t

def main(args=None):
    rclpy.init(args=args)
    node = MainNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**Key design decisions in the skeleton**:
- **MultiThreadedExecutor** with 4 threads (required -- SingleThreadedExecutor would block during 150ms MCTS)
- **Separate callback groups** for sensors vs timers (prevents MCTS from blocking sensor updates)
- **Two timers**: slow (5Hz) for MCTS decisions, fast (~50Hz) for smooth velocity republishing (matches original dual-rate architecture)
- **Costmap QoS**: `RELIABLE` + `TRANSIENT_LOCAL` to match Nav2's publisher QoS
- **Stay behavior**: Robot waits until human is within 1.5m before starting MCTS

**Quaternion convention warning**: The original code passes quaternions as `[w,x,y,z]` to `scipy.spatial.transform.Rotation.from_quat()`, which expects `[x,y,z,w]`. This is either a bug or an intentional convention mismatch. During porting, use the correct quaternion ordering and verify the euler angle extraction produces correct yaw values.

**Heading discretization**: The original code quantizes human heading to 20-degree bins:
```python
human_z = (np.abs(human_z) // theta_thr) * theta_thr * np.sign(human_z)
```

#### Step 2.7: Create ROS2 Launch File

```python
# launch/main.launch.py
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    follow_dir = get_package_share_directory('follow')

    return LaunchDescription([
        # Nav2 Lifecycle Manager (REQUIRED -- without this, map_server
        # and costmap nodes will never transition to 'active' state)
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_costmap',
            parameters=[{
                'autostart': True,
                'node_names': ['map_server'],
            }],
        ),

        # Map server (Nav2 version -- lifecycle node)
        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            parameters=[{
                'yaml_filename': os.path.join(follow_dir, 'maps', 'cropped.yaml'),
            }],
        ),

        # Main follow-ahead node
        Node(
            package='follow',
            executable='main_node',
            name='follow_ahead_main',
            parameters=[os.path.join(follow_dir, 'config', 'params.yaml')],
            output='screen',
        ),
    ])
```

**Critical notes**:
- Parameter paths MUST be absolute (use `get_package_share_directory()`)
- Nav2 nodes are lifecycle nodes -- they need a `lifecycle_manager` to activate them
- The `data_files` in `setup.py` must install launch/, config/, models/, and maps/ directories

---

### Phase 3: Sensor Abstraction Layer

**Goal**: Decouple the system from specific sensors (Vicon, ZED2) to support simulation and different hardware.

#### Step 3.1: Define Sensor Interface

The main node expects three inputs via ROS2 topics. Use topic remapping in launch files to swap sensor sources:

| Input | Topic (configurable) | Message Type | Source (Sim) | Source (Real) |
|-------|---------------------|-------------|-------------|--------------|
| Human pose | `vicon/helmet/root` | TransformStamped | Fake publisher | Vicon/ZED2 |
| Robot pose | `vicon/robot/root` | TransformStamped | `/odom` or `/tf` | Vicon/odom |
| Costmap | `/global_costmap/costmap` | OccupancyGrid | Nav2 costmap | Nav2 costmap |

#### Step 3.2: Simulation Environment Setup

The Docker container includes TurtleBot3 + Nav2.

```
follow-ahead-project/env/sim/
  world/                    # Gazebo/Gz Sim world files
  launch/
    sim.launch.py           # Launch Gazebo + TurtleBot3 + Nav2
  config/
    nav2_params.yaml        # Nav2 configuration
  scripts/
    fake_human_publisher.py # Publishes simulated human poses
```

**Gazebo note**: Check `$ROS_DISTRO`. On Jazzy, use `ros_gz_sim` (Gz Sim 8). On Humble, use `gazebo_ros` (Gazebo Classic 11). The TurtleBot3 launch files handle this automatically.

**Pain points**:
- Simulating a walking human in Gazebo requires either a scripted actor or manual teleop
- TurtleBot3 has different kinematics than Robotnik base
- The reward function assumes precise pose estimation; simulation may introduce different noise

---

### Phase 4: LSTM Model Integration

**Goal**: Get LSTM inference working in the ROS2 pipeline.

#### Step 4.1: Verify Model Loading

(Should already be done in Prerequisites P.1)

#### Step 4.2: Integrate with Main Node

Instantiate `prob_dist` in the main node and call `forward(history)` when 15 points are accumulated.

**Pain points**:
- GPU availability in Docker/deployment target (CPU fallback works, <5ms for this small model)
- The model expects 5Hz input; must match timer frequency

#### Step 4.3: Retrain Model (Optional)

If the pre-trained model doesn't transfer well:

1. Obtain `data_3d_h36m.npz` (see Step 1.3 for format details)
2. Run `train_lstm.py` (100K epochs, ~2-4 hours on GPU)
3. Export new `.pth` file (use `torch.save(model.state_dict(), 'model.pth')` for forward compatibility)
4. Verify accuracy against test set

---

### Phase 5: RL Model Integration

**Goal**: Get A2C value function working for MCTS node evaluation.

#### Step 5.1: Verify Model Loading

(Should already be done in Prerequisites P.1)

#### Step 5.2: Observation Space Investigation

**Critical**: The RL training environment (`nav_env.py`) uses 4D observations `[dx, dy, human_ori, agent_ori]` and 16 discrete actions. But MCTS inference (`search.py` lines 99-100) feeds a 3D observation `[dx, dy, d_theta]` to `evaluate_state()`. This dimensional mismatch needs investigation:
- Either the model was trained with a different environment than `nav_env.py`
- Or the model's first layer handles variable-length input (unlikely for MlpPolicy)
- Or this is a bug that produces degraded value estimates

Resolve this before porting. Check what observation the saved model actually expects by inspecting `model.observation_space`.

#### Step 5.3: Retrain RL Model (If Needed)

The training environment is defined in `nav_env.py` (Gymnasium-compatible). If retraining:

1. Port `nav_env.py` to follow-ahead-project
2. Align observation space between `nav_env.py` and `search.py`
3. Train A2C with `stable_baselines3`

**Pain points**:
- Reward function tuning is critical -- the "multiply_rewards" name suggests iterative reward shaping
- Training stability depends on hyperparameters
- Observation normalization (from `replayBuffer.py`: stored mean/std) must be consistent between training and inference

---

### Phase 6: Integration Testing

**Goal**: End-to-end system working in simulation.

#### Step 6.1: Unit Tests

Test each module independently:
- `navi_state.py`: State transitions, reward calculation (verify piecewise reward matches original)
- `human_prob_dist.py`: LSTM inference with known inputs
- `rl_interface.py`: Value estimation with known states
- `mcts_node.py`: Node expansion, safety checks
- `mcts_search.py`: Tree search produces valid actions
- Utility: Create `test_cmd_vel.py` to publish test Twist messages and verify robot motion

#### Step 6.2: Integration Test in Simulation

1. Launch Gazebo with TurtleBot3: `ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py`
2. Launch Nav2 for costmap
3. Publish fake human trajectory
4. Run main node: `ros2 run follow main_node`
5. Verify: robot follows human, avoids obstacles, stays in front
6. Visualize in RViz2 (port `include/rviz.rviz` to RViz2 format for marker topics)

#### Step 6.3: Performance Validation

- MCTS completes within 150ms budget
- Decision loop maintains 5Hz
- LSTM inference < 5ms
- Robot behavior matches original system qualitatively

---

### Phase 7: Real Hardware Integration

**Goal**: Deploy on physical robot.

#### Step 7.1: Sensor Integration

- **Vicon**: Use `vicon_receiver` package (pure ROS2 Vicon client) -- **NOT** `ros1_bridge` (unavailable on Jazzy)
- **ZED Camera**: Use `zed-ros2-wrapper` (official, build from source)
- **Dynamixel**: Use `ros-$ROS_DISTRO-dynamixel-sdk` (available in apt)

#### Step 7.2: Robot-Specific Tuning

- Adjust velocity limits for actual robot
- Tune safety parameters for real environment
- Calibrate coordinate frame transforms (the original 90-degree rotation is hardware-specific)
- Verify quaternion ordering from your specific sensor

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

1. **Ground truth labeling**: After observing 15+1 points, compute the actual human action using the `generate_target()` function from the training script. This function uses a `tanh_power_value = 0.2` parameter -- **port this function to the runtime package**:
   ```python
   def generate_target(seq, input_length=14, tanh_power=0.2):
       p1 = seq[input_length - 1]
       p2 = seq[input_length - 2]
       p0 = seq[-1]  # Future point (ground truth)
       curr_theta = np.arctan2(p1[1]-p2[1], p1[0]-p2[0])
       fut_theta = np.arctan2(p0[1]-p1[1], p0[0]-p1[0])
       alpha = fut_theta - curr_theta
       f_left = max(np.tanh(alpha), 0) ** tanh_power
       f_right = max(np.tanh(-alpha), 0) ** tanh_power
       f_straight = 1 - f_left - f_right
       return [f_left, f_straight, f_right]
   ```

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
   - Optional: Elastic Weight Consolidation (EWC) -- regularizes weights to stay close to pre-trained values

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

**Implementation timing**: PID code can be written during Phase 2 (pure Python, no dependencies), but integration requires `main_node.py` (Step 2.6) and testing requires simulation (Phase 3/6).

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
        self.integral = max(-1.0, min(1.0, self.integral))  # Anti-windup
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        self.prev_error = error
        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative
```

**B. Angular Velocity PID**: Same structure, different gains (Kp=1.0, Ki=0.05, Kd=0.3).

#### Integration with Main Node

```python
# In main_node.py
self.linear_pid = LinearPID()
self.angular_pid = AngularPID()

# Subscribe to odometry for feedback
self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

def move(self):
    desired_v, desired_w = self.action_to_velocity(self.best_action)
    v_cmd = desired_v + self.linear_pid.update(desired_v, self.actual_v, self.dt)
    w_cmd = desired_w + self.angular_pid.update(desired_w, self.actual_w, self.dt)
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
class ServoPID:
    def __init__(self, Kp=0.05, Ki=0.01, Kd=0.02):
        ...
    def update(self, target_pixel, current_pixel, dt):
        # target_pixel = image center (640)
        # current_pixel = human bounding box center
        error = target_pixel - current_pixel
        return pid_output  # Degrees to rotate
```

#### Pain Points

- **Gain tuning**: Requires real-world testing (simulation gains won't transfer directly)
- **Integral windup**: Anti-windup clamp is included but gains need tuning
- **Noise in odometry**: May need low-pass filtering on velocity feedback
- **Interaction with MCTS**: PID operates at higher frequency than MCTS decisions; need to handle action transitions smoothly

---

### Extension C: Motor-Data Synthesis with PID (Exploratory)

**Goal**: Use PID controller data to improve trajectory simulation in MCTS. This extension is exploratory and should be treated as future work.

#### Concept

The MCTS forward model (`navi_state.calculate_new_state()`) uses an idealized kinematic model. PID data reveals the actual velocity response, which can calibrate the forward model:

```python
class CalibratedKinematicModel:
    def __init__(self):
        self.v_scale = 1.0   # Updated from PID data
        self.v_offset = 0.0
        self.w_scale = 1.0
        self.w_offset = 0.0

    def update_from_pid(self, commanded_vs, actual_vs):
        self.v_scale, self.v_offset = np.polyfit(commanded_vs, actual_vs, 1)
```

#### Optional: CasADi Integration

The project includes CasADi as a dependency. It could be used for trajectory optimization or MPC as an alternative/supplement to MCTS, but this is a significant scope expansion beyond the core port.

---

## 4. Implementation Order and Dependencies

```
Prerequisites: P.1 (model verification), P.2 (deps), P.3 (distro check)
                                    ↓
Phase 1: Package Scaffolding
  ├── Step 1.1: follow package structure          ← START HERE
  ├── Step 1.2: rotate_motor package (deferred)
  └── Step 1.3: lstm-fc training package          ← Can parallel with Phase 2

Phase 2: Core Migration
  ├── Steps 2.1-2.5: Pure Python modules          ← Can run in parallel
  │     2.1: navi_state.py     (Low effort)
  │     2.2: human_prob_dist.py (Low-Medium: fix CUDA, torch.load)
  │     2.3: rl_interface.py    (Medium: fix import coupling, CUDA, nav_env)
  │     2.4: nodes.py           (Low effort)
  │     2.5: search.py          (Low-Medium: fix paths, add treelib dep)
  └── Step 2.6: main_node.py   ← Depends on ALL of 2.1-2.5 (critical path)

Phase 3: Sensor Abstraction     ← Start 3.2 in parallel with Phase 2 if possible
  ├── Step 3.1: Interface definition
  └── Step 3.2: Simulation setup

Phase 4: LSTM Integration       ← Depends on 2.2, 2.6
Phase 5: RL Integration         ← Depends on 2.3, 2.6 (parallel with Phase 4)
Phase 6: Integration Testing    ← Depends on ALL above

Phase 7: Hardware Integration   ← Final step

Extensions (after Phase 6 for integration; code can be written earlier):
  A: Online LSTM Learning       ← Depends on Phase 4; needs generate_target() ported
  B: PID Controller             ← Code in Phase 2; integration after 2.6; testing after Phase 3
  C: Motor-Data Synthesis       ← Depends on Extension B (exploratory)
```

### Critical Path

Steps 2.1, 2.2, and 2.3 can proceed in parallel. Step 2.4 depends on 2.1. Step 2.5 depends on 2.2, 2.3, and 2.4. Step 2.6 depends on all of 2.1-2.5. Phases 4 and 5 are independent and can be done in parallel.

```
P.1 → 1.1 → {2.1, 2.2, 2.3} → 2.4 → 2.5 → 2.6 → 3.2 → 6.2 → 7.1
```

Estimated minimum steps on critical path: 9 (including prerequisites).

---

## 5. Known Pain Points and Mitigations

### 5.1 ROS2 Ecosystem Gaps

| Dependency | ROS1 Package | ROS2 Status | Mitigation |
|------------|-------------|-------------|------------|
| Vicon | `vicon_bridge` | Partial ROS2 ports exist | Use `vicon_receiver` (pure ROS2 client) |
| ZED Camera | `zed_wrapper` | Official ROS2 wrapper available | Use `zed-ros2-wrapper` (build from source) |
| Dynamixel | `dynamixel_workbench` | `dynamixel_sdk` ROS2 available | Use `dynamixel_sdk` (apt installable) |
| Navigation | `move_base` | **Nav2** (full replacement) | Migrate to Nav2 APIs |
| Map Server | `map_server` | `nav2_map_server` (lifecycle node) | Direct replacement + lifecycle manager |

**Note**: `ros1_bridge` is NOT available on Jazzy (Ubuntu 24.04) and may not work on Humble without significant effort. Plan for native ROS2 solutions.

### 5.2 Nav2 Migration

Nav2 differs significantly from ROS1 navigation:

- Costmap topic: `/move_base/global_costmap/costmap` → `/global_costmap/costmap` (NOT `/map` -- that's the static map, not the inflated costmap)
- **Lifecycle nodes**: Nav2 nodes must be transitioned `unconfigured → inactive → active` via `nav2_lifecycle_manager` (they don't start publishing automatically)
- Behavior trees: Nav2 uses BT instead of finite state machines
- For our use case, we mainly need the **costmap** -- can use Nav2 costmap node standalone

### 5.3 Model Compatibility

- **PyTorch**: Use `torch.load(path, map_location=device, weights_only=False)`. For forward compatibility, prefer saving/loading `state_dict` instead of full model objects
- **Stable-Baselines3**: The actual API used is `model.policy.predict_values(state)` (NOT `evaluate_actions()`). Check SB3 version compatibility
- **Mitigation**: Include model retraining scripts and document exact versions

### 5.4 Real-Time Performance

- MCTS with 150ms budget is tight; the 150ms computation will block all other callbacks if using `SingleThreadedExecutor`
- **Required**: Use `MultiThreadedExecutor` with separate callback groups for sensors vs timers
- When using `MultiThreadedExecutor`, callbacks may run concurrently -- use `threading.Lock` for shared state (robot pose, human pose, costmap data)

### 5.5 Coordinate Frame Conventions

The original code applies a 90-degree rotation to convert Vicon frames to map frames. This is **hardware-specific** and must be recalibrated for any new setup.

```python
# Original (main.py lines 70-80):
theta_rotation = 90 * (np.pi / 180)
rot_matrix = [[cos(theta), -sin(theta)], [sin(theta), cos(theta)]]
```

**Quaternion convention warning**: The original code passes `[w, x, y, z]` to `scipy.Rotation.from_quat()` which expects `[x, y, z, w]` (scalar-last). This produces incorrect euler angles. Either the original system compensated for this elsewhere, or it is a bug. During porting, use the correct ordering and verify yaw extraction.

### 5.6 Simulation Fidelity

- The MCTS forward model assumes instantaneous velocity changes (no acceleration limits)
- Real robots have momentum and motor response lag
- In simulation, TurtleBot3 has different dynamics than Robotnik
- **Mitigation**: PID controller (Extension B) helps, plus calibrated kinematic model (Extension C)

---

## 6. File-by-File Migration Reference

| Original File | Target File | Changes Required | Effort |
|---|---|---|---|
| `follow/scripts/main.py` | `follow/follow/main_node.py` | Heavy: all ROS APIs, entry point, callback groups, QoS | **High** |
| `follow/scripts/search.py` | `follow/follow/mcts_search.py` | Fix hardcoded path, add `treelib` dep | Low-Medium |
| `follow/scripts/nodes.py` | `follow/follow/mcts_node.py` | None (uses params dict) | Low |
| `follow/scripts/navi_state.py` | `follow/follow/navi_state.py` | Remove unused matplotlib import | Low |
| `follow/scripts/human_prob_dist.py` | `follow/follow/human_prob_dist.py` | Fix `.cuda()` hardcoding, `torch.load` | Low-Medium |
| `follow/scripts/RL_interface.py` | `follow/follow/rl_interface.py` | Fix CUDA, fix `nav_env` import, fix default arg | **Medium** |
| `follow/scripts/nav_env.py` | `follow/follow/nav_env.py` | Copy (needed by rl_interface at import) | Low |
| `follow/scripts/util.py` | `follow/follow/util.py` | None (possibly dead code, copy for reference) | Low |
| `follow/scripts/replayBuffer.py` | `follow/follow/replay_buffer.py` | Currently unused; keep for reference | Low |
| `follow/launch/main.launch` | `follow/launch/main.launch.py` | XML → Python, absolute paths, lifecycle mgr | Medium |
| `follow/package.xml` | `follow/package.xml` | catkin → `ament_python` (export block) | Medium |
| `follow/CMakeLists.txt` | `follow/setup.py` + `setup.cfg` | CMake → setuptools | Medium |
| `rotate_motor/scripts/track_human.py` | `rotate_motor/.../track_human_node.py` | Heavy: ROS + async services + bug fixes | **High** |
| `hmn_traj_prob_dest/LSTM_classification.py` | `lstm-fc/train_lstm.py` | None (standalone) | Low |
| `follow/include/rviz.rviz` | `follow/config/rviz2.rviz` | Convert to RViz2 format | Low |

**Key insight**: Only 2 files out of 15 require heavy ROS migration. However, 3 additional files need non-trivial refactoring (CUDA fixes, import restructuring).

---

## 7. Configuration Files Migration

The original project has multiple config files that must be ported to Nav2 format:

| Original File | Purpose | Nav2 Equivalent |
|---|---|---|
| `costmap_common_params.yaml` | robot_radius (0.3m), inflation_radius (0.55m), obstacle_range (2.5m) | `nav2_params.yaml` → `costmap_2d` section |
| `global_costmap_params.yaml` | Global costmap frame, update rate | `nav2_params.yaml` → `global_costmap` section |
| `local_costmap_params.yaml` | Local costmap size, rolling window | `nav2_params.yaml` → `local_costmap` section |
| `base_local_planner_params.yaml` | Max velocity (0.45m/s), max angular (1.0 rad/s) | `nav2_params.yaml` → `controller_server` section |
| `rviz.rviz` | Visualization config (markers, map, TF) | Convert to RViz2 format |
| `cropped.yaml` / `cropped.pgm` | Main map used in launch | Copy to `maps/` directory |
| `box_map.yaml`, `obstacle.yaml`, etc. | Alternative test maps | Copy for testing |

**Warning**: `croped_map.yaml` (note: typo in filename) contains a hardcoded absolute path `/home/sahar/catkin_ws/src/...` -- fix during migration.

**External dependency**: The original `main.launch` includes `$(find navigation)/launch/main.launch` from a separate `navigation` package. This likely configures `move_base` with the above costmap parameters. The Nav2 equivalent is `nav2_bringup` with a `nav2_params.yaml` that consolidates all these settings.

---

## 8. Testing Strategy

### Unit Tests (per module)

```
tests/
  test_navi_state.py      # State transitions, piecewise reward verification
  test_human_prob.py      # LSTM inference accuracy
  test_rl_interface.py    # Value estimation (check predict_values API)
  test_mcts_node.py       # Node expansion, elliptical safety checks
  test_mcts_search.py     # Action selection, UCB with probabilities
  test_pid.py             # PID convergence (Extension B)
  test_cmd_vel.py         # Publish test Twist messages, verify robot motion
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
4. MCTS decision loop maintains 5Hz (< 200ms per cycle)
5. MCTS tree expansion completes within 150ms budget
6. System runs stably for > 10 minutes

---

## 9. Summary: What to Build, In What Order

| # | Task | Depends On | Deliverable |
|---|------|-----------|-------------|
| 0 | Verify model compatibility (P.1) | Nothing | **Go/no-go gate** |
| 0.5 | Update Docker dependencies (P.2) | Nothing | Working dev environment |
| 1 | ROS2 package scaffolding | #0 | Package structure, setup.py, package.xml |
| 2 | Port pure Python modules (with fixes) | #1 | navi_state, search, nodes, human_prob, rl_interface |
| 3 | Port main_node.py to ROS2 | #2 | Working ROS2 node with subs/pubs/timer/executor |
| 4 | Create launch file + lifecycle manager | #3 | main.launch.py |
| 5 | Create params.yaml + migrate configs | #3 | Externalized configuration |
| 6 | Simulation environment | #1 | Gazebo world + fake human publisher |
| 7 | LSTM integration test | #2, #3 | Confirmed inference in ROS2 pipeline |
| 8 | RL integration test + obs space investigation | #2, #3 | Confirmed value estimation works |
| 9 | Integration test (sim) | #3-8 | End-to-end behavior in Gazebo |
| 10 | PID controller (Ext B) | #3, #6 | Smoother velocity tracking |
| 11 | Online LSTM learning (Ext A) | #7, #9 | Adaptive human prediction |
| 12 | Calibrated dynamics (Ext C) | #10 | Better MCTS forward model (exploratory) |
| 13 | Hardware integration | #9 | Deploy on physical robot |
