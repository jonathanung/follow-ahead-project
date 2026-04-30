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
