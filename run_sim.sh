#!/bin/bash
pkill Xvfb; pkill xpra; pkill gzserver; pkill gzclient
rm -f /tmp/.X100-lock /tmp/.X101-lock
sleep 1

source /opt/ros/humble/setup.bash
source /workspaces/ros2_ws/install/setup.bash

Xvfb :101 -screen 0 1280x1024x24 &
export DISPLAY=:101
sleep 2

xpra start :100 \
  --bind-tcp=0.0.0.0:10000 \
  --start-child="rviz2 -d /workspaces/ros2_ws/src/follow-ahead-project/follow/config/follow.rviz" \
  --exit-with-children &
sleep 5

ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 map odom &
ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 odom base_footprint &
sleep 2

# Robot starts at (-1, 0) — directly behind human who starts at (0, 0)
ros2 run follow fake_odom --ros-args -p start_x:=-1.0 -p start_y:=0.0 -p start_theta:=0.0 &
sleep 2

# Human walks slowly in +X direction, ping-pong
ros2 run follow fake_vicon --ros-args \
  -p human_motion_mode:=straight \
  -p straight_line_speed:=0.15 \
  -p straight_line_duration_sec:=20.0 \
  -p straight_line_heading_rad:=0.0 \
  -p straight_line_ping_pong:=true &
sleep 2

ros2 run follow main
