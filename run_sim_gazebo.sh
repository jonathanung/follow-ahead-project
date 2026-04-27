#!/bin/bash
pkill Xvfb; pkill xpra; pkill gzserver; pkill gzclient
rm -f /tmp/.X100-lock /tmp/.X101-lock
sleep 1
source /opt/ros/humble/setup.bash
source /workspaces/ros2_ws/install/setup.bash
Xvfb :101 -screen 0 1280x1024x24 &
export DISPLAY=:101
sleep 2
xpra start :100 --bind-tcp=0.0.0.0:10000 --start-child="gazebo --verbose $(ros2 pkg prefix follow)/share/follow/worlds/follow_world.world -s libgazebo_ros_factory.so" --exit-with-children &
sleep 10
ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 map odom &
ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 odom base_footprint &
sleep 2
export TURTLEBOT3_MODEL=burger
ros2 run gazebo_ros spawn_entity.py -file /opt/ros/humble/share/turtlebot3_gazebo/models/turtlebot3_burger/model.sdf -entity turtlebot -x 0.0 -y 0.0 -z 0.0 &
sleep 5
ros2 run nav2_map_server map_server --ros-args -p yaml_filename:=/workspaces/ros2_ws/install/follow/share/follow/include/cropped.yaml -p use_sim_time:=true &
sleep 3
ros2 lifecycle set /map_server configure
ros2 lifecycle set /map_server activate
sleep 2
ros2 launch nav2_bringup navigation_launch.py use_sim_time:=true params_file:=/workspaces/ros2_ws/install/follow/share/follow/params/nav2_params.yaml &
sleep 5
ros2 run follow fake_vicon &
sleep 2
ros2 run follow main
