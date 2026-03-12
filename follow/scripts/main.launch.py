import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node


SCRIPTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'scripts'
)


def generate_launch_description():
    gazebo = ExecuteProcess(
        cmd=[
            'ros2', 'launch', 'turtlebot3_gazebo', 'empty_world.launch.py'
        ],
        output='screen',
    )

    mcts_node = TimerAction(
        period=5.0,
        actions=[
            ExecuteProcess(
                cmd=['python3', os.path.join(SCRIPTS_DIR, 'mcts_node.py')],
                output='screen',
            )
        ],
    )

    fake_human = TimerAction(
        period=5.0,
        actions=[
            ExecuteProcess(
                cmd=['python3', os.path.join(SCRIPTS_DIR, 'fake_human_publisher.py')],
                output='screen',
            )
        ],
    )

    return LaunchDescription([
        gazebo,
        mcts_node,
        fake_human,
    ])