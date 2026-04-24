import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('follow')

    # Full nav2 + map + real Vicon stack
    follow_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg, 'launch', 'follow.launch.py')
        ),
        launch_arguments={
            'use_sim_time': 'false',
            'use_vicon':    'true',
        }.items()
    )

    main_params = os.path.join(pkg, 'params', 'main_params.yaml')

    main_node = Node(
        package='follow',
        executable='main',
        name='follow_ahead',
        output='screen',
        parameters=[main_params, {'sim': False}],
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', os.path.join(pkg, 'rviz', 'sim.rviz')],
        output='screen',
    )

    return LaunchDescription([
        follow_launch,
        main_node,
        rviz_node,
    ])
