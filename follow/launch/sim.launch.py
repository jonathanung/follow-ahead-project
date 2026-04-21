import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('follow')

    follow_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg, 'launch', 'follow.launch.py')
        ),
        launch_arguments={'use_sim_time': 'false'}.items()
    )

    fake_vicon_node = Node(
        package='follow',
        executable='fake_vicon',
        name='fake_vicon',
        output='screen',
        parameters=[{
            'human_motion_mode': 'circle',
            'circle_radius': 2.5,
            'circle_angular_speed': 0.2,   # linear speed = r * w = 0.5 m/s
        }]
    )

    main_node = Node(
        package='follow',
        executable='main',
        name='follow_ahead',
        output='screen',
    )

    fake_odom_node = Node(
        package='follow',
        executable='fake_odom',
        name='fake_odom',
        output='screen',
        parameters=[{
            'start_x': 0.0,
            'start_y': 0.0,
            'start_theta': 0.0,
        }]
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', os.path.join(pkg, 'rviz', 'sim.rviz')],
        output='screen',
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'human_motion_mode',
            default_value='straight',
            description='Scripted human motion: straight | circle | rectangle'
        ),
        follow_launch,
        fake_odom_node,
        fake_vicon_node,
        main_node,
        rviz_node,
    ])
