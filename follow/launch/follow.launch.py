import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, OpaqueFunction, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


DEFAULT_PARAMS_FILES = {
    "map_file": os.path.join(
        get_package_share_directory('follow'),
        'include',
        'cropped.yaml',
    ),

    "nav2_params": os.path.join(
        get_package_share_directory('follow'), 'params', 'nav2_params.yaml'
    ),

    "vicon_params": os.path.join(                                    # add this
        get_package_share_directory('follow'), 'params', 'vicon_params.yaml'
    ),
}

def launch_setup(context, *args, **kwargs):
    map_file = LaunchConfiguration("map_file").perform(context)
    nav2_params_file = LaunchConfiguration("nav2_params").perform(context)  # add this
    vicon_params_file = LaunchConfiguration("vicon_params").perform(context)  # add this
    map_to_odom_tf = Node(
    package='tf2_ros',
    executable='static_transform_publisher',
    name='map_to_odom',
    arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom']
    )
    vicon_node = Node(
        package='vicon_receiver',
        executable='vicon_client',
        name='vicon_client',
        output='screen',
        parameters=[vicon_params_file]
    )

    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[
            {'use_sim_time': False},
            {'yaml_filename': map_file}
        ]
    )

    lifecycle_manager_node = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager',
        output='screen',
        emulate_tty=True,
        parameters=[
            {'use_sim_time': False},
            {'autostart': True},
            {'node_names': ['map_server']}
        ]
    )

    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('nav2_bringup'), 'launch', 'navigation_launch.py')
        ),

        launch_arguments={
            'params_file': nav2_params_file,
            'use_sim_time': 'false',
        }.items()
    )
    
    return [
        map_to_odom_tf,
        # vicon_node,
        map_server_node,
        lifecycle_manager_node,
        nav2_launch,
        # main_node,
    ]

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'map_file',
            default_value=DEFAULT_PARAMS_FILES['map_file'],
            description='Full path to the map yaml file to load'
        ),
        DeclareLaunchArgument(
            'nav2_params',
            default_value=DEFAULT_PARAMS_FILES['nav2_params'],
            description='Full path to the nav2 params file'
        ),
        DeclareLaunchArgument(
            'vicon_params',
            default_value=DEFAULT_PARAMS_FILES['vicon_params'],
            description='Full path to the vicon params file'
        ),
        OpaqueFunction(function=launch_setup),
    ])
                
