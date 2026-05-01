import os
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    pkg = get_package_share_directory('follow')

    test_case = LaunchConfiguration('test_case').perform(context)

    cases_file = os.path.join(pkg, 'params', 'test_cases.yaml')
    with open(cases_file, 'r') as f:
        all_cases = yaml.safe_load(f)['test_cases']

    if test_case not in all_cases:
        available = ', '.join(all_cases.keys())
        raise ValueError(f"Unknown test_case '{test_case}'. Available: {available}")

    raw = dict(all_cases[test_case])
    raw.pop('description', None)

    # Split robot start params (→ fake_odom) from human motion params (→ fake_vicon).
    # Override args (robot_start_x / _y / _theta) let run_experiment.py inject
    # per-run perturbations without touching test_cases.yaml.
    yaml_x     = raw.pop('robot_start_x',     0.0)
    yaml_y     = raw.pop('robot_start_y',      0.0)
    yaml_theta = raw.pop('robot_start_theta',  0.0)

    ox = LaunchConfiguration('robot_start_x').perform(context)
    oy = LaunchConfiguration('robot_start_y').perform(context)
    ot = LaunchConfiguration('robot_start_theta').perform(context)

    robot_params = {
        'start_x':     float(ox) if ox else yaml_x,
        'start_y':     float(oy) if oy else yaml_y,
        'start_theta': float(ot) if ot else yaml_theta,
    }
    human_params = raw  # everything that remains

    map_name = LaunchConfiguration('map').perform(context)
    map_options = {
        'my_room': os.path.join(pkg, 'include', 'my_room.yaml'),
        'cropped': os.path.join(pkg, 'include', 'cropped.yaml'),
        'open':    os.path.join(pkg, 'include', 'open.yaml'),
    }
    if map_name not in map_options:
        raise ValueError(f"Unknown map '{map_name}'. Options: {list(map_options.keys())}")
    map_file = map_options[map_name]

    follow_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg, 'launch', 'follow.launch.py')
        ),
        launch_arguments={'use_sim_time': 'false', 'map_file': map_file}.items()
    )

    fake_vicon_node = Node(
        package='follow',
        executable='fake_vicon',
        name='fake_vicon',
        output='screen',
        parameters=[human_params],
    )

    main_params = os.path.join(pkg, 'params', 'main_params.yaml')

    main_node = Node(
        package='follow',
        executable='main',
        name='follow_ahead',
        output='screen',
        parameters=[main_params, {'sim': True}],
        additional_env={'FOLLOW_TEST_CASE': os.environ.get('FOLLOW_TEST_CASE', test_case)},
    )

    odom_params = dict(robot_params)
    if map_name == 'my_room':
        odom_params['map_yaml_path'] = map_file

    fake_odom_node = Node(
        package='follow',
        executable='fake_odom',
        name='fake_odom',
        output='screen',
        parameters=[odom_params],
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', os.path.join(pkg, 'rviz', 'sim.rviz')],
        output='screen',
    )

    return [follow_launch, fake_odom_node, fake_vicon_node, main_node, rviz_node]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'test_case',
            default_value='circle',
            description=(
                'Human motion scenario to run. '
                'Options: straight | circle | stationary | square | oscillate | zigzag | '
                'gentle_arc | approach_and_hold | gentle_zigzag'
            )
        ),
        DeclareLaunchArgument(
            'map', default_value='cropped',
            description='Map to use: cropped (default) | my_room (lab map) | open (no obstacles)'
        ),
        DeclareLaunchArgument('robot_start_x',     default_value='',
                              description='Override robot start x (m); empty = use test_cases.yaml'),
        DeclareLaunchArgument('robot_start_y',     default_value='',
                              description='Override robot start y (m); empty = use test_cases.yaml'),
        DeclareLaunchArgument('robot_start_theta', default_value='',
                              description='Override robot start theta (rad); empty = use test_cases.yaml'),
        OpaqueFunction(function=launch_setup),
    ])
