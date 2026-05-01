from setuptools import setup

package_name = 'follow'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/params', [
            'params/nav2_params.yaml',
            'params/vicon_params.yaml',
            'params/test_cases.yaml',
            'params/main_params.yaml',
        ]),
        ('share/' + package_name + '/launch', [
            'launch/follow.launch.py',
            'launch/sim.launch.py',
            'launch/robot.launch.py',
        ]),
        ('share/' + package_name + '/rviz', ['rviz/sim.rviz']),
        ('share/' + package_name + '/include', [
            'include/cropped.yaml',
            'include/cropped.pgm',
            'include/my_room.yaml',
            'include/my_room.pgm',
            'include/open.yaml',
            'include/open.pgm',
            'include/human_prob.pth',
            'include/multiply_rewards_1.zip',
        ]),
        ('share/' + package_name + '/worlds', [
            'worlds/follow_world.world',
        ]),        
        ],    
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sahar',
    maintainer_email='sahar@todo.todo',
    description='Follow-ahead robot package (ROS2)',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'main = follow.main:main',
            'fake_vicon = follow.fake_vicon:main',
            'fake_odom = follow.fake_odom:main',
            'vicon_bridge = follow.vicon_bridge:main',
        ],
    },)
o