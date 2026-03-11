from setuptools import setup, find_packages

package_name = 'follow'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'mcts_node = follow.scripts.mcts_node:main',
        ],
    },
)
