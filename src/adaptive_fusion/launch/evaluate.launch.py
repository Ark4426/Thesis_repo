"""
evaluate.launch.py
Wraps simulation.launch.py (single source of truth for the sim stack) and
adds the evaluation-only pieces:
  - a static alpha publisher for the fixed-weight baselines B1/B2/B3
    (disabled with use_fixed_alpha:=false when evaluate.py drives alpha
    from the trained PPO policy — B5)
  - the robot_localization EKF for baseline B4 (ekf:=true), which fuses
    /fusion/visual_pose_cov and /pose per config/ekf.yaml and publishes
    /odometry/filtered
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess,
                            IncludeLaunchDescription, TimerAction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('adaptive_fusion')

    world_arg    = DeclareLaunchArgument('world',    default_value='w1_static')
    seed_arg     = DeclareLaunchArgument('seed',     default_value='0')
    alpha_arg    = DeclareLaunchArgument('alpha',    default_value='0.5')
    headless_arg = DeclareLaunchArgument('headless', default_value='true')
    fixed_arg    = DeclareLaunchArgument('use_fixed_alpha', default_value='true')
    ekf_arg      = DeclareLaunchArgument('ekf',      default_value='false')

    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg, 'launch', 'simulation.launch.py')),
        launch_arguments={
            'world':    LaunchConfiguration('world'),
            'seed':     LaunchConfiguration('seed'),
            'headless': LaunchConfiguration('headless'),
            'rviz':     'false',
        }.items())

    # Static alpha for B1 (1.0) / B2 (0.0) / B3 (0.5)
    alpha_pub = TimerAction(period=8.5, actions=[
        ExecuteProcess(
            cmd=['ros2', 'topic', 'pub', '--rate', '10',
                 '/fusion/alpha', 'std_msgs/msg/Float32',
                 ['{data: ', LaunchConfiguration('alpha'), '}']],
            output='screen',
            condition=IfCondition(LaunchConfiguration('use_fixed_alpha'))),
    ])

    # Baseline B4: covariance-weighted EKF fusion (Methodology §3.5)
    ekf_node = TimerAction(period=9.0, actions=[
        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node',
            output='screen',
            parameters=[os.path.join(pkg, 'config', 'ekf.yaml')],
            condition=IfCondition(LaunchConfiguration('ekf'))),
    ])

    return LaunchDescription([
        world_arg, seed_arg, alpha_arg, headless_arg, fixed_arg, ekf_arg,
        sim, alpha_pub, ekf_node,
    ])
