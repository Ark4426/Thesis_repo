"""
simulation.launch.py
Launches the full simulation stack (Methodology §3.4):
  Gazebo Classic 11 → TurtleBot3 Waffle Pi (RGB-D variant) →
  RTAB-Map RGB-D odometry + SLAM → SLAM Toolbox (online async) →
  sensor_quality_extractor → fusion_node → exploration_controller
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess,
                            TimerAction, SetEnvironmentVariable)
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    pkg    = get_package_share_directory('adaptive_fusion')
    tb3_gz = get_package_share_directory('turtlebot3_gazebo')

    world_arg    = DeclareLaunchArgument('world', default_value='w1_static')
    seed_arg     = DeclareLaunchArgument('seed',  default_value='0')
    headless_arg = DeclareLaunchArgument('headless', default_value='true')
    rviz_arg     = DeclareLaunchArgument('rviz', default_value='false')

    world    = LaunchConfiguration('world')
    seed     = LaunchConfiguration('seed')
    headless = LaunchConfiguration('headless')
    rviz     = LaunchConfiguration('rviz')

    world_file = PathJoinSubstitution([pkg, 'worlds', [world, '.world']])

    # Gazebo needs model:// URIs from turtlebot3_gazebo (meshes) and this package
    model_path = ':'.join(filter(None, [
        os.path.join(tb3_gz, 'models'),
        os.path.join(pkg, 'models'),
        os.environ.get('GAZEBO_MODEL_PATH', ''),
    ]))

    # ── Gazebo ────────────────────────────────────────────────────────────────
    # gazebo_ros_state is a world plugin declared inside each .world file;
    # only init and factory are system plugins.
    gz_headless = ExecuteProcess(
        cmd=['gzserver', '--verbose', world_file,
             '-s', 'libgazebo_ros_init.so',
             '-s', 'libgazebo_ros_factory.so'],
        output='screen',
        condition=IfCondition(headless))

    gz_gui = ExecuteProcess(
        cmd=['gazebo', '--verbose', world_file,
             '-s', 'libgazebo_ros_init.so',
             '-s', 'libgazebo_ros_factory.so'],
        output='screen',
        condition=UnlessCondition(headless))

    # ── TurtleBot3 Waffle Pi spawn (RGB-D variant of the stock SDF) ───────────
    sdf_model = os.path.join(pkg, 'models', 'turtlebot3_waffle_pi_rgbd', 'model.sdf')

    urdf_file = os.path.join(
        get_package_share_directory('turtlebot3_description'),
        'urdf', 'turtlebot3_waffle_pi.urdf')

    # The installed URDF carries unexpanded ${namespace} placeholders, which
    # would break every TF frame name — strip them before publishing.
    robot_description = ''
    if os.path.exists(urdf_file):
        robot_description = open(urdf_file).read().replace('${namespace}', '')

    spawn_robot = TimerAction(period=4.0, actions=[
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=[
                '-entity', 'turtlebot3_waffle_pi',
                '-file', sdf_model,
                # aisle start: y=0 corridor runs clear from x=-8 to x=+8
                '-x', '-7.0', '-y', '0.0', '-z', '0.01',
                '-R', '0.0', '-P', '0.0', '-Y', '0.0',
            ],
            output='screen'),
    ])

    robot_state_pub = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': True,
        }])

    # ── Odometry noise (diff-drive TF disabled in SDF; this node owns
    #    odom→base_footprint and adds realistic drift to the perfect odom) ────
    odom_noise = TimerAction(period=5.0, actions=[
        Node(
            package='adaptive_fusion',
            executable='odom_noise',
            name='odom_noise',
            output='screen',
            parameters=[
                {'use_sim_time': True},
                {'seed': seed},
            ]),
    ])

    # ── RTAB-Map RGB-D visual odometry (publishes /rtabmap/odom + odom_info) ──
    rgbd_odometry = TimerAction(period=6.0, actions=[
        Node(
            package='rtabmap_odom',
            executable='rgbd_odometry',
            name='rgbd_odometry',
            namespace='rtabmap',
            output='screen',
            parameters=[{
                'frame_id': 'base_footprint',
                'odom_frame_id': 'odom_vis',
                # Gazebo's diff-drive odometry is world-anchored, so the LiDAR
                # estimate lives in world coordinates; start visual odometry at
                # the spawn pose to keep both estimates in the same frame.
                'initial_pose': '-7.0 0.0 0.0 0.0 0.0 0.0',
                # Wheel-odometry motion prior (§3.4: the drift-injected odom
                # feeds BOTH SLAM pipelines). Without it, raw F2M visual
                # odometry accumulates ~8 %-of-distance yaw drift and resets
                # to a garbage origin after tracking loss (measured
                # 2026-06-12: v_err 3.6 m after 100 s vs LiDAR 0.3 m).
                'guess_frame_id': 'odom',
                'publish_tf': False,        # slam_toolbox owns map->odom
                # Gazebo's depth camera stamps rgb and depth identically,
                # so exact sync is safe and avoids mispaired frames.
                'approx_sync': False,
                'qos': 2,
                'use_sim_time': True,
                # During tracking loss publish the wheel-odom-propagated pose
                # (large covariance) instead of a null pose at the ORIGIN —
                # blending a null pose injects a |robot position| error spike
                # (measured 8.68 m in W1) and put pilot-1 rewards on the
                # -600 floor in the dark/dynamic worlds where losses are
                # frequent. The state features (matches -> 0) still tell the
                # agent vision is down.
                'publish_null_when_lost': False,
                'Odom/ResetCountdown': '1',  # auto-recover after tracking loss
                'Reg/Force3DoF': 'true',
                'Vis/MinInliers': '15',
            }],
            remappings=[
                ('rgb/image',       '/camera/image_raw'),
                ('rgb/camera_info', '/camera/camera_info'),
                ('depth/image',     '/camera/depth/image_raw'),
            ]),
    ])

    # ── RTAB-Map SLAM (consumes visual odometry; mapping + loop closure) ──────
    rtabmap_node = TimerAction(period=6.0, actions=[
        Node(
            package='rtabmap_slam',
            executable='rtabmap',
            name='rtabmap',
            namespace='rtabmap',
            output='screen',
            arguments=['-d'],   # delete previous session database
            parameters=[
                os.path.join(pkg, 'config', 'rtabmap_rgbd.yaml'),
                {'use_sim_time': True, 'approx_sync': True,
                 'approx_sync_max_interval': 0.05, 'qos': 2},
            ],
            remappings=[
                ('rgb/image',       '/camera/image_raw'),
                ('rgb/camera_info', '/camera/camera_info'),
                ('depth/image',     '/camera/depth/image_raw'),
            ]),
    ])

    # ── SLAM Toolbox (LiDAR SLAM, online async) ───────────────────────────────
    slam_toolbox_node = TimerAction(period=6.0, actions=[
        Node(
            package='slam_toolbox',
            executable='async_slam_toolbox_node',
            name='slam_toolbox',
            output='screen',
            parameters=[
                os.path.join(pkg, 'config', 'slam_toolbox.yaml'),
                {'use_sim_time': True},
            ]),
    ])

    # ── Custom nodes ──────────────────────────────────────────────────────────
    sensor_extractor = TimerAction(period=8.0, actions=[
        Node(
            package='adaptive_fusion',
            executable='sensor_quality_extractor',
            name='sensor_quality_extractor',
            output='screen',
            parameters=[{'use_sim_time': True}]),
    ])

    fusion_node = TimerAction(period=8.0, actions=[
        Node(
            package='adaptive_fusion',
            executable='fusion_node',
            name='fusion_node',
            output='screen',
            parameters=[{'use_sim_time': True}]),
    ])

    # ── Vehicle animator (forklifts/carts — kinematic via set_entity_state) ──
    vehicle_animator = TimerAction(period=6.0, actions=[
        Node(
            package='adaptive_fusion',
            executable='vehicle_animator',
            name='vehicle_animator',
            output='screen',
            parameters=[{'use_sim_time': True}]),
    ])

    # ── RViz (sensor/fusion visualisation for thesis figures) ─────────────────
    rviz_node = TimerAction(period=10.0, actions=[
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', os.path.join(pkg, 'config', 'fusion.rviz')],
            parameters=[{'use_sim_time': True}],
            condition=IfCondition(rviz)),
    ])

    explorer = TimerAction(period=12.0, actions=[
        Node(
            package='adaptive_fusion',
            executable='exploration_controller',
            name='exploration_controller',
            output='screen',
            parameters=[
                {'use_sim_time': True},
                {'seed': seed},
            ]),
    ])

    return LaunchDescription([
        SetEnvironmentVariable('TURTLEBOT3_MODEL', 'waffle_pi'),
        SetEnvironmentVariable('GAZEBO_MODEL_PATH', model_path),
        world_arg, seed_arg, headless_arg, rviz_arg,
        gz_headless, gz_gui,
        robot_state_pub,
        spawn_robot,
        odom_noise,
        rgbd_odometry,
        rtabmap_node,
        slam_toolbox_node,
        sensor_extractor,
        fusion_node,
        vehicle_animator,
        rviz_node,
        explorer,
    ])
