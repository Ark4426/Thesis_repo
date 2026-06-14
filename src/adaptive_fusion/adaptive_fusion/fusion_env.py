"""
FusionEnv — Gymnasium environment wrapping the ROS 2 / Gazebo simulation.

Observation : Float32[6]  — sensor quality state vector from /fusion/state
Action      : Float32[1]  — alpha in [0, 1]
Reward      : -||fused_pos - gt_pos||_2  (clipped at -3.0 per step)
Episode     : 600 steps (60 s at 10 Hz)

Usage:
    env = FusionEnv(world='w1_static')
    obs, _ = env.reset(seed=42)
    obs, rew, done, trunc, info = env.step(np.array([0.7]))
"""

import os
import time
import threading
import subprocess
import numpy as np

import gymnasium as gym
from gymnasium import spaces

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from std_msgs.msg import Float32, Float32MultiArray, Empty as EmptyMsg
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Empty as EmptySrv

try:
    # Humble's slam_toolbox has no reset service; episode resets are done by
    # re-loading the pose graph serialized right after startup.
    from slam_toolbox.srv import SerializePoseGraph, DeserializePoseGraph
except ImportError:
    SerializePoseGraph = DeserializePoseGraph = None

try:
    from rtabmap_msgs.srv import ResetPose
except ImportError:
    ResetPose = None

SLAM_GRAPH_SNAPSHOT = '/tmp/adaptive_fusion_initial_graph'


WORLDS = ['w1_static', 'w2_low_dynamic', 'w3_high_dynamic', 'w4_visually_degraded']

# Must match the spawn pose in simulation.launch.py — visual odometry is
# re-seeded here after every world reset to keep all estimates world-framed.
SPAWN_POSE = (-7.0, 0.0, 0.0)   # x, y, yaw

# §3.2 deviation: clip raised 1 m → 3 m after pilot-2. At 1 m the normal
# operating range (W1 LiDAR 0.26 m … W4 vision 3.21 m) sat mostly ABOVE the
# clip, so every step saturated at -1.0, returns became identical, advantages
# went to ~0 and learning stalled (the -600 floor). At 3 m the per-step
# penalty is informative across the real sensor range (good actions ≈ -0.3,
# bad actions ≈ -3.0), restoring the gradient that teaches sensor switching,
# while still bounding catastrophic tracking-loss spikes.
MAX_REWARD_CLIP = 3.0   # clip per-step penalty at 3 m


class _FusionROSNode(Node):
    """Internal ROS 2 node that handles all pub/sub for FusionEnv."""

    def __init__(self):
        super().__init__('fusion_env_node')

        qos = QoSProfile(depth=5,
                         reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE)

        self._state      = np.zeros(6, dtype=np.float32)
        self._fused_pos  = np.zeros(3)
        self._gt_pos     = np.zeros(3)
        self._got_fused  = False
        self._state_lock = threading.Lock()
        self._new_state  = threading.Event()

        self.create_subscription(Float32MultiArray, '/fusion/state',
                                 self._cb_state, qos)
        self.create_subscription(PoseStamped, '/fusion/fused_pose',
                                 self._cb_fused, qos)
        self.create_subscription(PoseStamped, '/fusion/ground_truth',
                                 self._cb_gt, qos)

        self._alpha_pub = self.create_publisher(Float32, '/fusion/alpha', 10)
        self._ep_reset_pub = self.create_publisher(EmptyMsg, '/fusion/episode_reset', 10)

        self._reset_world_cli = self.create_client(EmptySrv, '/reset_world')
        # rtabmap's services are sub-scoped under the node name
        self._reset_rtab_cli  = self.create_client(EmptySrv, '/rtabmap/rtabmap/reset')
        self._reset_vodom_cli = (self.create_client(ResetPose, '/rtabmap/reset_odom_to_pose')
                                 if ResetPose is not None else None)
        self._slam_save_cli = self._slam_load_cli = None
        if SerializePoseGraph is not None:
            self._slam_save_cli = self.create_client(
                SerializePoseGraph, '/slam_toolbox/serialize_map')
            self._slam_load_cli = self.create_client(
                DeserializePoseGraph, '/slam_toolbox/deserialize_map')

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _cb_state(self, msg):
        with self._state_lock:
            self._state = np.array(msg.data, dtype=np.float32)
        self._new_state.set()

    def _cb_fused(self, msg):
        p = msg.pose.position
        self._fused_pos = np.array([p.x, p.y, p.z])
        self._got_fused = True

    def _cb_gt(self, msg):
        p = msg.pose.position
        self._gt_pos = np.array([p.x, p.y, p.z])

    # ── API used by FusionEnv ─────────────────────────────────────────────────

    def publish_alpha(self, alpha: float):
        self._alpha_pub.publish(Float32(data=float(alpha)))

    def publish_episode_reset(self):
        self._ep_reset_pub.publish(EmptyMsg())

    def get_state(self) -> np.ndarray:
        with self._state_lock:
            return self._state.copy()

    def get_reward(self) -> float:
        err = float(np.linalg.norm(self._fused_pos - self._gt_pos))
        return -float(np.clip(err, 0.0, MAX_REWARD_CLIP))

    def fusion_alive(self) -> bool:
        return self._got_fused

    def mark_pipeline_down(self):
        """Forget liveness from a previous simulation instance, so
        wait_for_pipeline() really waits for the NEW one after a world
        relaunch (otherwise the stale flag returns immediately and the
        startup pose-graph snapshot can capture the wrong world)."""
        self._got_fused = False

    def wait_for_state(self, timeout=0.3):
        self._new_state.clear()
        self._new_state.wait(timeout=timeout)

    def _call_empty(self, client, name):
        if client.wait_for_service(timeout_sec=2.0):
            client.call_async(EmptySrv.Request())
        else:
            self.get_logger().warn(f'{name} service unavailable')

    def call_reset_world(self):
        self._call_empty(self._reset_world_cli, '/reset_world')

    def call_reset_rtabmap(self):
        self._call_empty(self._reset_rtab_cli, '/rtabmap/reset')

    def call_reset_visual_odom(self):
        if self._reset_vodom_cli is None:
            return
        if self._reset_vodom_cli.wait_for_service(timeout_sec=2.0):
            req = ResetPose.Request()
            req.x, req.y, req.z = SPAWN_POSE[0], SPAWN_POSE[1], 0.0
            req.roll, req.pitch, req.yaw = 0.0, 0.0, SPAWN_POSE[2]
            self._reset_vodom_cli.call_async(req)
        else:
            self.get_logger().warn('/rtabmap/reset_odom_to_pose service unavailable')

    def save_slam_snapshot(self):
        """Serialize the just-initialised pose graph (called once at startup)."""
        if self._slam_save_cli is None:
            return
        if self._slam_save_cli.wait_for_service(timeout_sec=5.0):
            req = SerializePoseGraph.Request()
            req.filename = SLAM_GRAPH_SNAPSHOT
            self._slam_save_cli.call_async(req)
        else:
            self.get_logger().warn('/slam_toolbox/serialize_map unavailable')

    def call_reset_slam_toolbox(self):
        """Reset = re-load the startup pose graph anchored at the spawn pose."""
        if self._slam_load_cli is None:
            return
        if self._slam_load_cli.wait_for_service(timeout_sec=2.0):
            req = DeserializePoseGraph.Request()
            req.filename = SLAM_GRAPH_SNAPSHOT
            req.match_type = DeserializePoseGraph.Request.START_AT_GIVEN_POSE
            req.initial_pose.x = SPAWN_POSE[0]
            req.initial_pose.y = SPAWN_POSE[1]
            req.initial_pose.theta = SPAWN_POSE[2]
            self._slam_load_cli.call_async(req)
        else:
            self.get_logger().warn('/slam_toolbox/deserialize_map unavailable')


class FusionEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, world: str = 'w1_static', headless: bool = True,
                 worlds=None, world_block: int = 10):
        """
        world       : fixed world (used when `worlds` is None)
        worlds      : list of worlds to sample uniformly during training
                      (§3.3 domain randomisation). Gazebo cannot swap worlds
                      in-place, so sampling is done in blocks: every
                      `world_block` episodes the simulation is relaunched
                      with a freshly sampled world. Uniform in expectation;
                      the block size only trades relaunch overhead against
                      within-block correlation.
        world_block : episodes per sampled world before relaunching
        """
        super().__init__()

        assert world in WORLDS, f'Unknown world: {world}. Choose from {WORLDS}'
        if worlds is not None:
            for w in worlds:
                assert w in WORLDS, f'Unknown world: {w}. Choose from {WORLDS}'
        self.world         = world
        self.headless      = headless
        self._worlds       = list(worlds) if worlds else None
        self._world_block  = max(1, int(world_block))
        self._eps_in_block = 0

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(6,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        self._max_steps   = 600
        self._step_count  = 0
        self._launch_proc = None
        self._ros_node    = None
        self._spin_thread = None
        self._rng         = np.random.default_rng(0)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def _start_ros(self):
        if not rclpy.ok():
            rclpy.init()
        if self._ros_node is None:
            self._ros_node = _FusionROSNode()
            self._spin_thread = threading.Thread(
                target=rclpy.spin, args=(self._ros_node,), daemon=True)
            self._spin_thread.start()

    def _shutdown_simulation(self):
        if self._launch_proc is None:
            return
        self._launch_proc.terminate()
        try:
            self._launch_proc.wait(timeout=8.0)
        except subprocess.TimeoutExpired:
            self._launch_proc.kill()
        self._launch_proc = None
        # A deadlocked gzserver can survive SIGTERM and poison every later
        # launch (spawn fails, duplicate /clock) — make sure it is gone.
        subprocess.run(['pkill', '-9', '-f', 'gzserver'], check=False)
        time.sleep(2.0)

    def _launch_simulation(self, seed: int):
        self._shutdown_simulation()

        env = os.environ.copy()
        env['TURTLEBOT3_MODEL'] = 'waffle_pi'

        cmd = [
            'ros2', 'launch', 'adaptive_fusion', 'simulation.launch.py',
            f'world:={self.world}',
            f'seed:={seed}',
            f'headless:={str(self.headless).lower()}',
        ]
        self._launch_proc = subprocess.Popen(
            cmd, env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _wait_for_pipeline(self, timeout: float = 90.0):
        """Block until the fusion pipeline is publishing."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            self._ros_node.wait_for_state(timeout=1.0)
            if self._ros_node.fusion_alive():
                return True
        return False

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        ep_seed = int(self._rng.integers(0, 10000))

        # §3.3 world sampling: relaunch with a fresh uniformly-sampled world
        # at every block boundary (Gazebo cannot swap worlds in-place).
        need_launch = self._launch_proc is None
        if (self._worlds is not None and not need_launch
                and self._eps_in_block >= self._world_block):
            self._shutdown_simulation()
            need_launch = True

        if need_launch:
            if self._worlds is not None:
                self.world = str(self._rng.choice(self._worlds))
            if self._ros_node is not None:
                self._ros_node.mark_pipeline_down()
            self._launch_simulation(ep_seed)
            self._start_ros()
            if not self._wait_for_pipeline():
                raise RuntimeError(
                    'Fusion pipeline did not come up within 90 s — '
                    'check the simulation launch log.')
            # snapshot the fresh pose graph (per world) for episode resets
            self._ros_node.save_slam_snapshot()
            self._eps_in_block = 0
        else:
            # Soft reset: world + visual odometry + both SLAM maps,
            # then restart the exploration pattern.
            self._ros_node.call_reset_world()
            time.sleep(1.0)
            self._ros_node.call_reset_visual_odom()
            self._ros_node.call_reset_rtabmap()
            self._ros_node.call_reset_slam_toolbox()
            time.sleep(2.0)
            self._ros_node.publish_episode_reset()

        self._eps_in_block += 1
        self._step_count = 0

        # Collect a fresh state after the reset settles
        for _ in range(10):
            self._ros_node.wait_for_state(timeout=0.3)

        return self._ros_node.get_state(), {}

    def step(self, action):
        alpha = float(np.clip(action[0], 0.0, 1.0))
        self._ros_node.publish_alpha(alpha)
        self._ros_node.wait_for_state(timeout=0.2)

        obs     = self._ros_node.get_state()
        reward  = self._ros_node.get_reward()
        self._step_count += 1
        terminated = False
        truncated  = self._step_count >= self._max_steps

        return obs, reward, terminated, truncated, {'alpha': alpha}

    def close(self):
        self._shutdown_simulation()
        if rclpy.ok():
            rclpy.shutdown()
