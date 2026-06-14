"""
Exploration Controller
Drives TurtleBot3 around a rectangular patrol circuit through the warehouse
aisles using wheel-odometry feedback (waypoint following), giving repeated
turns and loop closures instead of a single straight line:

    (-8.9, 0) → (8.9, 0) → (8.9, 5) → (-8.9, 5) → loop

Subscribes /fusion/episode_reset (std_msgs/Empty): restarts the circuit from
the first waypoint — published by FusionEnv at every episode boundary.
"""

import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty
import numpy as np


# Rectangular patrol circuit: y=0 aisle → east corridor → y=5 aisle → west
WAYPOINTS = [(8.9, 0.0), (8.9, 5.0), (-8.9, 5.0), (-8.9, 0.0)]

FWD_SPEED_BASE = 0.3    # m/s
TURN_GAIN = 1.8         # P gain on heading error
MAX_TURN = 0.9          # rad/s
WP_TOLERANCE = 0.35     # m


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class ExplorationController(Node):
    def __init__(self):
        super().__init__('exploration_controller')

        self.declare_parameter('seed', 0)
        seed = self.get_parameter('seed').value
        rng = np.random.default_rng(seed)
        self._fwd_speed = FWD_SPEED_BASE + rng.uniform(-0.04, 0.04)

        self._pose = None          # (x, y, yaw) from wheel odometry
        self._wp_idx = 0

        self._cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Odometry, '/odom', self._cb_odom, 20)
        self.create_subscription(Empty, '/fusion/episode_reset',
                                 self._cb_episode_reset, 10)

        self.create_timer(0.1, self._control_loop)   # 10 Hz
        self.get_logger().info(
            f'ExplorationController: patrol circuit, speed={self._fwd_speed:.2f} m/s')

    def _cb_odom(self, msg):
        p = msg.pose.pose.position
        self._pose = (p.x, p.y, yaw_from_quat(msg.pose.pose.orientation))

    def _cb_episode_reset(self, _msg):
        self._wp_idx = 0
        self.get_logger().info('Episode reset — circuit restarted')

    def _control_loop(self):
        if self._pose is None:
            return

        x, y, yaw = self._pose
        tx, ty = WAYPOINTS[self._wp_idx]

        if math.hypot(tx - x, ty - y) < WP_TOLERANCE:
            self._wp_idx = (self._wp_idx + 1) % len(WAYPOINTS)
            tx, ty = WAYPOINTS[self._wp_idx]

        heading_err = math.atan2(ty - y, tx - x) - yaw
        heading_err = math.atan2(math.sin(heading_err), math.cos(heading_err))

        twist = Twist()
        twist.angular.z = float(np.clip(TURN_GAIN * heading_err, -MAX_TURN, MAX_TURN))
        # slow down while turning sharply, stop-and-turn at corners
        if abs(heading_err) < 0.4:
            twist.linear.x = self._fwd_speed
        elif abs(heading_err) < 1.0:
            twist.linear.x = 0.5 * self._fwd_speed
        else:
            twist.linear.x = 0.0
        self._cmd_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = ExplorationController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
