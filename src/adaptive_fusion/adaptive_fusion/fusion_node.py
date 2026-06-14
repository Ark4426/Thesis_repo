"""
Fusion Node
Blends visual and LiDAR pose estimates using alpha from the DRL agent.
  fused_position    = alpha * visual_pos + (1-alpha) * lidar_pos
  fused_orientation = SLERP(lidar_quat, visual_quat, alpha)

Subscribes: /fusion/visual_pose, /fusion/lidar_pose, /fusion/alpha
Publishes:  /fusion/fused_pose
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import numpy as np

from std_msgs.msg import Float32
from geometry_msgs.msg import PoseStamped


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two quaternions."""
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        return (q0 + t * (q1 - q0)) / np.linalg.norm(q0 + t * (q1 - q0))
    theta_0 = np.arccos(dot)
    theta   = theta_0 * t
    sin_t   = np.sin(theta)
    sin_0   = np.sin(theta_0)
    s0 = np.cos(theta) - dot * sin_t / sin_0
    s1 = sin_t / sin_0
    return s0 * q0 + s1 * q1


def _quat_to_array(q) -> np.ndarray:
    return np.array([q.x, q.y, q.z, q.w])


class FusionNode(Node):
    def __init__(self):
        super().__init__('fusion_node')

        qos = QoSProfile(depth=10,
                         reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE)

        self._alpha        = 0.5
        self._visual_pose  = None
        self._lidar_pose   = None

        self.create_subscription(Float32,      '/fusion/alpha',       self._cb_alpha,  10)
        self.create_subscription(PoseStamped,  '/fusion/visual_pose', self._cb_visual, qos)
        self.create_subscription(PoseStamped,  '/fusion/lidar_pose',  self._cb_lidar,  qos)

        self.fused_pub = self.create_publisher(PoseStamped, '/fusion/fused_pose', 10)

        self.create_timer(0.1, self._publish_fused)

    def _cb_alpha(self, msg):
        self._alpha = float(np.clip(msg.data, 0.0, 1.0))

    def _cb_visual(self, msg):
        self._visual_pose = msg

    def _cb_lidar(self, msg):
        self._lidar_pose = msg

    def _publish_fused(self):
        if self._visual_pose is None or self._lidar_pose is None:
            return

        a  = self._alpha
        vp = self._visual_pose.pose
        lp = self._lidar_pose.pose

        fused = PoseStamped()
        fused.header.stamp    = self.get_clock().now().to_msg()
        fused.header.frame_id = 'map'

        # Linear blend of positions
        fused.pose.position.x = a * vp.position.x + (1.0 - a) * lp.position.x
        fused.pose.position.y = a * vp.position.y + (1.0 - a) * lp.position.y
        fused.pose.position.z = a * vp.position.z + (1.0 - a) * lp.position.z

        # SLERP on orientations
        q_lidar  = _quat_to_array(lp.orientation)
        q_visual = _quat_to_array(vp.orientation)
        # Guard against zero quaternions (before SLAM initialises)
        if np.linalg.norm(q_lidar) < 1e-6:
            q_lidar = np.array([0.0, 0.0, 0.0, 1.0])
        if np.linalg.norm(q_visual) < 1e-6:
            q_visual = np.array([0.0, 0.0, 0.0, 1.0])

        q_fused = _slerp(q_lidar, q_visual, a)
        fused.pose.orientation.x = q_fused[0]
        fused.pose.orientation.y = q_fused[1]
        fused.pose.orientation.z = q_fused[2]
        fused.pose.orientation.w = q_fused[3]

        self.fused_pub.publish(fused)


def main(args=None):
    rclpy.init(args=args)
    node = FusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
