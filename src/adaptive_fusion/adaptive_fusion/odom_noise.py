"""
Odometry Noise Node
Gazebo's TB3 diff-drive odometry is world-anchored and essentially perfect,
which would hand the LiDAR SLAM baseline a ground-truth motion prior. This
node consumes the perfect /odom, accumulates realistic drift on the
incremental motion, and re-publishes the noisy estimate as the odom->base
TF (the diff-drive plugin's own TF broadcast is disabled in the model SDF).

Noise model (odometry drift, cf. Thrun et al. 2005 §5.4):
  translation noise ~ N(0, (k_d * |dd|)^2)  per step, applied per axis
  rotation noise    ~ N(0, (k_r * |dyaw| + k_dr * |dd|)^2)

A position jump > jump_threshold (teleport / world reset) re-seeds the
noisy pose from the true pose and clears accumulated drift.
"""

import math
import numpy as np

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
import tf2_ros


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class OdomNoise(Node):
    def __init__(self):
        super().__init__('odom_noise')

        self.declare_parameter('k_d', 0.02)        # translational drift per metre
        self.declare_parameter('k_r', 0.01)        # rotational drift per radian
        self.declare_parameter('k_dr', 0.002)      # rotational drift per metre
        self.declare_parameter('jump_threshold', 1.0)
        self.declare_parameter('seed', 0)

        self.k_d  = self.get_parameter('k_d').value
        self.k_r  = self.get_parameter('k_r').value
        self.k_dr = self.get_parameter('k_dr').value
        self.jump = self.get_parameter('jump_threshold').value
        self.rng  = np.random.default_rng(self.get_parameter('seed').value)

        self.prev = None          # (x, y, yaw) true
        self.noisy = None         # (x, y, yaw) noisy

        self.tf_bc = tf2_ros.TransformBroadcaster(self)
        self.noisy_pub = self.create_publisher(Odometry, '/odom_noisy', 10)
        self.create_subscription(Odometry, '/odom', self._cb_odom, 20)

    def _cb_odom(self, msg):
        p = msg.pose.pose.position
        x, y = p.x, p.y
        yaw = yaw_from_quat(msg.pose.pose.orientation)

        if self.prev is None:
            self.prev = (x, y, yaw)
            self.noisy = (x, y, yaw)
        else:
            px, py, pyaw = self.prev
            dx_w, dy_w = x - px, y - py
            dd = math.hypot(dx_w, dy_w)

            if dd > self.jump:
                # teleport (world reset): re-seed, drop accumulated drift
                self.prev = (x, y, yaw)
                self.noisy = (x, y, yaw)
            else:
                dyaw = math.atan2(math.sin(yaw - pyaw), math.cos(yaw - pyaw))
                # incremental motion in the previous body frame
                c, s = math.cos(-pyaw), math.sin(-pyaw)
                dx_b = c * dx_w - s * dy_w
                dy_b = s * dx_w + c * dy_w

                sd = self.k_d * dd
                sr = self.k_r * abs(dyaw) + self.k_dr * dd
                dx_b += self.rng.normal(0.0, sd) if sd > 0 else 0.0
                dy_b += self.rng.normal(0.0, sd) if sd > 0 else 0.0
                dyaw += self.rng.normal(0.0, sr) if sr > 0 else 0.0

                nx, ny, nyaw = self.noisy
                cn, sn = math.cos(nyaw), math.sin(nyaw)
                nx += cn * dx_b - sn * dy_b
                ny += sn * dx_b + cn * dy_b
                nyaw = math.atan2(math.sin(nyaw + dyaw), math.cos(nyaw + dyaw))

                self.prev = (x, y, yaw)
                self.noisy = (nx, ny, nyaw)

        nx, ny, nyaw = self.noisy

        tf = TransformStamped()
        tf.header.stamp = msg.header.stamp
        tf.header.frame_id = 'odom'
        tf.child_frame_id = 'base_footprint'
        tf.transform.translation.x = nx
        tf.transform.translation.y = ny
        tf.transform.translation.z = 0.0
        tf.transform.rotation.z = math.sin(nyaw / 2.0)
        tf.transform.rotation.w = math.cos(nyaw / 2.0)
        self.tf_bc.sendTransform(tf)

        out = Odometry()
        out.header = msg.header
        out.header.frame_id = 'odom'
        out.child_frame_id = 'base_footprint'
        out.pose.pose.position.x = nx
        out.pose.pose.position.y = ny
        out.pose.pose.orientation.z = tf.transform.rotation.z
        out.pose.pose.orientation.w = tf.transform.rotation.w
        out.twist = msg.twist
        self.noisy_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = OdomNoise()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
