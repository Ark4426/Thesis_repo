"""
Sensor Quality Extractor Node
Computes the 6-feature state vector defined in Methodology §3.1:
  f0: visual feature matches in the current frame   (normalised count)
  f1: mean reprojection error of those matches      (RTAB-Map local-bundle
      mean inlier distance when available; outlier-ratio proxy otherwise)
  f2: visual tracking inlier ratio
  f3: LiDAR scan-matching quality score             (valid-return ratio)
  f4: trace of the LiDAR pose covariance            (from slam_toolbox /pose)
  f5: cross-modal pose disagreement                 (metres, clipped)
All features are normalised to [0, 1].
Publishes /fusion/state, /fusion/visual_pose, /fusion/lidar_pose, /fusion/ground_truth
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import numpy as np

from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from gazebo_msgs.msg import ModelStates

try:
    from rtabmap_msgs.msg import OdomInfo
except ImportError:
    OdomInfo = None

import tf2_ros
from tf2_ros import TransformException


# Normalisation constants
MAX_MATCHES = 400.0          # ORB/NFeatures in rtabmap_rgbd.yaml
REPROJ_DIST_MAX = 0.2        # m — saturates the mean inlier-distance feature
MAX_DISAGREEMENT = 1.0       # m — clips the disagreement signal (healthy
                             # cross-modal disagreement lives well below 1 m;
                             # matches the 1 m reward clip of §3.2)
COV_TRACE_MAX = 0.25         # m² — saturates the covariance-trace feature
                             # (degradation threshold of §3.4 is 0.05 m² → 0.2)


class SensorQualityExtractor(Node):
    def __init__(self):
        super().__init__('sensor_quality_extractor')

        qos = QoSProfile(depth=10,
                         reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE)

        # ── Subscribers ──────────────────────────────────────────────────────
        self.create_subscription(Odometry, '/rtabmap/odom',
                                 self._cb_visual_odom, qos)
        self.create_subscription(LaserScan, '/scan',
                                 self._cb_scan, qos)
        self.create_subscription(PoseWithCovarianceStamped, '/pose',
                                 self._cb_lidar_pose_cov, 10)
        self.create_subscription(ModelStates, '/gazebo/model_states',
                                 self._cb_gt, 10)

        if OdomInfo is not None:
            self.create_subscription(OdomInfo, '/rtabmap/odom_info',
                                     self._cb_odom_info, qos)
        else:
            self.get_logger().warn('rtabmap_msgs not found — visual features unavailable')

        # ── Publishers ───────────────────────────────────────────────────────
        self.state_pub  = self.create_publisher(Float32MultiArray, '/fusion/state', 10)
        self.vpose_pub  = self.create_publisher(PoseStamped, '/fusion/visual_pose', 10)
        self.lpose_pub  = self.create_publisher(PoseStamped, '/fusion/lidar_pose', 10)
        self.gt_pub     = self.create_publisher(PoseStamped, '/fusion/ground_truth', 10)
        # B4 EKF inputs: the same two pose streams the fusion blend uses,
        # with covariances attached. Visual = /rtabmap/odom relabelled to
        # 'map' (odom_vis is world-anchored but absent from the TF tree);
        # LiDAR = the continuous TF pose with the latest /pose covariance
        # (raw /pose messages are too sparse under load to drive an EKF).
        self.vcov_pub   = self.create_publisher(PoseWithCovarianceStamped,
                                                '/fusion/visual_pose_cov', 10)
        self.lcov_pub   = self.create_publisher(PoseWithCovarianceStamped,
                                                '/fusion/lidar_pose_cov', 10)

        # ── TF buffer (for SLAM Toolbox map→base_footprint) ──────────────────
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ── State ─────────────────────────────────────────────────────────────
        self._matches     = 0
        self._inliers     = 0
        self._inlier_dist = 0.0
        self._visual_pos  = np.zeros(3)
        self._lidar_pos   = np.zeros(3)
        self._scan_valid  = 1.0
        self._cov_trace   = 0.0
        self._lidar_cov   = None     # latest full covariance from /pose
        self._gt_pos      = np.zeros(3)
        self._robot_name  = 'turtlebot3_waffle_pi'

        # Publish at 10 Hz
        self.create_timer(0.1, self._publish_state)

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _cb_odom_info(self, msg):
        self._matches     = int(msg.matches)
        self._inliers     = int(msg.inliers)
        self._inlier_dist = float(msg.local_bundle_avg_inlier_distance)

    def _cb_visual_odom(self, msg):
        p = msg.pose.pose.position
        self._visual_pos = np.array([p.x, p.y, p.z])

        ps = PoseStamped()
        ps.header = msg.header
        ps.pose   = msg.pose.pose
        self.vpose_pub.publish(ps)

        pc = PoseWithCovarianceStamped()
        pc.header.stamp    = msg.header.stamp
        pc.header.frame_id = 'map'
        pc.pose            = msg.pose
        self.vcov_pub.publish(pc)

    def _cb_lidar_pose_cov(self, msg):
        c = msg.pose.covariance
        # planar SLAM: x, y, yaw variances carry the information
        self._cov_trace = float(c[0] + c[7])
        self._lidar_cov = list(c)

    def _cb_scan(self, msg):
        ranges = np.array(msg.ranges, dtype=np.float32)
        valid  = np.isfinite(ranges) & (ranges > msg.range_min) & (ranges < msg.range_max)
        self._scan_valid = int(np.sum(valid)) / max(len(ranges), 1)

        # Update LiDAR pose from TF (slam_toolbox map→odom ⊕ wheel odom→base)
        try:
            tf = self.tf_buffer.lookup_transform(
                'map', 'base_footprint', rclpy.time.Time())
            t = tf.transform.translation
            self._lidar_pos = np.array([t.x, t.y, t.z])

            ps = PoseStamped()
            ps.header.stamp    = self.get_clock().now().to_msg()
            ps.header.frame_id = 'map'
            ps.pose.position.x = t.x
            ps.pose.position.y = t.y
            ps.pose.position.z = t.z
            ps.pose.orientation = tf.transform.rotation
            self.lpose_pub.publish(ps)

            pc = PoseWithCovarianceStamped()
            pc.header = ps.header
            pc.pose.pose = ps.pose
            if self._lidar_cov is not None:
                pc.pose.covariance = self._lidar_cov
            else:
                # before the first /pose message: modest default uncertainty
                pc.pose.covariance[0]  = 0.05
                pc.pose.covariance[7]  = 0.05
                pc.pose.covariance[35] = 0.05
            self.lcov_pub.publish(pc)
        except TransformException:
            pass

    def _cb_gt(self, msg):
        if self._robot_name in msg.name:
            idx = msg.name.index(self._robot_name)
            p = msg.pose[idx].position
            self._gt_pos = np.array([p.x, p.y, p.z])

            ps = PoseStamped()
            ps.header.stamp    = self.get_clock().now().to_msg()
            ps.header.frame_id = 'map'
            ps.pose = msg.pose[idx]
            self.gt_pub.publish(ps)

    # ── State vector assembly ─────────────────────────────────────────────────

    def _publish_state(self):
        f0 = float(np.clip(self._matches / MAX_MATCHES, 0.0, 1.0))

        # Reprojection-error feature: mean inlier distance from local bundle
        # adjustment when published; otherwise outlier ratio of the matches.
        if self._inlier_dist > 0.0:
            f1 = float(np.clip(self._inlier_dist / REPROJ_DIST_MAX, 0.0, 1.0))
        else:
            matches = max(self._matches, 1)
            f1 = float(np.clip((matches - self._inliers) / matches, 0.0, 1.0))

        f2 = float(np.clip(self._inliers / max(self._matches, 1), 0.0, 1.0))

        f3 = float(np.clip(self._scan_valid, 0.0, 1.0))

        f4 = float(np.clip(self._cov_trace / COV_TRACE_MAX, 0.0, 1.0))

        disagreement = float(np.linalg.norm(self._visual_pos - self._lidar_pos))
        f5 = float(np.clip(disagreement / MAX_DISAGREEMENT, 0.0, 1.0))

        msg = Float32MultiArray()
        msg.data = [f0, f1, f2, f3, f4, f5]
        self.state_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = SensorQualityExtractor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
