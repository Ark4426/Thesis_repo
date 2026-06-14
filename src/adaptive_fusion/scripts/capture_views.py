#!/usr/bin/env python3
"""
capture_views.py — saves thesis-figure images from a running simulation:
  <prefix>_overview.png   bird's-eye view of the warehouse (overhead camera)
  <prefix>_rgb.png        robot's RGB camera
  <prefix>_depth.png      robot's depth camera (colormapped)
  <prefix>_scan.png       LiDAR scan plotted around the robot

Usage (sim must be running):
    python3 capture_views.py --prefix /tmp/w3
"""
import argparse
import time

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, LaserScan


class Capture(Node):
    def __init__(self, prefix):
        super().__init__('capture_views')
        self.prefix = prefix
        self.done = {'overview': False, 'rgb': False, 'depth': False, 'scan': False}
        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Image, '/overview/image_raw',
                                 lambda m: self._rgb(m, 'overview'), qos)
        self.create_subscription(Image, '/camera/image_raw',
                                 lambda m: self._rgb(m, 'rgb'), qos)
        self.create_subscription(Image, '/camera/depth/image_raw', self._depth, qos)
        self.create_subscription(LaserScan, '/scan', self._scan, qos)

    def _save(self, key, img):
        path = f'{self.prefix}_{key}.png'
        cv2.imwrite(path, img)
        self.done[key] = True
        print(f'saved {path}')

    def _rgb(self, m, key):
        if self.done[key]:
            return
        img = np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.width, -1)
        self._save(key, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    def _depth(self, m):
        if self.done['depth']:
            return
        d = np.frombuffer(bytes(m.data), np.float32).reshape(m.height, m.width).copy()
        d[~np.isfinite(d)] = 0.0
        norm = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        self._save('depth', cv2.applyColorMap(norm, cv2.COLORMAP_TURBO))

    def _scan(self, m):
        if self.done['scan']:
            return
        size, scale = 800, 100.0   # 100 px per metre, 8 m span
        img = np.full((size, size, 3), 255, np.uint8)
        cv2.circle(img, (size // 2, size // 2), 14, (60, 60, 60), -1)
        angles = m.angle_min + np.arange(len(m.ranges)) * m.angle_increment
        for r, a in zip(m.ranges, angles):
            if np.isfinite(r) and m.range_min < r < m.range_max:
                x = int(size / 2 + r * np.cos(a) * scale)
                y = int(size / 2 - r * np.sin(a) * scale)
                if 0 <= x < size and 0 <= y < size:
                    cv2.circle(img, (x, y), 3, (0, 0, 220), -1)
        self._save('scan', img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prefix', default='/tmp/sim')
    args = ap.parse_args()

    rclpy.init()
    node = Capture(args.prefix)
    t0 = time.time()
    while time.time() - t0 < 20 and not all(node.done.values()):
        rclpy.spin_once(node, timeout_sec=0.2)
    for k, ok in node.done.items():
        if not ok:
            print(f'WARNING: no {k} image received')
    rclpy.shutdown()


if __name__ == '__main__':
    main()
