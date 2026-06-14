#!/usr/bin/env python3
"""
evaluate.py
Runs 200 evaluation trajectories: 4 worlds × 5 configurations × 10 runs.

Configurations:
  B1 — vision-only     (alpha=1.0, fixed)
  B2 — LiDAR-only      (alpha=0.0, fixed)
  B3 — equal-weight    (alpha=0.5, fixed)
  B4 — EKF baseline    (robot_localization fusing both SLAM poses,
                        recorded from /odometry/filtered)
  B5 — PPO agent       (trained model drives /fusion/alpha;
                        the static alpha publisher is disabled)

Each trajectory saves:
  estimated.txt / ground_truth.txt — TUM format, for evo ATE/RPE
  states.csv — t, f0..f5, alpha at the estimate rate, for the §3.7
               interpretability analysis (alpha(t) plots + Integrated
               Gradients on real states)

Usage:
    python3 evaluate.py --model ~/thesis_ws/models/ppo_fusion_final.zip
"""

import argparse
import os
import sys
import subprocess
import time
import threading
from datetime import datetime

import numpy as np

import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, Float32MultiArray

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from adaptive_fusion.fusion_env import WORLDS

CONFIGS = {
    'B1_vision_only':  {'alpha': 1.0,  'use_agent': False, 'ekf': False},
    'B2_lidar_only':   {'alpha': 0.0,  'use_agent': False, 'ekf': False},
    'B3_equal_weight': {'alpha': 0.5,  'use_agent': False, 'ekf': False},
    'B4_ekf':          {'alpha': 0.5,  'use_agent': False, 'ekf': True},
    'B5_ppo_agent':    {'alpha': None, 'use_agent': True,  'ekf': False},
}

RUNS_PER_CONDITION = 10
EPISODE_SIM_TIME   = 60.0  # record 60 s of sim time (§3.4), regardless of
                           # the estimate topic's actual publish rate
MAX_EST_ROWS       = 1200  # hard cap as a safety net
STARTUP_WAIT       = 10.0  # before attaching the recorder
RECORD_TIMEOUT     = 180.0 # covers ~30 s pipeline init + 60 s episode + margin


class TrajRecorder(Node):
    """Records the estimate, ground truth, state vector and alpha."""

    def __init__(self, out_path: str, use_ekf: bool):
        super().__init__('traj_recorder')
        self._out_path   = out_path
        self._est_rows   = []
        self._gt_rows    = []
        self._state_rows = []
        self._state      = np.zeros(6, dtype=np.float32)
        self._alpha      = float('nan')
        self._lock       = threading.Lock()
        self._done       = threading.Event()

        qos = QoSProfile(depth=5,
                         reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE)

        if use_ekf:
            self.create_subscription(Odometry, '/odometry/filtered',
                                     self._cb_est_odom, qos)
        else:
            self.create_subscription(PoseStamped, '/fusion/fused_pose',
                                     self._cb_est_pose, qos)
        self.create_subscription(PoseStamped, '/fusion/ground_truth',
                                 self._cb_gt, qos)
        self.create_subscription(Float32MultiArray, '/fusion/state',
                                 self._cb_state, qos)
        self.create_subscription(Float32, '/fusion/alpha',
                                 self._cb_alpha, 10)

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _cb_state(self, msg):
        self._state = np.array(msg.data, dtype=np.float32)

    def _cb_alpha(self, msg):
        self._alpha = float(msg.data)

    def _record_est(self, stamp, pose):
        t = stamp.sec + stamp.nanosec * 1e-9
        p, q = pose.position, pose.orientation
        with self._lock:
            self._est_rows.append([t, p.x, p.y, p.z, q.x, q.y, q.z, q.w])
            self._state_rows.append([t, *self._state.tolist(), self._alpha])
            elapsed = t - self._est_rows[0][0]
            if elapsed >= EPISODE_SIM_TIME or len(self._est_rows) >= MAX_EST_ROWS:
                self._done.set()

    def _cb_est_pose(self, msg):
        self._record_est(msg.header.stamp, msg.pose)

    def _cb_est_odom(self, msg):
        self._record_est(msg.header.stamp, msg.pose.pose)

    def _cb_gt(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        p, q = msg.pose.position, msg.pose.orientation
        with self._lock:
            self._gt_rows.append([t, p.x, p.y, p.z, q.x, q.y, q.z, q.w])

    # ── Output ────────────────────────────────────────────────────────────────

    def wait(self, timeout=RECORD_TIMEOUT):
        return self._done.wait(timeout=timeout)

    def save(self):
        os.makedirs(self._out_path, exist_ok=True)
        self._write_tum(os.path.join(self._out_path, 'estimated.txt'),
                        self._est_rows)
        self._write_tum(os.path.join(self._out_path, 'ground_truth.txt'),
                        self._gt_rows)
        with open(os.path.join(self._out_path, 'states.csv'), 'w') as f:
            f.write('t,f0,f1,f2,f3,f4,f5,alpha\n')
            for r in self._state_rows:
                f.write(','.join(f'{v:.6f}' for v in r) + '\n')

    @staticmethod
    def _write_tum(path: str, rows: list):
        with open(path, 'w') as f:
            f.write('# timestamp tx ty tz qx qy qz qw\n')
            for r in rows:
                f.write(' '.join(f'{v:.6f}' for v in r) + '\n')


def run_trajectory(world: str, config_name: str, config: dict,
                   run_idx: int, out_dir: str, model=None):
    """Launch simulation, record one trajectory, save TUM files."""
    traj_dir = os.path.join(out_dir, world, config_name, f'run_{run_idx:02d}')
    os.makedirs(traj_dir, exist_ok=True)

    seed = run_idx * 100 + WORLDS.index(world) * 10

    launch_cmd = [
        'ros2', 'launch', 'adaptive_fusion', 'evaluate.launch.py',
        f'world:={world}',
        f'seed:={seed}',
        f'alpha:={config["alpha"] if config["alpha"] is not None else 0.5}',
        f'use_fixed_alpha:={"false" if config["use_agent"] else "true"}',
        f'ekf:={"true" if config["ekf"] else "false"}',
        'headless:=true',
    ]
    proc = subprocess.Popen(launch_cmd,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(STARTUP_WAIT)

    if not rclpy.ok():
        rclpy.init()

    recorder = TrajRecorder(traj_dir, use_ekf=config['ekf'])
    spin_thread = threading.Thread(
        target=rclpy.spin, args=(recorder,), daemon=True)
    spin_thread.start()

    # B5: the trained policy maps /fusion/state -> /fusion/alpha
    alpha_node = None
    if config['use_agent'] and model is not None:
        alpha_node = rclpy.create_node('alpha_publisher_eval')
        alpha_pub  = alpha_node.create_publisher(Float32, '/fusion/alpha', 10)

        def cb_state(msg):
            state = np.array(msg.data, dtype=np.float32)
            action, _ = model.predict(state, deterministic=True)
            alpha_pub.publish(Float32(data=float(action[0])))

        alpha_node.create_subscription(
            Float32MultiArray, '/fusion/state', cb_state, 5)
        agent_thread = threading.Thread(
            target=rclpy.spin, args=(alpha_node,), daemon=True)
        agent_thread.start()

    completed = recorder.wait()
    recorder.save()

    print(f'  [{world}][{config_name}][run {run_idx}] '
          f'{"OK" if completed else "TIMEOUT"} — {len(recorder._est_rows)} poses')

    recorder.destroy_node()
    if alpha_node is not None:
        alpha_node.destroy_node()

    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    # A deadlocked gzserver can survive SIGTERM and poison the next launch.
    subprocess.run(['pkill', '-9', '-f', 'gzserver'], check=False)
    time.sleep(2.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None,
                        help='Path to trained PPO model .zip')
    parser.add_argument('--out-dir', default=os.path.expanduser('~/thesis_ws/results'))
    parser.add_argument('--worlds', nargs='+', default=WORLDS)
    parser.add_argument('--configs', nargs='+', default=list(CONFIGS.keys()))
    parser.add_argument('--runs', type=int, default=RUNS_PER_CONDITION)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(args.out_dir, f'eval_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)

    model = None
    if args.model:
        model_path = args.model if args.model.endswith('.zip') else args.model + '.zip'
        if os.path.exists(model_path):
            from stable_baselines3 import PPO
            model = PPO.load(args.model)
            print(f'Loaded model from {model_path}')
        else:
            print(f'Model not found: {model_path}')

    total = len(args.worlds) * len(args.configs) * args.runs
    done  = 0

    for world in args.worlds:
        for cfg_name in args.configs:
            if cfg_name not in CONFIGS:
                continue
            cfg = CONFIGS[cfg_name]
            if cfg['use_agent'] and model is None:
                print(f'Skipping {cfg_name} — no model provided')
                continue
            for run in range(args.runs):
                done += 1
                print(f'[{done}/{total}] {world} / {cfg_name} / run {run}')
                run_trajectory(world, cfg_name, cfg, run, run_dir, model)

    print(f'\nEvaluation complete. Results in: {run_dir}')
    print('Next step: python3 analyse_results.py --results-dir', run_dir)


if __name__ == '__main__':
    main()
