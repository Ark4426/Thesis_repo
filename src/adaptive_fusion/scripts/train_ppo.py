#!/usr/bin/env python3
"""
train_ppo.py
Trains the PPO fusion-weight agent for 2 million environment steps.

Run from the thesis_ws root:
    python3 src/adaptive_fusion/scripts/train_ppo.py

The simulation.launch.py must be running before this script is started,
OR pass --launch to have this script manage the simulation subprocess.
"""

import argparse
import os
import sys
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback, BaseCallback)
from stable_baselines3.common.monitor import Monitor

# Add package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from adaptive_fusion.fusion_env import FusionEnv, WORLDS


class EpisodeLogCallback(BaseCallback):
    """Logs mean reward and alpha statistics every N episodes."""

    def __init__(self, log_freq: int = 50, verbose: int = 0):
        super().__init__(verbose)
        self._log_freq = log_freq
        self._ep_rewards: list = []
        self._alphas: list = []

    def _on_step(self) -> bool:
        info = self.locals.get('infos', [{}])[0]
        if 'alpha' in info:
            self._alphas.append(info['alpha'])

        dones = self.locals.get('dones', [False])
        rewards = self.locals.get('rewards', [0.0])
        self._ep_rewards.extend(rewards)

        if any(dones) and len(self._ep_rewards) >= self._log_freq:
            mean_r = float(np.mean(self._ep_rewards))
            mean_a = float(np.mean(self._alphas)) if self._alphas else float('nan')
            self.logger.record('rollout/mean_ep_reward', mean_r)
            self.logger.record('rollout/mean_alpha', mean_a)
            self._ep_rewards.clear()
            self._alphas.clear()

        return True


def main():
    parser = argparse.ArgumentParser(description='Train PPO fusion agent')
    parser.add_argument('--worlds', nargs='+', default=list(WORLDS),
                        choices=WORLDS,
                        help='Worlds sampled uniformly during training '
                             '(§3.3). Pass a single world to disable '
                             'randomisation.')
    parser.add_argument('--world-block', type=int, default=10,
                        help='Episodes per sampled world before the '
                             'simulation is relaunched with a new world.')
    parser.add_argument('--total-steps', type=int, default=2_000_000)
    parser.add_argument('--save-freq', type=int, default=10_000,
                        help='Steps between checkpoints. Smaller = less work '
                             'lost when stopping/resuming across days '
                             '(10k steps ~= 27 min wall-clock at ~0.16 s/step).')
    parser.add_argument('--save-dir', default=os.path.expanduser('~/thesis_ws/models'))
    parser.add_argument('--seed', type=int, default=0,
                        help='RNG seed. Default 0 gives a balanced world draw '
                             'at block=10 (12 easy / 8 hard, first easy block '
                             '3); seed 42 drew 85%% hard worlds and stalled '
                             'pilot-2. Re-check with check_seeds.py if '
                             'world-block changes.')
    parser.add_argument('--headless', action='store_true', default=True)
    parser.add_argument('--resume-from', default=None,
                        help='Checkpoint .zip to resume from; continues the '
                             'step count toward --total-steps.')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    log_dir = os.path.join(args.save_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    print(f'[train_ppo] worlds={args.worlds}  block={args.world_block}  '
          f'steps={args.total_steps:,}')
    print(f'[train_ppo] save_dir={args.save_dir}')

    # ── Environment ───────────────────────────────────────────────────────────
    if len(args.worlds) == 1:
        fusion_env = FusionEnv(world=args.worlds[0], headless=args.headless)
    else:
        fusion_env = FusionEnv(worlds=args.worlds,
                               world_block=args.world_block,
                               headless=args.headless)
    env = Monitor(fusion_env, filename=os.path.join(log_dir, 'monitor'))

    # ── PPO model (§3.3 hyperparameters) ─────────────────────────────────────
    if args.resume_from:
        model = PPO.load(args.resume_from, env=env, tensorboard_log=log_dir)
        reset_timesteps = False
        print(f'[train_ppo] Resumed from {args.resume_from} '
              f'at {model.num_timesteps:,} steps.')
    else:
        model = PPO(
            policy='MlpPolicy',
            env=env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            # §3.3 deviation: entropy bonus added after pilot-2 plateaued at
            # the reward floor (seed-42 hard-world draw + saturated reward →
            # advantages collapsed to ~0). Keeps exploration alive so the
            # policy can escape the always-vision local optimum.
            ent_coef=0.05,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=[64, 64]),  # two hidden layers, 64 units
            tensorboard_log=log_dir,
            seed=args.seed,
            verbose=1,
        )
        reset_timesteps = True
        print('[train_ppo] PPO model created.')
        print(f'[train_ppo] Policy: {model.policy}')

    # ── Callbacks ─────────────────────────────────────────────────────────────
    checkpoint_cb = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=os.path.join(args.save_dir, 'checkpoints'),
        name_prefix='ppo_fusion')
    print(f'[train_ppo] checkpoint every {args.save_freq:,} steps '
          f'(~{args.save_freq * 0.16 / 60:.0f} min wall-clock)')

    episode_log_cb = EpisodeLogCallback(log_freq=50)

    # ── Train ─────────────────────────────────────────────────────────────────
    t0 = time.time()
    model.learn(
        total_timesteps=args.total_steps,
        callback=[checkpoint_cb, episode_log_cb],
        progress_bar=True,
        reset_num_timesteps=reset_timesteps,
    )
    elapsed = time.time() - t0

    # ── Save final model ──────────────────────────────────────────────────────
    final_path = os.path.join(args.save_dir, 'ppo_fusion_final')
    model.save(final_path)
    print(f'\n[train_ppo] Training complete in {elapsed/3600:.1f} h')
    print(f'[train_ppo] Final model saved to {final_path}.zip')

    env.close()


if __name__ == '__main__':
    main()
