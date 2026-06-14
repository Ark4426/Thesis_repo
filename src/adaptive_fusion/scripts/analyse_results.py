#!/usr/bin/env python3
"""
analyse_results.py
Computes ATE/RPE from evaluation trajectories, runs hypothesis tests,
and produces the interpretability analysis (§3.7):
  1. alpha(t) traces over the sensor-quality features with §3.4
     degradation labels shaded
  2. Integrated Gradients per-feature attribution heatmap (Captum),
     computed on the REAL states recorded by evaluate.py (states.csv)

Usage:
    python3 analyse_results.py --results-dir ~/thesis_ws/results/eval_YYYYMMDD_HHMMSS
                               --model ~/thesis_ws/models/ppo_fusion_final.zip
"""

import argparse
import os
import glob
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

# evo for ATE/RPE
from evo.tools import file_interface
from evo.core import metrics, sync
from evo.core.metrics import PoseRelation

# Stable-Baselines3
from stable_baselines3 import PPO

# Integrated Gradients via Captum
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients

CONFIGS   = ['B1_vision_only', 'B2_lidar_only', 'B3_equal_weight',
             'B4_ekf', 'B5_ppo_agent']
WORLDS    = ['w1_static', 'w2_low_dynamic', 'w3_high_dynamic',
             'w4_visually_degraded']
FEAT_NAMES = ['feat_count', 'reproj_err', 'inlier_ratio',
              'scan_quality', 'cov_trace', 'pose_disagree']


def visual_degraded(f: np.ndarray) -> np.ndarray:
    """§3.4: matches < 100 OR mean inlier distance > 0.10 m OR inlier
    ratio < 0.5 — expressed on the normalised features
    (f0 = matches/400, f1 = dist/0.2 m, f2 = ratio)."""
    return (f[:, 0] < 100.0 / 400.0) | (f[:, 1] > 0.10 / 0.2) | (f[:, 2] < 0.5)


def lidar_degraded(f: np.ndarray) -> np.ndarray:
    """§3.4: scan quality < 0.25 OR covariance trace > 0.05 m²
    (f3 = quality, f4 = trace/0.25 m²)."""
    return (f[:, 3] < 0.25) | (f[:, 4] > 0.05 / 0.25)


# ─────────────────────────────────────────────────────────────────────────────
# ATE / RPE via evo
# ─────────────────────────────────────────────────────────────────────────────

def compute_ate(est_file: str, gt_file: str) -> float:
    """Returns ATE RMSE in metres."""
    try:
        traj_est = file_interface.read_tum_trajectory_file(est_file)
        traj_ref = file_interface.read_tum_trajectory_file(gt_file)
        traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)
        traj_est.align(traj_ref, correct_scale=False, correct_only_scale=False)
        metric = metrics.APE(PoseRelation.translation_part)
        metric.process_data((traj_ref, traj_est))
        return metric.get_statistic(metrics.StatisticsType.rmse)
    except Exception as e:
        print(f'    ATE error ({est_file}): {e}')
        return float('nan')


def compute_rpe(est_file: str, gt_file: str, delta: int = 1) -> float:
    """Returns RPE RMSE in metres."""
    try:
        traj_est = file_interface.read_tum_trajectory_file(est_file)
        traj_ref = file_interface.read_tum_trajectory_file(gt_file)
        traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)
        metric = metrics.RPE(PoseRelation.translation_part,
                             delta=delta, delta_unit=metrics.Unit.frames,
                             all_pairs=False)
        metric.process_data((traj_ref, traj_est))
        return metric.get_statistic(metrics.StatisticsType.rmse)
    except Exception as e:
        print(f'    RPE error: {e}')
        return float('nan')


# ─────────────────────────────────────────────────────────────────────────────
# Results aggregation
# ─────────────────────────────────────────────────────────────────────────────

def collect_metrics(results_dir: str) -> pd.DataFrame:
    rows = []
    for world in WORLDS:
        for cfg in CONFIGS:
            run_dirs = sorted(glob.glob(
                os.path.join(results_dir, world, cfg, 'run_*')))
            for rd in run_dirs:
                est = os.path.join(rd, 'estimated.txt')
                gt  = os.path.join(rd, 'ground_truth.txt')
                if not os.path.exists(est) or not os.path.exists(gt):
                    continue
                ate = compute_ate(est, gt)
                rpe = compute_rpe(est, gt)
                rows.append({
                    'world': world, 'config': cfg,
                    'run': os.path.basename(rd),
                    'ate_rmse': ate, 'rpe_rmse': rpe})
                print(f'  {world}/{cfg}/{os.path.basename(rd)}: '
                      f'ATE={ate:.4f}  RPE={rpe:.4f}')
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Hypothesis testing (§3.6)
# ─────────────────────────────────────────────────────────────────────────────

def hypothesis_tests(df: pd.DataFrame, out_dir: str):
    ppo = df[df.config == 'B5_ppo_agent']
    results = []
    for world in WORLDS:
        for baseline in ['B1_vision_only', 'B2_lidar_only',
                         'B3_equal_weight', 'B4_ekf']:
            b   = df[(df.world == world) & (df.config == baseline)]['ate_rmse'].dropna()
            p   = ppo[ppo.world == world]['ate_rmse'].dropna()
            if len(b) < 2 or len(p) < 2:
                continue
            t_stat, p_val = stats.ttest_rel(b.values[:len(p)], p.values[:len(b)])
            mean_b = b.mean()
            mean_p = p.mean()
            reduction = (mean_b - mean_p) / mean_b * 100 if mean_b > 0 else 0
            results.append({
                'world': world, 'baseline': baseline,
                'mean_baseline': mean_b, 'mean_ppo': mean_p,
                'ate_reduction_pct': reduction,
                't_stat': t_stat, 'p_value': p_val,
                'significant': p_val < 0.05,
                'H1_met': reduction >= 15.0 and p_val < 0.05})
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(out_dir, 'hypothesis_tests.csv'), index=False)
    print('\n── Hypothesis Tests ─────────────────────────────')
    print(results_df[['world', 'baseline', 'ate_reduction_pct',
                       'p_value', 'H1_met']].to_string(index=False))

    # H3: ATE variability across worlds per configuration
    h3 = df.groupby('config')['ate_rmse'].agg(['mean', 'std']).round(4)
    h3.to_csv(os.path.join(out_dir, 'h3_variance.csv'))
    print('\n── H3: ATE mean/std across all worlds ───────────')
    print(h3.to_string())
    return results_df


# ─────────────────────────────────────────────────────────────────────────────
# ATE summary table and box plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_ate_boxplot(df: pd.DataFrame, out_dir: str):
    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)
    for ax, world in zip(axes, WORLDS):
        sub = df[df.world == world]
        data = [sub[sub.config == c]['ate_rmse'].dropna().values
                for c in CONFIGS]
        ax.boxplot(data, tick_labels=[c.replace('_', '\n') for c in CONFIGS])
        ax.set_title(world.replace('_', ' '))
        ax.set_ylabel('ATE RMSE (m)')
        ax.grid(axis='y', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'ate_boxplot.png'), dpi=150)
    plt.close()
    print('Saved ate_boxplot.png')


# ─────────────────────────────────────────────────────────────────────────────
# Real recorded states (evaluate.py states.csv)
# ─────────────────────────────────────────────────────────────────────────────

def load_states(results_dir: str, world: str, config: str = 'B5_ppo_agent'):
    """Returns list of (t, states[N,6], alpha[N]) per run."""
    out = []
    for rd in sorted(glob.glob(os.path.join(results_dir, world, config, 'run_*'))):
        csv = os.path.join(rd, 'states.csv')
        if not os.path.exists(csv):
            continue
        d = pd.read_csv(csv)
        if len(d) < 10:
            continue
        t = d['t'].values - d['t'].values[0]
        f = d[['f0', 'f1', 'f2', 'f3', 'f4', 'f5']].values.astype(np.float32)
        a = d['alpha'].values.astype(np.float32)
        out.append((t, f, a))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# alpha(t) traces with degradation shading (§3.7 analysis 1)
# ─────────────────────────────────────────────────────────────────────────────

def plot_alpha_traces(results_dir: str, out_dir: str):
    fig, axes = plt.subplots(len(WORLDS), 1, figsize=(12, 3 * len(WORLDS)),
                             sharex=True)
    plotted = False
    for ax, world in zip(np.atleast_1d(axes), WORLDS):
        runs = load_states(results_dir, world)
        if not runs:
            ax.set_visible(False)
            continue
        t, f, a = runs[0]   # representative run; all runs go into the IG stats
        vd = visual_degraded(f)
        ld = lidar_degraded(f)
        ax.fill_between(t, 0, 1, where=vd, color='tab:red', alpha=0.15,
                        label='visual degraded (§3.4)')
        ax.fill_between(t, 0, 1, where=ld, color='tab:blue', alpha=0.15,
                        label='LiDAR degraded (§3.4)')
        ax.plot(t, a, 'k-', lw=1.2, label='alpha (vision weight)')
        ax.plot(t, f[:, 0], color='tab:green', lw=0.7, alpha=0.7,
                label='f0 feat_count')
        ax.plot(t, f[:, 5], color='tab:purple', lw=0.7, alpha=0.7,
                label='f5 disagreement')
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel('alpha / features')
        ax.set_title(world.replace('_', ' '))
        ax.grid(alpha=0.3)
        plotted = True
    if not plotted:
        plt.close()
        print('No states.csv found — skipping alpha trace plot')
        return
    np.atleast_1d(axes)[0].legend(loc='upper right', fontsize=8, ncol=2)
    np.atleast_1d(axes)[-1].set_xlabel('episode time (s)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'alpha_traces.png'), dpi=150)
    plt.close()
    print('Saved alpha_traces.png')


# ─────────────────────────────────────────────────────────────────────────────
# Integrated Gradients — policy interpretability (§3.7 analysis 2)
# ─────────────────────────────────────────────────────────────────────────────

class _DeterministicPolicy(nn.Module):
    """Differentiable mean action of the SB3 PPO policy (for Captum)."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.policy.extract_features(
            obs, self.policy.features_extractor)
        latent_pi = self.policy.mlp_extractor.forward_actor(features)
        return self.policy.action_net(latent_pi)


def integrated_gradients_analysis(model_path: str, results_dir: str,
                                  out_dir: str):
    """Per-feature IG attributions on the real recorded evaluation states."""
    if not os.path.exists(model_path.replace('.zip', '') + '.zip'):
        print(f'Model not found at {model_path} — skipping IG analysis')
        return

    model  = PPO.load(model_path)
    model.policy.set_training_mode(False)
    wrapper = _DeterministicPolicy(model.policy)
    ig = IntegratedGradients(wrapper)

    ig_scores = {}
    for world in WORLDS:
        runs = load_states(results_dir, world)
        if not runs:
            continue
        states_np = np.concatenate([f for _, f, _ in runs], axis=0)
        states_t  = torch.tensor(states_np, dtype=torch.float32)
        baseline  = torch.zeros_like(states_t)
        attr = ig.attribute(states_t, baselines=baseline, n_steps=50)
        ig_scores[world] = attr.abs().mean(0).detach().numpy()

    if not ig_scores:
        print('No states.csv found — skipping IG analysis')
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    matrix = np.array([ig_scores.get(w, np.zeros(6)) for w in WORLDS])
    im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd')
    ax.set_xticks(range(6))
    ax.set_xticklabels(FEAT_NAMES, rotation=30, ha='right')
    ax.set_yticks(range(len(WORLDS)))
    ax.set_yticklabels([w.replace('_', ' ') for w in WORLDS])
    plt.colorbar(im, ax=ax, label='Mean |IG attribution|')
    ax.set_title('Integrated Gradients: feature influence on α per world')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'ig_attribution_heatmap.png'), dpi=150)
    plt.close()
    print('Saved ig_attribution_heatmap.png')

    pd.DataFrame(ig_scores, index=FEAT_NAMES).to_csv(
        os.path.join(out_dir, 'ig_scores.csv'))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', required=True)
    parser.add_argument('--model', default=None)
    parser.add_argument('--out-dir', default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.results_dir, 'analysis')
    os.makedirs(out_dir, exist_ok=True)

    print('── Collecting ATE/RPE metrics ──────────────────')
    df = collect_metrics(args.results_dir)
    df.to_csv(os.path.join(out_dir, 'metrics.csv'), index=False)

    print('\n── Summary table ───────────────────────────────')
    summary = df.groupby(['world', 'config'])['ate_rmse'].agg(
        ['mean', 'std', 'count']).round(4)
    print(summary.to_string())
    summary.to_csv(os.path.join(out_dir, 'ate_summary.csv'))

    print('\n── Hypothesis tests (§3.6) ─────────────────────')
    hypothesis_tests(df, out_dir)

    print('\n── ATE box plot ────────────────────────────────')
    plot_ate_boxplot(df, out_dir)

    print('\n── alpha(t) traces (§3.7) ──────────────────────')
    plot_alpha_traces(args.results_dir, out_dir)

    if args.model:
        print('\n── Integrated Gradients (§3.7) ─────────────────')
        integrated_gradients_analysis(args.model, args.results_dir, out_dir)

    print(f'\nAll outputs saved to: {out_dir}')


if __name__ == '__main__':
    main()
