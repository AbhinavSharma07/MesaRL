"""Evaluates the frozen trained policy on distribution-shifted bandit
families to quantify where the in-episode adaptation strategy breaks down."""

import json
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch

import viz_style
from env.bandit_family import sample_batch
from eval.adaptation import evaluate_adaptation, summarize_regret
from model.transformer_policy import TransformerPolicy
from training.train_meta_rl import TrainingConfig, collect_rollout

SHIFT_MODES = ["train", "ood_nonstationary", "ood_correlated", "ood_wide_prior"]


def evaluate_all_modes(
    policy: TransformerPolicy, config: TrainingConfig, num_episodes: int = 1000, seed: int = 777
) -> dict:
    return {
        mode: evaluate_adaptation(policy, config, mode=mode, num_episodes=num_episodes, seed=seed)
        for mode in SHIFT_MODES
    }


def nonstationary_shock_curve(
    policy: TransformerPolicy,
    config: TrainingConfig,
    num_episodes: int = 1000,
    seed: int = 778,
    window: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Aligns regret to trials relative to each episode's change point
    (offset = t - change_point) and averages -- shows the shock at the
    change point and whether the policy re-adapts afterward."""
    device = torch.device("cpu")
    rng = np.random.default_rng(seed)
    task_batch = sample_batch("ood_nonstationary", rng, num_episodes, config.num_arms, config.num_trials)
    rollout = collect_rollout(policy, task_batch, device, rng)

    change_points = task_batch.change_points
    T = config.num_trials
    offsets = np.arange(-window, window + 1)
    aligned_regret = np.full((num_episodes, len(offsets)), np.nan)

    for i, offset in enumerate(offsets):
        t = change_points + offset
        valid = (t >= 0) & (t < T)
        aligned_regret[valid, i] = rollout.regret[np.arange(num_episodes)[valid], t[valid]]

    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_curve = np.nanmean(aligned_regret, axis=0)
    return offsets, mean_curve


def plot_shift_comparison(results: dict, save_path) -> None:
    fig, ax = viz_style.new_figure()
    colors = [viz_style.SERIES_BLUE, viz_style.SERIES_ORANGE, viz_style.SERIES_AQUA, viz_style.SERIES_YELLOW]

    max_x = 0
    for (mode, regret), color in zip(results.items(), colors):
        mean_regret = regret.mean(axis=0)
        trials = np.arange(1, len(mean_regret) + 1)
        max_x = max(max_x, trials[-1])
        ax.plot(trials, mean_regret, color=color, linewidth=2)
        ax.annotate(
            mode, xy=(trials[-1], mean_regret[-1]), xytext=(6, 0), textcoords="offset points",
            color=color, fontsize=9, va="center", fontweight="bold",
        )

    ax.set_xlim(1, max_x * 1.25)
    ax.set_xlabel("Trial within episode")
    ax.set_ylabel("Mean regret")
    ax.set_title("Regret under distribution shift vs. training distribution")
    viz_style.save(fig, save_path)


def plot_shock_recovery(offsets: np.ndarray, mean_curve: np.ndarray, save_path) -> None:
    fig, ax = viz_style.new_figure()
    ax.axvline(0, color=viz_style.BASELINE, linewidth=1.5, linestyle="--")
    ax.plot(offsets, mean_curve, color=viz_style.SERIES_BLUE, linewidth=2)
    ax.annotate(
        "change point", xy=(0, mean_curve.max()), xytext=(4, 0), textcoords="offset points",
        color=viz_style.INK_SECONDARY, fontsize=9,
    )
    ax.set_xlabel("Trials relative to the mid-episode change point")
    ax.set_ylabel("Mean regret")
    ax.set_title("Shock and recovery around a non-stationary change point")
    viz_style.save(fig, save_path)


def run_distribution_shift_eval(
    run_dir: Path, num_episodes: int = 1000, seed: int = 777, shock_window: int = 20
) -> dict:
    from eval.adaptation import load_checkpoint

    run_dir = Path(run_dir)
    policy, config = load_checkpoint(run_dir / "checkpoint.pt")

    results = evaluate_all_modes(policy, config, num_episodes=num_episodes, seed=seed)
    summary = {mode: summarize_regret(regret) for mode, regret in results.items()}

    offsets, shock_curve = nonstationary_shock_curve(
        policy, config, num_episodes=num_episodes, seed=seed + 1, window=shock_window
    )
    shock_at_change = float(shock_curve[shock_window])
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        shock_before = float(np.nanmean(shock_curve[:shock_window]))
        shock_recovery_tail = float(np.nanmean(shock_curve[-5:]))

    output = {
        "per_mode": summary,
        "nonstationary_shock": {
            "regret_just_before_change": shock_before,
            "regret_at_change": shock_at_change,
            "regret_recovery_tail": shock_recovery_tail,
            "recovers_toward_pre_change_level": shock_recovery_tail < shock_at_change,
        },
    }

    with open(run_dir / "distribution_shift.json", "w") as f:
        json.dump(output, f, indent=2)

    plot_shift_comparison(results, run_dir / "distribution_shift_curves.png")
    plot_shock_recovery(offsets, shock_curve, run_dir / "distribution_shift_shock.png")
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate the trained policy under distribution shift")
    parser.add_argument("--run-dir", type=str, default="runs/main")
    parser.add_argument("--num-episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=777)
    args = parser.parse_args()

    summary = run_distribution_shift_eval(Path(args.run_dir), args.num_episodes, args.seed)
    print(json.dumps(summary, indent=2))
