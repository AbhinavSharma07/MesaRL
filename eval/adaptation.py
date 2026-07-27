"""Frozen-weight adaptation proof.

Loads a trained checkpoint, freezes it, and rolls it out on freshly sampled
task instances -- since every bandit instance is drawn anew from a
continuous prior, any batch sampled with a seed never used during training
is automatically "held out." The question this module answers: does the
frozen network's regret *decrease* over the course of an episode, with zero
weight updates? A uniform-random baseline (same task instances, no learning
of any kind) is evaluated alongside it as the null hypothesis -- regret
decreasing only for the trained policy, not the random baseline, is the
direct empirical evidence that the network is running some internal
learning algorithm at inference time.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

import viz_style
from env.bandit_family import sample_batch
from model.transformer_policy import TransformerPolicy, TransformerPolicyConfig
from training.train_meta_rl import TrainingConfig, collect_rollout


def load_checkpoint(path) -> tuple[TransformerPolicy, TrainingConfig]:
    checkpoint = torch.load(Path(path), map_location="cpu")
    config = TrainingConfig(**checkpoint["config"])
    model_config = TransformerPolicyConfig(
        num_arms=config.num_arms,
        max_trials=config.num_trials,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ff=config.d_ff,
        dropout=config.dropout,
    )
    policy = TransformerPolicy(model_config)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()
    return policy, config


def random_policy_regret(
    mode: str, num_episodes: int, num_arms: int, num_trials: int, seed: int
) -> np.ndarray:
    """Regret curve for a policy that ignores all history and picks a
    uniformly random arm every trial -- the "did not learn anything" null."""
    rng = np.random.default_rng(seed)
    task_batch = sample_batch(mode, rng, num_episodes, num_arms, num_trials)
    regret = np.zeros((num_episodes, num_trials))
    for t in range(num_trials):
        actions = rng.integers(0, num_arms, size=num_episodes)
        means_t = task_batch.means_at(t)
        chosen_mean = means_t[np.arange(num_episodes), actions]
        regret[:, t] = task_batch.optimal_mean_at(t) - chosen_mean
    return regret


def evaluate_adaptation(
    policy: TransformerPolicy,
    config: TrainingConfig,
    mode: str = "train",
    num_episodes: int = 1000,
    seed: int = 12345,
    greedy: bool = False,
) -> np.ndarray:
    """(num_episodes, num_trials) regret curve for the frozen policy on
    freshly sampled task instances."""
    device = torch.device("cpu")
    rng = np.random.default_rng(seed)
    task_batch = sample_batch(mode, rng, num_episodes, config.num_arms, config.num_trials)
    rollout = collect_rollout(policy, task_batch, device, rng, greedy=greedy)
    return rollout.regret


def summarize_regret(regret: np.ndarray, first_n: int = 10, last_n: int = 10) -> dict:
    first = float(regret[:, :first_n].mean())
    last = float(regret[:, -last_n:].mean())
    improvement_ratio = (first - last) / first if first > 1e-8 else 0.0
    return {
        "mean_regret_first_n": first,
        "mean_regret_last_n": last,
        "improvement_ratio": improvement_ratio,
        "mean_cumulative_regret": float(regret.sum(axis=1).mean()),
    }


def plot_regret_curves(curves: dict[str, np.ndarray], save_path, title: str) -> None:
    fig, ax = viz_style.new_figure()
    colors = [viz_style.SERIES_BLUE, viz_style.SERIES_ORANGE, viz_style.SERIES_AQUA, viz_style.SERIES_VIOLET]

    max_x = 0
    for (name, regret), color in zip(curves.items(), colors):
        mean_regret = regret.mean(axis=0)
        trials = np.arange(1, len(mean_regret) + 1)
        max_x = max(max_x, trials[-1])
        ax.plot(trials, mean_regret, color=color, linewidth=2)
        ax.annotate(
            name,
            xy=(trials[-1], mean_regret[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            color=color,
            fontsize=9,
            va="center",
            fontweight="bold",
        )

    ax.set_xlim(1, max_x * 1.18)
    ax.set_xlabel("Trial within episode")
    ax.set_ylabel("Mean regret (optimal arm's true mean - chosen arm's true mean)")
    ax.set_title(title)
    viz_style.save(fig, save_path)


def run_adaptation_eval(
    run_dir: Path, num_episodes: int = 1000, seed: int = 12345
) -> dict:
    run_dir = Path(run_dir)
    policy, config = load_checkpoint(run_dir / "checkpoint.pt")

    trained_regret = evaluate_adaptation(policy, config, num_episodes=num_episodes, seed=seed)
    random_regret = random_policy_regret(
        "train", num_episodes, config.num_arms, config.num_trials, seed=seed + 1
    )

    summary = {
        "trained_policy": summarize_regret(trained_regret),
        "random_baseline": summarize_regret(random_regret),
    }

    with open(run_dir / "adaptation_eval.json", "w") as f:
        json.dump(summary, f, indent=2)

    plot_regret_curves(
        {"Trained policy": trained_regret, "Random baseline": random_regret},
        run_dir / "adaptation_regret_curve.png",
        title="Within-episode regret on held-out tasks (frozen weights)",
    )
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate frozen-weight within-episode adaptation")
    parser.add_argument("--run-dir", type=str, default="runs/main")
    parser.add_argument("--num-episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    summary = run_adaptation_eval(Path(args.run_dir), args.num_episodes, args.seed)
    print(json.dumps(summary, indent=2))
