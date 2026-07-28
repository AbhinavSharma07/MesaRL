"""Compares the trained network's actual per-trial action distribution
against each hand-designed candidate algorithm (analysis/candidate_algorithms),
using the SAME realized (action, reward) history for both. This answers
"which known bandit algorithm does the emergent mesa-optimizer's behavior
most resemble, and where does it diverge?"

Candidates are evaluated in "shadow" mode: they never drive their own
trajectory, only predict what they would have done given the network's
actual realized history -- so a same-trial comparison is meaningful even
though the candidates never got to act.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

import viz_style
from analysis.candidate_algorithms import (
    epsilon_greedy_distribution,
    thompson_sampling_distribution,
    ucb1_distribution,
    win_stay_lose_shift_distribution,
)
from env.bandit_family import TRAIN_NOISE_RANGE, sample_batch
from training.train_meta_rl import RolloutBatch, TrainingConfig, collect_rollout
from model.transformer_policy import TransformerPolicy

ASSUMED_OBS_NOISE_STD = float(np.mean(TRAIN_NOISE_RANGE))


def network_policy_distributions(
    policy: TransformerPolicy, rollout: RolloutBatch, device: torch.device
) -> np.ndarray:
    """(B, T, K) softmax distribution the network actually used at every
    trial -- recomputed via a single causal forward pass over the completed
    episode, which (by the causal-masking invariant tested in
    tests/test_transformer_policy.py) reproduces exactly what was used to
    sample each action during rollout."""
    policy.eval()
    with torch.no_grad():
        logits, _, _ = policy(rollout.hist_actions, rollout.hist_rewards)
    return torch.softmax(logits, dim=-1).cpu().numpy()


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """KL(p || q) along the last axis, elementwise-safe against zeros --
    this direction handles near-deterministic candidates (UCB1,
    win-stay-lose-shift) gracefully: a zero-probability entry in p simply
    contributes 0 regardless of q, rather than blowing up to infinity the
    way KL(network || candidate) would whenever the network places any mass
    where a deterministic candidate places none."""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return (p * np.log(p / q)).sum(axis=-1)


def build_candidate_distributions(actions_np: np.ndarray, rewards_np: np.ndarray, num_arms: int) -> dict:
    return {
        "Thompson sampling": thompson_sampling_distribution(
            actions_np, rewards_np, num_arms, assumed_obs_noise_std=ASSUMED_OBS_NOISE_STD
        ),
        "UCB1": ucb1_distribution(actions_np, rewards_np, num_arms),
        "Epsilon-greedy": epsilon_greedy_distribution(actions_np, rewards_np, num_arms),
        "Win-stay-lose-shift": win_stay_lose_shift_distribution(actions_np, rewards_np, num_arms),
    }


def compare_to_candidates(
    policy: TransformerPolicy,
    config: TrainingConfig,
    num_episodes: int = 300,
    seed: int = 999,
) -> dict:
    """Returns {candidate_name: {"mean_kl_per_trial": (T,), "mean_kl_overall":
    float, "action_agreement_rate": float}}, plus the raw network/candidate
    per-trial action-distribution arrays are not returned (only summaries) to
    keep this cheap to call repeatedly."""
    device = torch.device("cpu")
    rng = np.random.default_rng(seed)
    task_batch = sample_batch("train", rng, num_episodes, config.num_arms, config.num_trials)
    rollout = collect_rollout(policy, task_batch, device, rng)

    network_dist = network_policy_distributions(policy, rollout, device)
    actions_np = rollout.actions_taken.cpu().numpy()
    rewards_np = rollout.rewards_obtained.cpu().numpy()

    candidate_distributions = build_candidate_distributions(actions_np, rewards_np, config.num_arms)

    results = {}
    for name, candidate_dist in candidate_distributions.items():
        kl = kl_divergence(candidate_dist, network_dist)  # (B, T)
        agreement = (candidate_dist.argmax(axis=-1) == network_dist.argmax(axis=-1)).astype(float)
        results[name] = {
            "mean_kl_per_trial": kl.mean(axis=0).tolist(),
            "mean_kl_overall": float(kl.mean()),
            "action_agreement_rate": float(agreement.mean()),
        }
    return results


def most_similar_candidate(results: dict) -> str:
    return min(results, key=lambda name: results[name]["mean_kl_overall"])


def plot_kl_curves(results: dict, save_path) -> None:
    fig, ax = viz_style.new_figure()
    colors = [
        viz_style.SERIES_BLUE,
        viz_style.SERIES_ORANGE,
        viz_style.SERIES_AQUA,
        viz_style.SERIES_YELLOW,
    ]

    max_x = 0
    for (name, stats), color in zip(results.items(), colors):
        kl_curve = np.array(stats["mean_kl_per_trial"])
        trials = np.arange(1, len(kl_curve) + 1)
        max_x = max(max_x, trials[-1])
        ax.plot(trials, kl_curve, color=color, linewidth=2)
        ax.annotate(
            name,
            xy=(trials[-1], kl_curve[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            color=color,
            fontsize=9,
            va="center",
            fontweight="bold",
        )

    ax.set_xlim(1, max_x * 1.3)
    ax.set_xlabel("Trial within episode")
    ax.set_ylabel("KL(candidate || network) -- lower means more similar")
    ax.set_title("Which known bandit algorithm does the network's behavior resemble?")
    viz_style.save(fig, save_path)


def run_candidate_comparison(run_dir: Path, num_episodes: int = 300, seed: int = 999) -> dict:
    from eval.adaptation import load_checkpoint

    run_dir = Path(run_dir)
    policy, config = load_checkpoint(run_dir / "checkpoint.pt")

    results = compare_to_candidates(policy, config, num_episodes=num_episodes, seed=seed)
    summary = {
        "most_similar_candidate": most_similar_candidate(results),
        "candidates": {
            name: {k: v for k, v in stats.items() if k != "mean_kl_per_trial"}
            for name, stats in results.items()
        },
    }

    with open(run_dir / "candidate_comparison.json", "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    plot_kl_curves(results, run_dir / "candidate_comparison_kl.png")
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare the trained policy to candidate bandit algorithms")
    parser.add_argument("--run-dir", type=str, default="runs/main")
    parser.add_argument("--num-episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=999)
    args = parser.parse_args()

    summary = run_candidate_comparison(Path(args.run_dir), args.num_episodes, args.seed)
    print(json.dumps(summary, indent=2))
