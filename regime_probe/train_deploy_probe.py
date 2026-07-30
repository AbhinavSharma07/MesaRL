"""Regime-probe: can a linear probe decode "is this a training-like or
held-out-signature episode" from early activations, and does behavior
diverge qualitatively (not just degrade)? A controlled toy testbed for the
train/deploy-detection component some deceptive-alignment concerns hinge on.
"""

import json
from pathlib import Path

import numpy as np
import torch

import viz_style
from analysis.probes import ProbeDataset, best_probe_layer, probe_all_layers
from env.bandit_family import sample_batch
from model.transformer_policy import TransformerPolicy
from training.train_meta_rl import TrainingConfig, collect_rollout

REGIME_MODES = {"regime_low_noise": 0, "regime_high_noise": 1}


def collect_regime_dataset(
    policy: TransformerPolicy,
    config: TrainingConfig,
    num_episodes_per_regime: int = 200,
    early_window: int = 15,
    seed: int = 0,
) -> tuple[ProbeDataset, dict, np.ndarray]:
    """Trial 0 carries zero regime information (no reward observed yet --
    purely a deterministic function of the sentinel token), so it's excluded
    from the probe dataset; the window used is trials [1, early_window]."""
    device = torch.device("cpu")
    rng = np.random.default_rng(seed)
    n_layers = config.n_layers
    activations_per_layer = [[] for _ in range(n_layers)]
    label_chunks = []
    group_chunks = []
    behavior = {}
    next_episode_id = 0

    for mode, label in REGIME_MODES.items():
        task_batch = sample_batch(mode, rng, num_episodes_per_regime, config.num_arms, config.num_trials)
        rollout = collect_rollout(policy, task_batch, device, rng)
        with torch.no_grad():
            logits, _, activations = policy(rollout.hist_actions, rollout.hist_rewards, return_activations=True)

        probs = torch.softmax(logits[:, 1 : early_window + 1, :], dim=-1)
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(-1)
        distinct_arms = [
            len(set(rollout.actions_taken[b, 1 : early_window + 1].tolist())) for b in range(num_episodes_per_regime)
        ]
        behavior[mode] = {
            "mean_action_entropy": entropy.mean().item(),
            "mean_distinct_arms_tried": float(np.mean(distinct_arms)),
        }

        for layer_idx in range(n_layers):
            resid = activations["resid_per_layer"][layer_idx][:, 1 : early_window + 1, :].numpy()
            activations_per_layer[layer_idx].append(resid.reshape(-1, resid.shape[-1]))
        label_chunks.append(np.full(num_episodes_per_regime * early_window, label))
        episode_ids = next_episode_id + np.arange(num_episodes_per_regime)
        group_chunks.append(np.repeat(episode_ids, early_window))
        next_episode_id += num_episodes_per_regime

    activations_per_layer = [
        np.concatenate(chunks, axis=0).astype(np.float32) for chunks in activations_per_layer
    ]
    labels = np.concatenate(label_chunks)
    groups = np.concatenate(group_chunks)
    dataset = ProbeDataset(activations_per_layer=activations_per_layer, labels=labels)
    return dataset, behavior, groups


def plot_regime_probe_accuracy(probe_results: dict, save_path) -> None:
    fig, ax = viz_style.new_figure()
    layers = list(probe_results.keys())
    x = np.arange(len(layers))
    width = 0.25

    series = [
        ("Probe accuracy", "val_accuracy", viz_style.SERIES_BLUE),
        ("Majority baseline", "majority_baseline_accuracy", viz_style.SERIES_ORANGE),
        ("Shuffled-label control", "shuffled_label_accuracy", viz_style.SERIES_AQUA),
    ]
    for i, (label, key, color) in enumerate(series):
        values = [probe_results[layer][key] for layer in layers]
        ax.bar(x + (i - 1) * width, values, width=width, color=color, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.set_ylabel("Accuracy")
    ax.set_title("Decoding train-like vs held-out-signature regime from early activations")
    ax.legend(frameon=False, fontsize=9)
    viz_style.save(fig, save_path)


def run_regime_probe_analysis(
    run_dir: Path, num_episodes_per_regime: int = 200, early_window: int = 15, seed: int = 0
) -> dict:
    from eval.adaptation import load_checkpoint

    run_dir = Path(run_dir)
    policy, config = load_checkpoint(run_dir / "checkpoint.pt")

    dataset, behavior, groups = collect_regime_dataset(
        policy, config, num_episodes_per_regime=num_episodes_per_regime, early_window=early_window, seed=seed
    )
    probe_results = probe_all_layers(dataset, num_classes=2, seed=seed, groups=groups)
    best_layer = best_probe_layer(probe_results)

    entropy_gap = (
        behavior["regime_high_noise"]["mean_action_entropy"] - behavior["regime_low_noise"]["mean_action_entropy"]
    )
    distinct_arms_gap = (
        behavior["regime_high_noise"]["mean_distinct_arms_tried"]
        - behavior["regime_low_noise"]["mean_distinct_arms_tried"]
    )

    summary = {
        "probe_results": probe_results,
        "best_layer": best_layer,
        "behavior_by_regime": behavior,
        "entropy_gap_high_minus_low": entropy_gap,
        "distinct_arms_gap_high_minus_low": distinct_arms_gap,
    }

    with open(run_dir / "regime_probe.json", "w") as f:
        json.dump(summary, f, indent=2)
    plot_regime_probe_accuracy(probe_results, run_dir / "regime_probe_accuracy.png")
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Probe for a train/deploy-regime-distinguishing signal")
    parser.add_argument("--run-dir", type=str, default="runs/main")
    parser.add_argument("--num-episodes-per-regime", type=int, default=200)
    parser.add_argument("--early-window", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    summary = run_regime_probe_analysis(
        Path(args.run_dir), args.num_episodes_per_regime, args.early_window, args.seed
    )
    print(json.dumps(summary, indent=2))
