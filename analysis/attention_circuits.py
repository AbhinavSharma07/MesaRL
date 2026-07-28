"""Searches for induction-head-like circuits: does any attention head attend
preferentially to earlier positions where the same arm about to be chosen
was previously played (the bandit-domain analogue of the induction heads
believed to underlie in-context learning in language transformers)? Scored
per (layer, head) as mean attention on same-arm keys minus different-arm
keys."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import viz_style
from env.bandit_family import sample_batch
from model.transformer_policy import TransformerPolicy
from training.train_meta_rl import TrainingConfig, collect_rollout


@dataclass
class AttentionDataset:
    attn_weights_per_layer: list  # list of (B, H, T, T) float32 arrays
    hist_actions: np.ndarray  # (B, T)
    actions_taken: np.ndarray  # (B, T)


def collect_attention_dataset(
    policy: TransformerPolicy, config: TrainingConfig, num_episodes: int, seed: int
) -> AttentionDataset:
    device = torch.device("cpu")
    rng = np.random.default_rng(seed)
    task_batch = sample_batch("train", rng, num_episodes, config.num_arms, config.num_trials)
    rollout = collect_rollout(policy, task_batch, device, rng)

    policy.eval()
    with torch.no_grad():
        _, _, activations = policy(rollout.hist_actions, rollout.hist_rewards, return_activations=True)

    attn_weights_per_layer = [a.numpy() for a in activations["attn_weights_per_layer"]]
    return AttentionDataset(
        attn_weights_per_layer=attn_weights_per_layer,
        hist_actions=rollout.hist_actions.numpy(),
        actions_taken=rollout.actions_taken.numpy(),
    )


def same_arm_attention_bias(dataset: AttentionDataset, min_query_position: int = 5) -> np.ndarray:
    """(n_layers, n_heads): mean attention weight on "same arm as about to
    be chosen" key positions minus mean attention weight on "different arm"
    key positions, averaged over every (episode, query position) pair where
    at least one same-arm key position exists in that episode's history."""
    n_layers = len(dataset.attn_weights_per_layer)
    n_heads = dataset.attn_weights_per_layer[0].shape[1]
    B, T = dataset.hist_actions.shape

    bias_sum = np.zeros((n_layers, n_heads))
    n_valid = 0

    for b in range(B):
        for q in range(min_query_position, T):
            chosen = dataset.actions_taken[b, q]
            key_range = dataset.hist_actions[b, : q + 1]  # positions 0..q
            same_mask = key_range == chosen
            if not same_mask.any():
                continue
            diff_mask = ~same_mask

            weights_stack = np.stack(
                [dataset.attn_weights_per_layer[layer][b, :, q, : q + 1] for layer in range(n_layers)]
            )  # (n_layers, n_heads, q+1)
            mean_same = weights_stack[:, :, same_mask].mean(axis=-1)
            mean_diff = weights_stack[:, :, diff_mask].mean(axis=-1) if diff_mask.any() else 0.0
            bias_sum += mean_same - mean_diff
            n_valid += 1

    return bias_sum / max(n_valid, 1)


def top_induction_head_candidates(bias: np.ndarray, top_k: int = 3) -> list:
    flat_idx = np.argsort(bias.ravel())[::-1][:top_k]
    layer_head_pairs = [np.unravel_index(idx, bias.shape) for idx in flat_idx]
    return [
        {"layer": int(layer), "head": int(head), "same_arm_attention_bias": float(bias[layer, head])}
        for layer, head in layer_head_pairs
    ]


def plot_bias_heatmap(bias: np.ndarray, save_path) -> None:
    import matplotlib.colors as mcolors

    fig, ax = viz_style.new_figure(figsize=(5.5, 4))
    max_abs = max(abs(bias.min()), abs(bias.max()), 1e-6)
    diverging_cmap = mcolors.LinearSegmentedColormap.from_list(
        "blue_gray_red", [viz_style.SERIES_BLUE, "#f0efec", viz_style.SERIES_RED]
    )
    norm = mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

    im = ax.imshow(bias, cmap=diverging_cmap, norm=norm, aspect="auto")
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title("Same-arm attention bias (induction-head search)")
    ax.set_xticks(range(bias.shape[1]))
    ax.set_yticks(range(bias.shape[0]))
    ax.grid(False)

    for layer in range(bias.shape[0]):
        for head in range(bias.shape[1]):
            ax.text(
                head, layer, f"{bias[layer, head]:.2f}", ha="center", va="center",
                color=viz_style.INK_PRIMARY, fontsize=9,
            )

    fig.colorbar(im, ax=ax, shrink=0.8, label="same-arm minus different-arm attention")
    viz_style.save(fig, save_path)


def run_attention_analysis(run_dir: Path, num_episodes: int = 200, seed: int = 0, top_k: int = 3) -> dict:
    import json

    from eval.adaptation import load_checkpoint

    run_dir = Path(run_dir)
    policy, config = load_checkpoint(run_dir / "checkpoint.pt")

    dataset = collect_attention_dataset(policy, config, num_episodes=num_episodes, seed=seed)
    bias = same_arm_attention_bias(dataset)
    candidates = top_induction_head_candidates(bias, top_k=top_k)

    summary = {"bias_matrix": bias.tolist(), "top_candidates": candidates}
    with open(run_dir / "attention_circuits.json", "w") as f:
        json.dump(summary, f, indent=2)
    plot_bias_heatmap(bias, run_dir / "attention_circuits_heatmap.png")
    return summary


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Search attention heads for induction-head-like circuits")
    parser.add_argument("--run-dir", type=str, default="runs/main")
    parser.add_argument("--num-episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    summary = run_attention_analysis(Path(args.run_dir), args.num_episodes, args.seed, args.top_k)
    print(json.dumps(summary["top_candidates"], indent=2))
