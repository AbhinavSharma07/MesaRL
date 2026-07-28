"""Activation patching: causal verification of the belief direction found by
analysis/probes.py. A linear probe only shows correlation, so we intervene
on its direction mid-episode and check whether the network's own action
distribution shifts toward the targeted arm -- compared against a
same-magnitude random-direction control, using a contrastive (target minus
current) patch vector."""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from env.bandit_family import sample_batch
from model.transformer_policy import TransformerPolicy
from training.train_meta_rl import TrainingConfig, collect_rollout


def forward_with_patch(
    policy: TransformerPolicy,
    prev_actions: torch.Tensor,
    prev_rewards: torch.Tensor,
    layer_idx: int,
    position_idx: int,
    patch_vector: torch.Tensor,
):
    """Replicates TransformerPolicy.forward, but adds patch_vector to the
    residual stream at position_idx right after block layer_idx. Must exactly
    match policy.forward when patch_vector == 0 (see tests/test_patching.py)."""
    B, L = prev_actions.shape
    action_emb = policy.action_embed(prev_actions)
    reward_emb = policy.reward_proj(prev_rewards.unsqueeze(-1))
    x = torch.cat([action_emb, reward_emb], dim=-1)
    positions = torch.arange(L, device=prev_actions.device)
    x = x + policy.pos_embed(positions).unsqueeze(0)

    for i, block in enumerate(policy.blocks):
        x, _ = block(x)
        if i == layer_idx:
            x = x.clone()
            x[:, position_idx, :] = x[:, position_idx, :] + patch_vector

    x = policy.ln_f(x)
    logits = policy.policy_head(x)
    value = policy.value_head(x).squeeze(-1)
    return logits, value


@dataclass
class PatchingResult:
    baseline_prob_on_target: float
    patched_prob_on_target: float
    control_patched_prob_on_target: float
    prob_shift: float
    control_prob_shift: float
    fraction_argmax_became_target: float
    control_fraction_argmax_became_target: float


def run_patching_experiment(
    policy: TransformerPolicy,
    config: TrainingConfig,
    probe,  # nn.Linear from analysis.probes.fit_probe_for_layer
    layer_idx: int,
    position_idx: int,
    num_episodes: int = 300,
    scale: float = 4.0,
    seed: int = 2024,
    target_arm: Optional[int] = None,
) -> PatchingResult:
    """Patches the residual stream at position_idx toward a target arm --
    by default one step from the current belief-probe prediction, or a fixed
    arm if target_arm is given -- and measures the probability shift onto it
    vs. a same-magnitude random-direction control."""
    device = torch.device("cpu")
    rng = np.random.default_rng(seed)
    task_batch = sample_batch("train", rng, num_episodes, config.num_arms, config.num_trials)
    rollout = collect_rollout(policy, task_batch, device, rng)

    prev_actions = rollout.hist_actions
    prev_rewards = rollout.hist_rewards
    B = prev_actions.shape[0]
    num_arms = config.num_arms

    with torch.no_grad():
        baseline_logits, _, activations = policy(prev_actions, prev_rewards, return_activations=True)
    baseline_probs = torch.softmax(baseline_logits[:, position_idx, :], dim=-1)  # (B, K)

    resid_at_position = activations["resid_per_layer"][layer_idx][:, position_idx, :]  # (B, d_model)
    with torch.no_grad():
        current_belief = probe(resid_at_position).argmax(dim=-1)  # (B,)
    if target_arm is None:
        target_arm_t = (current_belief + 1) % num_arms  # (B,) -- well-defined, differs from current belief
    else:
        target_arm_t = torch.full((B,), target_arm, dtype=torch.long)
    target_arm = target_arm_t

    probe_weight = probe.weight.detach()  # (K, d_model)
    direction = probe_weight[target_arm] - probe_weight[current_belief]  # (B, d_model)
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    patch_vector = scale * direction

    torch_rng = torch.Generator().manual_seed(seed)
    random_direction = torch.randn(B, policy.config.d_model, generator=torch_rng)
    random_direction = random_direction / random_direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    control_patch_vector = scale * random_direction

    with torch.no_grad():
        patched_logits, _ = forward_with_patch(
            policy, prev_actions, prev_rewards, layer_idx, position_idx, patch_vector
        )
        control_logits, _ = forward_with_patch(
            policy, prev_actions, prev_rewards, layer_idx, position_idx, control_patch_vector
        )
    patched_probs = torch.softmax(patched_logits[:, position_idx, :], dim=-1)
    control_probs = torch.softmax(control_logits[:, position_idx, :], dim=-1)

    batch_idx = torch.arange(B)
    baseline_on_target = baseline_probs[batch_idx, target_arm]
    patched_on_target = patched_probs[batch_idx, target_arm]
    control_on_target = control_probs[batch_idx, target_arm]

    return PatchingResult(
        baseline_prob_on_target=baseline_on_target.mean().item(),
        patched_prob_on_target=patched_on_target.mean().item(),
        control_patched_prob_on_target=control_on_target.mean().item(),
        prob_shift=(patched_on_target - baseline_on_target).mean().item(),
        control_prob_shift=(control_on_target - baseline_on_target).mean().item(),
        fraction_argmax_became_target=(
            (patched_probs.argmax(dim=-1) == target_arm) & (baseline_probs.argmax(dim=-1) != target_arm)
        ).float().mean().item(),
        control_fraction_argmax_became_target=(
            (control_probs.argmax(dim=-1) == target_arm) & (baseline_probs.argmax(dim=-1) != target_arm)
        ).float().mean().item(),
    )


def run_patching_analysis(
    run_dir: Path,
    num_episodes: int = 300,
    position_idx: Optional[int] = None,
    scale: float = 4.0,
    seed: int = 2024,
    target_arm: Optional[int] = None,
) -> dict:
    import json

    from analysis.probes import best_probe_layer, collect_probe_dataset, fit_probe_for_layer, probe_all_layers
    from eval.adaptation import load_checkpoint

    run_dir = Path(run_dir)
    policy, config = load_checkpoint(run_dir / "checkpoint.pt")
    position_idx = position_idx if position_idx is not None else config.num_trials // 2

    dataset = collect_probe_dataset(policy, config, num_episodes=num_episodes, seed=seed)
    probe_results = probe_all_layers(dataset, num_classes=config.num_arms, seed=seed)
    best_layer_name = best_probe_layer(probe_results)
    layer_idx = int(best_layer_name.split("_")[1])
    probe = fit_probe_for_layer(dataset, layer_idx=layer_idx, num_classes=config.num_arms, seed=seed)

    result = run_patching_experiment(
        policy, config, probe, layer_idx=layer_idx, position_idx=position_idx,
        num_episodes=num_episodes, scale=scale, seed=seed, target_arm=target_arm,
    )

    summary = {
        "probed_layer": best_layer_name,
        "probe_val_accuracy": probe_results[best_layer_name]["val_accuracy"],
        "position_idx": position_idx,
        "scale": scale,
        "target_arm": target_arm,
        **asdict(result),
    }
    with open(run_dir / "patching_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Causally test the probed belief direction via activation patching")
    parser.add_argument("--run-dir", type=str, default="runs/main")
    parser.add_argument("--num-episodes", type=int, default=300)
    parser.add_argument("--position-idx", type=int, default=None)
    parser.add_argument("--scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--target-arm", type=int, default=None)
    args = parser.parse_args()

    summary = run_patching_analysis(
        Path(args.run_dir), args.num_episodes, args.position_idx, args.scale, args.seed, args.target_arm
    )
    print(json.dumps(summary, indent=2))
