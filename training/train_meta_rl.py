"""PPO meta-training for the Transformer bandit policy.

The "base optimizer" here is ordinary PPO, run across thousands of freshly
sampled bandit task instances (env.bandit_family, mode="train"). The network
is never told how to explore a new bandit -- it only ever gets gradient
signal for *cumulative episode reward*. Any within-episode adaptation the
frozen network exhibits at test time (see eval/adaptation.py) is therefore
something it had to invent for itself: the emergent mesa-optimizer.

Because CausalSelfAttention masks out future positions, a single forward
pass over a *completed* episode's full (action, reward) history reproduces,
at every position, exactly the per-step distribution used to sample that
position's action during collection -- so PPO's log-prob/value recomputation
during the update only needs one forward pass per minibatch per epoch, not
one per timestep. Only *rollout collection* is inherently sequential (each
action depends on the previous trial's observed reward).
"""

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from env.bandit_family import BanditTaskBatch, sample_batch
from model.transformer_policy import TransformerPolicy, TransformerPolicyConfig


@dataclass
class TrainingConfig:
    num_arms: int = 5
    num_trials: int = 100
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 3
    d_ff: int = 128
    dropout: float = 0.0

    batch_size: int = 256
    num_iterations: int = 300
    ppo_epochs: int = 4
    minibatch_size: int = 64
    clip_eps: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 3e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    seed: int = 0


@dataclass
class RolloutBatch:
    hist_actions: torch.Tensor  # (B, T) long, model input convention (shifted)
    hist_rewards: torch.Tensor  # (B, T) float, model input convention (shifted)
    actions_taken: torch.Tensor  # (B, T) long
    rewards_obtained: torch.Tensor  # (B, T) float
    log_probs: torch.Tensor  # (B, T) float, log-prob under the collecting policy
    values: torch.Tensor  # (B, T) float, value estimate under the collecting policy
    regret: np.ndarray  # (B, T) float, optimal_mean(t) - true_mean(chosen_arm, t)


@torch.no_grad()
def collect_rollout(
    policy: TransformerPolicy,
    task_batch: BanditTaskBatch,
    device: torch.device,
    reward_rng: np.random.Generator,
    torch_rng: Optional[torch.Generator] = None,
    greedy: bool = False,
) -> RolloutBatch:
    policy.eval()
    B = task_batch.batch_size
    T = task_batch.arm_means.shape[1] and policy.config.max_trials
    T = policy.config.max_trials

    hist_actions = torch.full((B, T), policy.no_prev_action, dtype=torch.long, device=device)
    hist_rewards = torch.zeros((B, T), dtype=torch.float32, device=device)
    actions_taken = torch.zeros((B, T), dtype=torch.long, device=device)
    rewards_obtained = torch.zeros((B, T), dtype=torch.float32, device=device)
    log_probs = torch.zeros((B, T), dtype=torch.float32, device=device)
    values = torch.zeros((B, T), dtype=torch.float32, device=device)
    regret = np.zeros((B, T), dtype=np.float64)

    for t in range(T):
        action, log_prob, value = policy.act(
            hist_actions[:, : t + 1], hist_rewards[:, : t + 1], rng=torch_rng, greedy=greedy
        )
        action_np = action.cpu().numpy()
        reward_np = task_batch.reward(action_np, t, reward_rng)
        reward = torch.as_tensor(reward_np, dtype=torch.float32, device=device)

        actions_taken[:, t] = action
        rewards_obtained[:, t] = reward
        log_probs[:, t] = log_prob
        values[:, t] = value

        means_t = task_batch.means_at(t)
        chosen_mean = means_t[np.arange(B), action_np]
        regret[:, t] = task_batch.optimal_mean_at(t) - chosen_mean

        if t + 1 < T:
            hist_actions[:, t + 1] = action
            hist_rewards[:, t + 1] = reward

    return RolloutBatch(
        hist_actions=hist_actions,
        hist_rewards=hist_rewards,
        actions_taken=actions_taken,
        rewards_obtained=rewards_obtained,
        log_probs=log_probs,
        values=values,
        regret=regret,
    )


def compute_gae(
    rewards: torch.Tensor, values: torch.Tensor, gamma: float, gae_lambda: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """No bootstrap beyond the episode: the implicit value after the final
    trial is 0."""
    B, T = rewards.shape
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(B, device=rewards.device)
    for t in reversed(range(T)):
        next_value = values[:, t + 1] if t + 1 < T else torch.zeros(B, device=rewards.device)
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        gae = delta + gamma * gae_lambda * gae
        advantages[:, t] = gae
    returns = advantages + values
    return advantages, returns


def ppo_update(
    policy: TransformerPolicy,
    optimizer: torch.optim.Optimizer,
    rollout: RolloutBatch,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    config: TrainingConfig,
) -> dict:
    policy.train()
    B = rollout.hist_actions.shape[0]
    adv_mean, adv_std = advantages.mean(), advantages.std().clamp_min(1e-8)
    norm_advantages = (advantages - adv_mean) / adv_std

    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "n_updates": 0}
    for _ in range(config.ppo_epochs):
        perm = torch.randperm(B)
        for start in range(0, B, config.minibatch_size):
            idx = perm[start : start + config.minibatch_size]

            logits, values, _ = policy(rollout.hist_actions[idx], rollout.hist_rewards[idx])
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(rollout.actions_taken[idx])
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - rollout.log_probs[idx])
            mb_adv = norm_advantages[idx]
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1 - config.clip_eps, 1 + config.clip_eps) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values, returns[idx])
            loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
            optimizer.step()

            stats["policy_loss"] += policy_loss.item()
            stats["value_loss"] += value_loss.item()
            stats["entropy"] += entropy.item()
            stats["n_updates"] += 1

    n = max(stats["n_updates"], 1)
    return {
        "policy_loss": stats["policy_loss"] / n,
        "value_loss": stats["value_loss"] / n,
        "entropy": stats["entropy"] / n,
    }


def build_policy(config: TrainingConfig) -> TransformerPolicy:
    model_config = TransformerPolicyConfig(
        num_arms=config.num_arms,
        max_trials=config.num_trials,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ff=config.d_ff,
        dropout=config.dropout,
    )
    return TransformerPolicy(model_config)


def train(
    config: TrainingConfig,
    run_dir: Optional[Path] = None,
    log_every: int = 10,
    device: Optional[torch.device] = None,
) -> tuple[TransformerPolicy, list[dict]]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.seed)
    np_rng = np.random.default_rng(config.seed)

    policy = build_policy(config).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.lr)

    history = []
    for iteration in range(config.num_iterations):
        task_batch = sample_batch("train", np_rng, config.batch_size, config.num_arms, config.num_trials)
        rollout = collect_rollout(policy, task_batch, device, np_rng)
        advantages, returns = compute_gae(
            rollout.rewards_obtained, rollout.values, config.gamma, config.gae_lambda
        )
        update_stats = ppo_update(policy, optimizer, rollout, advantages, returns, config)

        mean_reward = rollout.rewards_obtained.mean().item()
        mean_regret_first10 = float(rollout.regret[:, :10].mean())
        mean_regret_last10 = float(rollout.regret[:, -10:].mean())
        record = {
            "iteration": iteration,
            "mean_episode_reward": mean_reward,
            "mean_regret_first10": mean_regret_first10,
            "mean_regret_last10": mean_regret_last10,
            **update_stats,
        }
        history.append(record)

        if log_every and iteration % log_every == 0:
            print(
                f"iter {iteration:5d} | reward {mean_reward:+.3f} | "
                f"regret[first10] {mean_regret_first10:.3f} -> regret[last10] {mean_regret_last10:.3f} | "
                f"policy_loss {update_stats['policy_loss']:+.4f} | entropy {update_stats['entropy']:.3f}"
            )

    if run_dir is not None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model_state_dict": policy.state_dict(), "config": asdict(config)},
            run_dir / "checkpoint.pt",
        )
        with open(run_dir / "training_log.json", "w") as f:
            json.dump(history, f, indent=2)

    return policy, history


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Meta-train the MesaRL Transformer bandit policy")
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--run-dir", type=str, default="runs/default")
    parser.add_argument("--quick", action="store_true", help="tiny smoke-test config")
    args = parser.parse_args()

    cfg = TrainingConfig(num_iterations=args.iterations, batch_size=args.batch_size)
    if args.quick:
        cfg = TrainingConfig(
            num_iterations=5,
            batch_size=16,
            num_trials=20,
            d_model=16,
            n_heads=2,
            n_layers=2,
            d_ff=32,
            minibatch_size=8,
            ppo_epochs=2,
        )

    train(cfg, run_dir=Path(args.run_dir))
