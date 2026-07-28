"""Linear probing: does the residual stream linearly encode "which arm
currently looks best"? The correlational half of reverse-engineering the
belief representation (analysis/patching.py is the causal half). Every
probe is checked against a majority-class baseline and a shuffled-label
control -- only meaningful if it clears both."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from env.bandit_family import sample_batch
from model.transformer_policy import TransformerPolicy
from training.train_meta_rl import TrainingConfig, collect_rollout


def best_arm_so_far(actions_taken: np.ndarray, rewards_obtained: np.ndarray, num_arms: int) -> np.ndarray:
    """(B, T) int array: argmax empirical mean reward over arms pulled
    STRICTLY BEFORE trial t (matching the causal information available to
    the policy when it chooses trial t's action). -1 wherever no arm has
    been pulled yet (trial 0), since there's no basis for a "belief" then --
    those positions should be excluded from probe training/evaluation."""
    B, T = actions_taken.shape
    counts = np.zeros((B, num_arms))
    sums = np.zeros((B, num_arms))
    labels = np.full((B, T), -1, dtype=int)
    batch_idx = np.arange(B)

    for t in range(T):
        has_any = counts.sum(axis=1) > 0
        means = np.divide(sums, counts, out=np.full_like(sums, -np.inf), where=counts > 0)
        labels[has_any, t] = means[has_any].argmax(axis=1)

        np.add.at(counts, (batch_idx, actions_taken[:, t]), 1)
        np.add.at(sums, (batch_idx, actions_taken[:, t]), rewards_obtained[:, t])

    return labels


@dataclass
class ProbeDataset:
    activations_per_layer: list  # list of (N, d_model) float32 arrays, one per layer
    labels: np.ndarray  # (N,) int


def collect_probe_dataset(
    policy: TransformerPolicy, config: TrainingConfig, num_episodes: int, seed: int
) -> ProbeDataset:
    device = torch.device("cpu")
    rng = np.random.default_rng(seed)
    task_batch = sample_batch("train", rng, num_episodes, config.num_arms, config.num_trials)
    rollout = collect_rollout(policy, task_batch, device, rng)

    policy.eval()
    with torch.no_grad():
        _, _, activations = policy(rollout.hist_actions, rollout.hist_rewards, return_activations=True)

    actions_np = rollout.actions_taken.cpu().numpy()
    rewards_np = rollout.rewards_obtained.cpu().numpy()
    labels_bt = best_arm_so_far(actions_np, rewards_np, config.num_arms)  # (B, T)

    valid = labels_bt >= 0  # exclude trial 0 (no belief yet)
    labels_flat = labels_bt[valid]

    activations_per_layer = []
    for resid in activations["resid_per_layer"]:
        resid_np = resid.numpy()  # (B, T, d_model)
        activations_per_layer.append(resid_np[valid].astype(np.float32))

    return ProbeDataset(activations_per_layer=activations_per_layer, labels=labels_flat)


def _train_linear_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    epochs: int = 200,
    lr: float = 0.05,
    weight_decay: float = 1e-3,
    seed: int = 0,
) -> tuple[nn.Linear, float]:
    torch.manual_seed(seed)
    d_model = X_train.shape[1]
    probe = nn.Linear(d_model, num_classes)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)

    X_train_t = torch.as_tensor(X_train)
    y_train_t = torch.as_tensor(y_train, dtype=torch.long)
    X_val_t = torch.as_tensor(X_val)
    y_val_t = torch.as_tensor(y_val, dtype=torch.long)

    for _ in range(epochs):
        optimizer.zero_grad()
        logits = probe(X_train_t)
        loss = nn.functional.cross_entropy(logits, y_train_t)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        val_acc = (probe(X_val_t).argmax(dim=-1) == y_val_t).float().mean().item()
    return probe, val_acc


def _train_val_split(n: int, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = max(1, int(n * val_fraction))
    return idx[n_val:], idx[:n_val]


def probe_all_layers(
    dataset: ProbeDataset, num_classes: int, val_fraction: float = 0.2, seed: int = 0
) -> dict:
    """Trains a linear probe on every layer's activations, returning per-
    layer {"val_accuracy", "majority_baseline_accuracy", "shuffled_label_accuracy"}."""
    n = len(dataset.labels)
    train_idx, val_idx = _train_val_split(n, val_fraction, seed)
    y_train, y_val = dataset.labels[train_idx], dataset.labels[val_idx]

    majority_class = np.bincount(y_train, minlength=num_classes).argmax()
    majority_baseline_accuracy = float((y_val == majority_class).mean())

    shuffled_y_train = np.random.default_rng(seed + 1).permutation(y_train)

    results = {}
    for layer_idx, activations in enumerate(dataset.activations_per_layer):
        X_train, X_val = activations[train_idx], activations[val_idx]

        _, val_acc = _train_linear_probe(X_train, y_train, X_val, y_val, num_classes, seed=seed)
        _, shuffled_val_acc = _train_linear_probe(
            X_train, shuffled_y_train, X_val, y_val, num_classes, seed=seed
        )

        results[f"layer_{layer_idx}"] = {
            "val_accuracy": val_acc,
            "majority_baseline_accuracy": majority_baseline_accuracy,
            "shuffled_label_accuracy": shuffled_val_acc,
        }
    return results


def best_probe_layer(results: dict) -> str:
    return max(results, key=lambda name: results[name]["val_accuracy"])


def fit_probe_for_layer(
    dataset: ProbeDataset, layer_idx: int, num_classes: int, val_fraction: float = 0.2, seed: int = 0
) -> nn.Linear:
    """Fits (and returns) the actual probe module for one layer, trained on
    the full non-held-out split -- used by analysis/patching.py to get real
    per-class direction vectors (probe.weight rows) to intervene with."""
    n = len(dataset.labels)
    train_idx, val_idx = _train_val_split(n, val_fraction, seed)
    X_train = dataset.activations_per_layer[layer_idx][train_idx]
    y_train = dataset.labels[train_idx]
    X_val = dataset.activations_per_layer[layer_idx][val_idx]
    y_val = dataset.labels[val_idx]
    probe, _ = _train_linear_probe(X_train, y_train, X_val, y_val, num_classes, seed=seed)
    return probe


def run_probe_analysis(run_dir: Path, num_episodes: int = 400, seed: int = 0) -> dict:
    import json

    from eval.adaptation import load_checkpoint

    run_dir = Path(run_dir)
    policy, config = load_checkpoint(run_dir / "checkpoint.pt")

    dataset = collect_probe_dataset(policy, config, num_episodes=num_episodes, seed=seed)
    results = probe_all_layers(dataset, num_classes=config.num_arms, seed=seed)
    summary = {"best_layer": best_probe_layer(results), "layers": results}

    with open(run_dir / "probe_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Probe the residual stream for a best-arm-so-far belief direction")
    parser.add_argument("--run-dir", type=str, default="runs/main")
    parser.add_argument("--num-episodes", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    summary = run_probe_analysis(Path(args.run_dir), args.num_episodes, args.seed)
    print(json.dumps(summary, indent=2))
