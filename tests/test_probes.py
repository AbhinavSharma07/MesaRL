import numpy as np
import pytest

from analysis.probes import (
    ProbeDataset,
    best_arm_so_far,
    best_probe_layer,
    collect_probe_dataset,
    fit_probe_for_layer,
    probe_all_layers,
)
from training.train_meta_rl import TrainingConfig, build_policy

TINY_CONFIG = TrainingConfig(
    num_arms=3,
    num_trials=10,
    d_model=16,
    n_heads=2,
    n_layers=2,
    d_ff=32,
    batch_size=8,
)


def test_best_arm_so_far_trial_zero_is_always_invalid():
    actions = np.zeros((5, 6), dtype=int)
    rewards = np.zeros((5, 6))
    labels = best_arm_so_far(actions, rewards, num_arms=3)
    assert (labels[:, 0] == -1).all()


def test_best_arm_so_far_matches_hand_computation():
    # arm 0 gets reward 1.0 at t=0, arm 1 gets reward -1.0 at t=1 -> at t=2,
    # best-so-far should be arm 0 (mean 1.0 > arm 1's mean -1.0).
    actions = np.array([[0, 1, 2]])
    rewards = np.array([[1.0, -1.0, 0.0]])
    labels = best_arm_so_far(actions, rewards, num_arms=3)
    assert labels[0, 0] == -1
    assert labels[0, 1] == 0  # only arm 0 pulled so far
    assert labels[0, 2] == 0  # arm 0 (mean 1.0) beats arm 1 (mean -1.0)


def test_best_arm_so_far_updates_when_second_arm_overtakes():
    actions = np.array([[0, 0, 1, 1]])
    rewards = np.array([[0.1, 0.1, 5.0, 5.0]])
    labels = best_arm_so_far(actions, rewards, num_arms=2)
    assert labels[0, 1] == 0  # only arm 0 seen (mean 0.1)
    assert labels[0, 2] == 0  # still only arm 0 has data
    assert labels[0, 3] == 1  # arm 1's single observation (5.0) now beats arm 0 (0.1)


def test_collect_probe_dataset_shapes():
    policy = build_policy(TINY_CONFIG)
    dataset = collect_probe_dataset(policy, TINY_CONFIG, num_episodes=20, seed=0)
    assert len(dataset.activations_per_layer) == TINY_CONFIG.n_layers
    n_valid = len(dataset.labels)
    for activations in dataset.activations_per_layer:
        assert activations.shape == (n_valid, TINY_CONFIG.d_model)
    assert (dataset.labels >= 0).all()
    assert n_valid <= 20 * TINY_CONFIG.num_trials  # trial-0 rows excluded


def test_probe_all_layers_returns_stats_for_every_layer():
    policy = build_policy(TINY_CONFIG)
    dataset = collect_probe_dataset(policy, TINY_CONFIG, num_episodes=40, seed=0)
    results = probe_all_layers(dataset, num_classes=TINY_CONFIG.num_arms, seed=0)
    assert set(results.keys()) == {f"layer_{i}" for i in range(TINY_CONFIG.n_layers)}
    for stats in results.values():
        assert 0.0 <= stats["val_accuracy"] <= 1.0
        assert 0.0 <= stats["majority_baseline_accuracy"] <= 1.0
        assert 0.0 <= stats["shuffled_label_accuracy"] <= 1.0


def test_probe_recovers_signal_on_synthetic_linearly_encoded_labels():
    # Sanity check on the probing PIPELINE itself (not a real network):
    # activations are literally a noisy one-hot encoding of the true label,
    # so a linear probe should recover it near-perfectly, while a
    # shuffled-label control should stay near chance.
    rng = np.random.default_rng(0)
    num_classes = 3
    n = 600
    labels = rng.integers(0, num_classes, size=n)
    onehot = np.eye(num_classes, dtype=np.float32)[labels]
    activations = onehot + rng.normal(0, 0.05, size=onehot.shape).astype(np.float32)

    dataset = ProbeDataset(activations_per_layer=[activations], labels=labels)
    results = probe_all_layers(dataset, num_classes=num_classes, seed=0)

    assert results["layer_0"]["val_accuracy"] > 0.9
    assert results["layer_0"]["shuffled_label_accuracy"] < 0.6


def test_best_probe_layer_picks_highest_accuracy():
    results = {
        "layer_0": {"val_accuracy": 0.4},
        "layer_1": {"val_accuracy": 0.8},
        "layer_2": {"val_accuracy": 0.6},
    }
    assert best_probe_layer(results) == "layer_1"


def test_fit_probe_for_layer_returns_correct_weight_shape():
    policy = build_policy(TINY_CONFIG)
    dataset = collect_probe_dataset(policy, TINY_CONFIG, num_episodes=20, seed=0)
    probe = fit_probe_for_layer(dataset, layer_idx=0, num_classes=TINY_CONFIG.num_arms, seed=0)
    assert probe.weight.shape == (TINY_CONFIG.num_arms, TINY_CONFIG.d_model)
