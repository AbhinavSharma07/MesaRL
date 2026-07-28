from pathlib import Path

import numpy as np
import pytest
import torch

from analysis.compare import (
    build_candidate_distributions,
    compare_to_candidates,
    kl_divergence,
    most_similar_candidate,
    network_policy_distributions,
    run_candidate_comparison,
)
from training.train_meta_rl import TrainingConfig, build_policy, collect_rollout, train
from env.bandit_family import sample_batch

TINY_CONFIG = TrainingConfig(
    num_arms=3,
    num_trials=10,
    d_model=16,
    n_heads=2,
    n_layers=2,
    d_ff=32,
    batch_size=8,
)


def test_kl_divergence_is_zero_for_identical_distributions():
    p = np.array([[0.2, 0.3, 0.5], [0.1, 0.1, 0.8]])
    kl = kl_divergence(p, p.copy())
    assert np.allclose(kl, 0.0, atol=1e-6)


def test_kl_divergence_known_two_point_value():
    p = np.array([[0.9, 0.1]])
    q = np.array([[0.5, 0.5]])
    expected = 0.9 * np.log(0.9 / 0.5) + 0.1 * np.log(0.1 / 0.5)
    kl = kl_divergence(p, q)
    assert kl[0] == pytest.approx(expected, rel=1e-4)


def test_kl_divergence_handles_zero_probability_in_p_gracefully():
    p = np.array([[1.0, 0.0, 0.0]])
    q = np.array([[0.4, 0.3, 0.3]])
    kl = kl_divergence(p, q)
    assert np.isfinite(kl).all()
    assert kl[0] > 0


def test_network_policy_distributions_shape_and_normalization():
    policy = build_policy(TINY_CONFIG)
    device = torch.device("cpu")
    rng = np.random.default_rng(0)
    task_batch = sample_batch("train", rng, TINY_CONFIG.batch_size, TINY_CONFIG.num_arms, TINY_CONFIG.num_trials)
    rollout = collect_rollout(policy, task_batch, device, rng)
    dist = network_policy_distributions(policy, rollout, device)
    assert dist.shape == (TINY_CONFIG.batch_size, TINY_CONFIG.num_trials, TINY_CONFIG.num_arms)
    assert np.allclose(dist.sum(axis=-1), 1.0, atol=1e-5)


def test_build_candidate_distributions_returns_all_four():
    rng = np.random.default_rng(0)
    actions = rng.integers(0, 3, size=(5, 10))
    rewards = rng.normal(size=(5, 10))
    candidates = build_candidate_distributions(actions, rewards, num_arms=3)
    assert set(candidates.keys()) == {
        "Thompson sampling", "UCB1", "Epsilon-greedy", "Win-stay-lose-shift"
    }
    for dist in candidates.values():
        assert dist.shape == (5, 10, 3)


def test_most_similar_candidate_picks_lowest_kl():
    results = {
        "A": {"mean_kl_overall": 0.5},
        "B": {"mean_kl_overall": 0.1},
        "C": {"mean_kl_overall": 0.9},
    }
    assert most_similar_candidate(results) == "B"


def test_compare_to_candidates_end_to_end_shapes():
    policy = build_policy(TINY_CONFIG)
    results = compare_to_candidates(policy, TINY_CONFIG, num_episodes=16, seed=1)
    assert set(results.keys()) == {
        "Thompson sampling", "UCB1", "Epsilon-greedy", "Win-stay-lose-shift"
    }
    for stats in results.values():
        assert len(stats["mean_kl_per_trial"]) == TINY_CONFIG.num_trials
        assert stats["mean_kl_overall"] >= 0
        assert 0.0 <= stats["action_agreement_rate"] <= 1.0


def test_run_candidate_comparison_end_to_end(tmp_path: Path):
    policy = build_policy(TINY_CONFIG)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    torch.save(
        {"model_state_dict": policy.state_dict(), "config": TINY_CONFIG.__dict__},
        run_dir / "checkpoint.pt",
    )

    summary = run_candidate_comparison(run_dir, num_episodes=8, seed=1)
    assert summary["most_similar_candidate"] in {
        "Thompson sampling", "UCB1", "Epsilon-greedy", "Win-stay-lose-shift"
    }
    assert (run_dir / "candidate_comparison.json").exists()
    assert (run_dir / "candidate_comparison_kl.png").exists()
