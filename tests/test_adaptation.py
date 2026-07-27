from pathlib import Path

import numpy as np
import torch

from eval.adaptation import (
    evaluate_adaptation,
    load_checkpoint,
    random_policy_regret,
    run_adaptation_eval,
    summarize_regret,
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


def test_random_policy_regret_shape_and_non_negative():
    regret = random_policy_regret("train", num_episodes=32, num_arms=4, num_trials=15, seed=0)
    assert regret.shape == (32, 15)
    assert (regret >= -1e-8).all()


def test_random_policy_regret_does_not_decrease_on_average():
    # A memoryless random policy has no mechanism to improve within an
    # episode, so first-half and second-half mean regret should be close
    # (no systematic decrease) -- unlike the trained policy.
    regret = random_policy_regret("train", num_episodes=4000, num_arms=5, num_trials=40, seed=1)
    first_half = regret[:, :20].mean()
    second_half = regret[:, 20:].mean()
    assert abs(first_half - second_half) < 0.1 * max(first_half, second_half, 1e-8)


def test_summarize_regret_on_synthetic_decreasing_curve():
    # Constant regret of 1.0 for first 10, 0.0 for last 10 trials.
    regret = np.concatenate([np.ones((5, 10)), np.zeros((5, 10))], axis=1)
    summary = summarize_regret(regret, first_n=10, last_n=10)
    assert summary["mean_regret_first_n"] == 1.0
    assert summary["mean_regret_last_n"] == 0.0
    assert summary["improvement_ratio"] == 1.0
    assert summary["mean_cumulative_regret"] == 10.0


def test_summarize_regret_handles_zero_first_n_gracefully():
    regret = np.zeros((3, 10))
    summary = summarize_regret(regret)
    assert summary["improvement_ratio"] == 0.0


def test_evaluate_adaptation_returns_correct_shape():
    policy = build_policy(TINY_CONFIG)
    regret = evaluate_adaptation(policy, TINY_CONFIG, num_episodes=16, seed=5)
    assert regret.shape == (16, TINY_CONFIG.num_trials)
    assert (regret >= -1e-8).all()


def test_checkpoint_round_trip(tmp_path: Path):
    policy = build_policy(TINY_CONFIG)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    torch.save(
        {"model_state_dict": policy.state_dict(), "config": TINY_CONFIG.__dict__},
        run_dir / "checkpoint.pt",
    )

    loaded_policy, loaded_config = load_checkpoint(run_dir / "checkpoint.pt")
    assert loaded_config == TINY_CONFIG
    for p1, p2 in zip(policy.parameters(), loaded_policy.parameters()):
        assert torch.equal(p1, p2)


def test_run_adaptation_eval_end_to_end(tmp_path: Path):
    policy = build_policy(TINY_CONFIG)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    torch.save(
        {"model_state_dict": policy.state_dict(), "config": TINY_CONFIG.__dict__},
        run_dir / "checkpoint.pt",
    )

    summary = run_adaptation_eval(run_dir, num_episodes=8, seed=1)
    assert "trained_policy" in summary and "random_baseline" in summary
    assert (run_dir / "adaptation_eval.json").exists()
    assert (run_dir / "adaptation_regret_curve.png").exists()
