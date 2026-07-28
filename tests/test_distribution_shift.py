from pathlib import Path

import numpy as np
import torch

from shift.distribution_shift import (
    SHIFT_MODES,
    evaluate_all_modes,
    nonstationary_shock_curve,
    run_distribution_shift_eval,
)
from training.train_meta_rl import TrainingConfig, build_policy

TINY_CONFIG = TrainingConfig(
    num_arms=3,
    num_trials=40,
    d_model=16,
    n_heads=2,
    n_layers=2,
    d_ff=32,
    batch_size=8,
)


def test_evaluate_all_modes_returns_every_mode_with_correct_shape():
    policy = build_policy(TINY_CONFIG)
    results = evaluate_all_modes(policy, TINY_CONFIG, num_episodes=10, seed=0)
    assert set(results.keys()) == set(SHIFT_MODES)
    for regret in results.values():
        assert regret.shape == (10, TINY_CONFIG.num_trials)
        assert (regret >= -1e-8).all()


def test_nonstationary_shock_curve_shape():
    policy = build_policy(TINY_CONFIG)
    offsets, curve = nonstationary_shock_curve(policy, TINY_CONFIG, num_episodes=20, seed=1, window=10)
    assert offsets.shape == (21,)
    assert curve.shape == (21,)
    assert list(offsets) == list(range(-10, 11))


def test_nonstationary_shock_curve_handles_large_window_without_crashing():
    # window larger than num_trials -> many offsets fall outside [0, T) for
    # every episode and must be averaged as NaN-safe, not raise/crash.
    policy = build_policy(TINY_CONFIG)
    offsets, curve = nonstationary_shock_curve(policy, TINY_CONFIG, num_episodes=10, seed=1, window=100)
    assert offsets.shape == (201,)
    assert np.isfinite(curve[np.isfinite(curve)]).all()


def test_run_distribution_shift_eval_end_to_end(tmp_path: Path):
    policy = build_policy(TINY_CONFIG)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    torch.save(
        {"model_state_dict": policy.state_dict(), "config": TINY_CONFIG.__dict__},
        run_dir / "checkpoint.pt",
    )

    output = run_distribution_shift_eval(run_dir, num_episodes=12, seed=0, shock_window=5)
    assert set(output["per_mode"].keys()) == set(SHIFT_MODES)
    assert "nonstationary_shock" in output
    assert (run_dir / "distribution_shift.json").exists()
    assert (run_dir / "distribution_shift_curves.png").exists()
    assert (run_dir / "distribution_shift_shock.png").exists()
