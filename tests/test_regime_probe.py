from pathlib import Path

import numpy as np
import torch

from regime_probe.train_deploy_probe import collect_regime_dataset, run_regime_probe_analysis
from training.train_meta_rl import TrainingConfig, build_policy

TINY_CONFIG = TrainingConfig(
    num_arms=3,
    num_trials=30,
    d_model=16,
    n_heads=2,
    n_layers=2,
    d_ff=32,
    batch_size=8,
)


def test_collect_regime_dataset_shapes_and_labels():
    policy = build_policy(TINY_CONFIG)
    dataset, behavior, groups = collect_regime_dataset(
        policy, TINY_CONFIG, num_episodes_per_regime=10, early_window=5, seed=0
    )
    n_expected = 2 * 10 * 5  # 2 regimes * episodes * early_window
    assert len(dataset.labels) == n_expected
    assert set(dataset.labels.tolist()) == {0, 1}
    assert (dataset.labels == 0).sum() == (dataset.labels == 1).sum()  # balanced
    for activations in dataset.activations_per_layer:
        assert activations.shape == (n_expected, TINY_CONFIG.d_model)

    assert groups.shape == (n_expected,)
    assert len(np.unique(groups)) == 2 * 10  # one group id per episode
    # every sample from the same episode must share the same label
    for group_id in np.unique(groups):
        assert len(set(dataset.labels[groups == group_id].tolist())) == 1

    assert set(behavior.keys()) == {"regime_low_noise", "regime_high_noise"}
    for stats in behavior.values():
        assert stats["mean_action_entropy"] >= 0
        assert stats["mean_distinct_arms_tried"] >= 1


def test_run_regime_probe_analysis_end_to_end(tmp_path: Path):
    policy = build_policy(TINY_CONFIG)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    torch.save(
        {"model_state_dict": policy.state_dict(), "config": TINY_CONFIG.__dict__},
        run_dir / "checkpoint.pt",
    )

    summary = run_regime_probe_analysis(run_dir, num_episodes_per_regime=15, early_window=5, seed=0)
    assert set(summary["probe_results"].keys()) == {"layer_0", "layer_1"}
    assert summary["best_layer"] in {"layer_0", "layer_1"}
    assert isinstance(summary["entropy_gap_high_minus_low"], float)
    assert isinstance(summary["distinct_arms_gap_high_minus_low"], float)
    assert (run_dir / "regime_probe.json").exists()
    assert (run_dir / "regime_probe_accuracy.png").exists()
