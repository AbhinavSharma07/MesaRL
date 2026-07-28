import torch

from analysis.patching import forward_with_patch, run_patching_experiment
from analysis.probes import collect_probe_dataset, fit_probe_for_layer
from env.bandit_family import sample_batch
from training.train_meta_rl import TrainingConfig, build_policy, collect_rollout

TINY_CONFIG = TrainingConfig(
    num_arms=3,
    num_trials=10,
    d_model=16,
    n_heads=2,
    n_layers=2,
    d_ff=32,
    batch_size=8,
)


def make_rollout(policy, config, num_episodes=8, seed=0):
    device = torch.device("cpu")
    import numpy as np

    rng = np.random.default_rng(seed)
    task_batch = sample_batch("train", rng, num_episodes, config.num_arms, config.num_trials)
    return collect_rollout(policy, task_batch, device, rng)


def test_zero_patch_reproduces_forward_exactly():
    policy = build_policy(TINY_CONFIG)
    policy.eval()
    rollout = make_rollout(policy, TINY_CONFIG)

    with torch.no_grad():
        expected_logits, expected_value, _ = policy(rollout.hist_actions, rollout.hist_rewards)
        zero_patch = torch.zeros(rollout.hist_actions.shape[0], TINY_CONFIG.d_model)
        patched_logits, patched_value = forward_with_patch(
            policy, rollout.hist_actions, rollout.hist_rewards, layer_idx=0, position_idx=5, patch_vector=zero_patch
        )

    assert torch.allclose(expected_logits, patched_logits, atol=1e-5)
    assert torch.allclose(expected_value, patched_value, atol=1e-5)


def test_nonzero_patch_leaves_earlier_positions_unchanged():
    policy = build_policy(TINY_CONFIG)
    policy.eval()
    rollout = make_rollout(policy, TINY_CONFIG)
    B = rollout.hist_actions.shape[0]
    position_idx = 6

    with torch.no_grad():
        baseline_logits, _, _ = policy(rollout.hist_actions, rollout.hist_rewards)
        patch = torch.randn(B, TINY_CONFIG.d_model) * 10.0
        patched_logits, _ = forward_with_patch(
            policy, rollout.hist_actions, rollout.hist_rewards, layer_idx=0, position_idx=position_idx, patch_vector=patch
        )

    # Causality must still hold: patching position_idx cannot affect the
    # logits computed for any strictly earlier query position.
    assert torch.allclose(baseline_logits[:, :position_idx], patched_logits[:, :position_idx], atol=1e-5)
    # But it should (with high probability, given a x10-scaled random patch)
    # change the logits at the patched position itself.
    assert not torch.allclose(baseline_logits[:, position_idx], patched_logits[:, position_idx], atol=1e-3)


def test_run_patching_experiment_returns_valid_ranges():
    policy = build_policy(TINY_CONFIG)
    dataset = collect_probe_dataset(policy, TINY_CONFIG, num_episodes=40, seed=0)
    probe = fit_probe_for_layer(dataset, layer_idx=0, num_classes=TINY_CONFIG.num_arms, seed=0)

    result = run_patching_experiment(
        policy, TINY_CONFIG, probe, layer_idx=0, position_idx=5, num_episodes=20, scale=4.0, seed=1
    )

    for prob in (result.baseline_prob_on_target, result.patched_prob_on_target, result.control_patched_prob_on_target):
        assert 0.0 <= prob <= 1.0
    for frac in (result.fraction_argmax_became_target, result.control_fraction_argmax_became_target):
        assert 0.0 <= frac <= 1.0
    assert -1.0 <= result.prob_shift <= 1.0
    assert -1.0 <= result.control_prob_shift <= 1.0


def test_run_patching_experiment_respects_fixed_target_arm():
    policy = build_policy(TINY_CONFIG)
    dataset = collect_probe_dataset(policy, TINY_CONFIG, num_episodes=40, seed=0)
    probe = fit_probe_for_layer(dataset, layer_idx=0, num_classes=TINY_CONFIG.num_arms, seed=0)

    result = run_patching_experiment(
        policy, TINY_CONFIG, probe, layer_idx=0, position_idx=5,
        num_episodes=20, scale=4.0, seed=1, target_arm=2,
    )
    assert 0.0 <= result.patched_prob_on_target <= 1.0
