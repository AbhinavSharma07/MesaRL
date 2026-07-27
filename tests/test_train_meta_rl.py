import copy

import numpy as np
import torch

from env.bandit_family import sample_batch
from training.train_meta_rl import (
    TrainingConfig,
    build_policy,
    collect_rollout,
    compute_gae,
    config_from_dict,
    ppo_update,
    train,
)

TINY_CONFIG = TrainingConfig(
    num_arms=3,
    num_trials=10,
    d_model=16,
    n_heads=2,
    n_layers=2,
    d_ff=32,
    batch_size=8,
    minibatch_size=4,
    ppo_epochs=2,
    seed=0,
)


def test_collect_rollout_shapes_and_shifted_history():
    policy = build_policy(TINY_CONFIG)
    device = torch.device("cpu")
    np_rng = np.random.default_rng(0)
    task_batch = sample_batch(
        "train", np_rng, TINY_CONFIG.batch_size, TINY_CONFIG.num_arms, TINY_CONFIG.num_trials
    )
    rollout = collect_rollout(policy, task_batch, device, np_rng)

    B, T = TINY_CONFIG.batch_size, TINY_CONFIG.num_trials
    assert rollout.actions_taken.shape == (B, T)
    assert rollout.rewards_obtained.shape == (B, T)
    assert rollout.log_probs.shape == (B, T)
    assert rollout.values.shape == (B, T)
    assert rollout.regret.shape == (B, T)

    assert (rollout.hist_actions[:, 0] == policy.no_prev_action).all()
    assert torch.equal(rollout.hist_actions[:, 1], rollout.actions_taken[:, 0])
    assert torch.allclose(rollout.hist_rewards[:, 1], rollout.rewards_obtained[:, 0])


def test_regret_is_always_non_negative():
    policy = build_policy(TINY_CONFIG)
    device = torch.device("cpu")
    np_rng = np.random.default_rng(1)
    task_batch = sample_batch(
        "train", np_rng, TINY_CONFIG.batch_size, TINY_CONFIG.num_arms, TINY_CONFIG.num_trials
    )
    rollout = collect_rollout(policy, task_batch, device, np_rng)
    assert (rollout.regret >= -1e-8).all()


def test_compute_gae_matches_hand_computation_with_lambda_one_gamma_one():
    rewards = torch.ones(1, 3)
    values = torch.zeros(1, 3)
    advantages, returns = compute_gae(rewards, values, gamma=1.0, gae_lambda=1.0)
    assert torch.allclose(advantages, torch.tensor([[3.0, 2.0, 1.0]]))
    assert torch.allclose(returns, torch.tensor([[3.0, 2.0, 1.0]]))


def test_compute_gae_zero_reward_zero_value_gives_zero_advantage():
    rewards = torch.zeros(2, 5)
    values = torch.zeros(2, 5)
    advantages, returns = compute_gae(rewards, values, gamma=0.99, gae_lambda=0.95)
    assert torch.allclose(advantages, torch.zeros(2, 5))
    assert torch.allclose(returns, torch.zeros(2, 5))


def test_ppo_update_changes_parameters_and_produces_finite_stats():
    policy = build_policy(TINY_CONFIG)
    optimizer = torch.optim.Adam(policy.parameters(), lr=TINY_CONFIG.lr)
    device = torch.device("cpu")
    np_rng = np.random.default_rng(2)
    task_batch = sample_batch(
        "train", np_rng, TINY_CONFIG.batch_size, TINY_CONFIG.num_arms, TINY_CONFIG.num_trials
    )
    rollout = collect_rollout(policy, task_batch, device, np_rng)
    advantages, returns = compute_gae(
        rollout.rewards_obtained, rollout.values, TINY_CONFIG.gamma, TINY_CONFIG.gae_lambda
    )

    params_before = copy.deepcopy(list(policy.parameters()))
    stats = ppo_update(policy, optimizer, rollout, advantages, returns, TINY_CONFIG, entropy_coef=0.01)

    assert all(np.isfinite(v) for v in stats.values())
    params_after = list(policy.parameters())
    changed = any(
        not torch.equal(before, after)
        for before, after in zip(params_before, params_after)
    )
    assert changed


def test_config_from_dict_ignores_unknown_and_fills_missing():
    stale_dict = {
        "num_arms": 7,
        "num_trials": 50,
        "entropy_coef": 0.01,  # renamed field from an older checkpoint format
        "not_a_real_field": 123,
    }
    config = config_from_dict(stale_dict)
    assert config.num_arms == 7
    assert config.num_trials == 50
    assert config.entropy_coef_start == TrainingConfig().entropy_coef_start  # fell back to default
    assert not hasattr(config, "entropy_coef")
    assert not hasattr(config, "not_a_real_field")


def test_entropy_anneals_from_start_to_end_over_training(tmp_path):
    config = TrainingConfig(
        num_arms=3, num_trials=8, d_model=16, n_heads=2, n_layers=2, d_ff=32,
        batch_size=4, minibatch_size=4, ppo_epochs=1, num_iterations=4,
        entropy_coef_start=0.05, entropy_coef_end=0.0,
    )
    _, history = train(config, run_dir=tmp_path / "run", log_every=0)
    entropy_coefs = [record["entropy_coef"] for record in history]
    assert entropy_coefs[0] == 0.05
    assert entropy_coefs[-1] == 0.0
    assert entropy_coefs == sorted(entropy_coefs, reverse=True)


def test_resume_continues_iteration_count_and_appends_history(tmp_path):
    run_dir = tmp_path / "run"
    base_config = TrainingConfig(
        num_arms=3, num_trials=8, d_model=16, n_heads=2, n_layers=2, d_ff=32,
        batch_size=4, minibatch_size=4, ppo_epochs=1, num_iterations=2, seed=0,
    )
    train(base_config, run_dir=run_dir, log_every=0)

    resumed_config = TrainingConfig(
        num_arms=3, num_trials=8, d_model=16, n_heads=2, n_layers=2, d_ff=32,
        batch_size=4, minibatch_size=4, ppo_epochs=1, num_iterations=3, seed=1,
    )
    _, history = train(
        resumed_config, run_dir=run_dir, log_every=0, resume_from=run_dir / "checkpoint.pt"
    )

    assert len(history) == 5  # 2 original + 3 additional
    assert [record["iteration"] for record in history] == [0, 1, 2, 3, 4]


def test_greedy_rollout_is_deterministic():
    policy = build_policy(TINY_CONFIG)
    policy.eval()
    device = torch.device("cpu")
    task_batch = sample_batch(
        "train", np.random.default_rng(7), TINY_CONFIG.batch_size, TINY_CONFIG.num_arms, TINY_CONFIG.num_trials
    )
    r1 = collect_rollout(policy, task_batch, device, np.random.default_rng(123), greedy=True)
    r2 = collect_rollout(policy, task_batch, device, np.random.default_rng(456), greedy=True)
    # Greedy action selection should be independent of the reward-noise RNG
    # stream used for sampling *rewards* -- only the observed rewards (and
    # hence subsequent actions, since they causally depend on reward noise)
    # may legitimately differ.
    assert torch.equal(r1.actions_taken[:, 0], r2.actions_taken[:, 0])
