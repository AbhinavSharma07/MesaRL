import numpy as np
import pytest

from env.bandit_family import (
    HIGH_NOISE_RANGE,
    TRAIN_NOISE_RANGE,
    TRAIN_NUM_ARMS,
    TRAIN_PRIOR_STD,
    WIDE_PRIOR_MULTIPLIER,
    sample_batch,
)

SEED = 0
BATCH = 4096
NUM_TRIALS = 100


def rng():
    return np.random.default_rng(SEED)


def test_train_mode_matches_prior_and_noise_range():
    batch = sample_batch("train", rng(), BATCH)
    assert batch.arm_means.shape == (BATCH, TRAIN_NUM_ARMS)
    assert abs(batch.arm_means.mean()) < 0.05
    assert abs(batch.arm_means.std() - TRAIN_PRIOR_STD) < 0.05
    assert batch.obs_noise_std.min() >= TRAIN_NOISE_RANGE[0]
    assert batch.obs_noise_std.max() <= TRAIN_NOISE_RANGE[1]
    assert batch.change_points is None


def test_regime_low_noise_matches_train_distribution():
    train = sample_batch("train", np.random.default_rng(1), BATCH)
    low = sample_batch("regime_low_noise", np.random.default_rng(1), BATCH)
    # Same sampling recipe -> statistically indistinguishable at this batch size.
    assert abs(train.arm_means.std() - low.arm_means.std()) < 0.05
    assert abs(train.obs_noise_std.mean() - low.obs_noise_std.mean()) < 0.05


def test_regime_high_noise_is_disjoint_from_training_range():
    batch = sample_batch("regime_high_noise", rng(), BATCH)
    assert batch.obs_noise_std.min() >= HIGH_NOISE_RANGE[0]
    assert batch.obs_noise_std.max() <= HIGH_NOISE_RANGE[1]
    # No overlap with the training noise range at all.
    assert HIGH_NOISE_RANGE[0] > TRAIN_NOISE_RANGE[1]


def test_ood_wide_prior_has_larger_spread():
    train = sample_batch("train", rng(), BATCH)
    wide = sample_batch("ood_wide_prior", rng(), BATCH)
    assert wide.arm_means.std() > train.arm_means.std() * (WIDE_PRIOR_MULTIPLIER - 1)


def test_ood_correlated_arms_share_variance():
    batch = sample_batch("ood_correlated", rng(), BATCH)
    # Arms within a task should be more correlated than across independent tasks.
    within_task_corr = np.mean(
        [np.corrcoef(batch.arm_means[i], batch.arm_means[i])[0, 1] for i in range(8)]
    )
    assert within_task_corr == pytest.approx(1.0)
    cross_arm_covariance = np.cov(batch.arm_means.T)
    off_diag = cross_arm_covariance[~np.eye(TRAIN_NUM_ARMS, dtype=bool)]
    assert off_diag.mean() > 0  # positive correlation from the shared offset


def test_ood_nonstationary_change_point_switches_optimal_arm_distribution():
    batch = sample_batch("ood_nonstationary", rng(), BATCH, num_trials=NUM_TRIALS)
    assert batch.change_points is not None
    assert batch.change_points.min() >= 1
    assert batch.change_points.max() < NUM_TRIALS

    pre = batch.optimal_arm_at(0)
    post = batch.optimal_arm_at(NUM_TRIALS - 1)
    # Independently redrawn post-change means -> optimal arm should differ for
    # most tasks (not guaranteed for all, since ties/redraws can coincide).
    assert (pre != post).mean() > 0.5


def test_means_at_before_and_after_change_point():
    batch = sample_batch("ood_nonstationary", rng(), 8, num_trials=NUM_TRIALS)
    change = batch.change_points[0]
    means_before = batch.means_at(change - 1)
    means_after = batch.means_at(change)
    assert np.array_equal(means_before[0], batch.arm_means[0])
    assert np.array_equal(means_after[0], batch.post_change_means[0])


def test_reward_is_deterministic_given_seeded_rng():
    batch = sample_batch("train", rng(), 16)
    actions = np.zeros(16, dtype=int)
    r1 = batch.reward(actions, 0, np.random.default_rng(42))
    r2 = batch.reward(actions, 0, np.random.default_rng(42))
    assert np.array_equal(r1, r2)


def test_reward_shape_and_optimal_mean_shape():
    batch = sample_batch("train", rng(), 32)
    actions = np.random.default_rng(2).integers(0, TRAIN_NUM_ARMS, size=32)
    rewards = batch.reward(actions, 0, np.random.default_rng(3))
    assert rewards.shape == (32,)
    assert batch.optimal_mean_at(0).shape == (32,)
    assert batch.optimal_arm_at(0).shape == (32,)


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        sample_batch("not_a_real_mode", rng(), 4)


def test_mismatched_batch_shapes_raise():
    from env.bandit_family import BanditTaskBatch

    with pytest.raises(ValueError):
        BanditTaskBatch(
            arm_means=np.zeros((4, 5)), obs_noise_std=np.zeros(3)
        )
