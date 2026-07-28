"""Hand-designed candidate bandit algorithms ("what known algorithm does the
mesa-optimizer resemble?", see analysis/compare.py). Each runs in "shadow"
mode against the network's realized history rather than its own rollout, and
none see the task's true arm means or noise std. "Thompson sampling" is a
near-Bayes-optimal heuristic, not the exact (intractable) Bayes-optimal policy."""

import numpy as np


def _running_stats_update(counts: np.ndarray, sums: np.ndarray, actions: np.ndarray, rewards: np.ndarray) -> None:
    """In-place: counts/sums += one observation of (actions[b], rewards[b]) per row b."""
    batch_idx = np.arange(counts.shape[0])
    np.add.at(counts, (batch_idx, actions), 1)
    np.add.at(sums, (batch_idx, actions), rewards)


def thompson_sampling_distribution(
    actions_taken: np.ndarray,
    rewards_obtained: np.ndarray,
    num_arms: int,
    prior_std: float = 1.0,
    assumed_obs_noise_std: float = 1.0,
    num_mc_samples: int = 400,
    seed: int = 0,
) -> np.ndarray:
    """(B, T, K) distribution via Monte Carlo posterior sampling under a
    Normal-Normal conjugate model: arm means ~ N(0, prior_std^2), each
    observation ~ N(arm_mean, assumed_obs_noise_std^2)."""
    rng = np.random.default_rng(seed)
    B, T = actions_taken.shape
    counts = np.zeros((B, num_arms))
    sums = np.zeros((B, num_arms))
    prior_precision = 1.0 / (prior_std**2)
    obs_precision = 1.0 / (assumed_obs_noise_std**2)

    distributions = np.zeros((B, T, num_arms))
    for t in range(T):
        posterior_precision = prior_precision + counts * obs_precision
        empirical_mean = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
        posterior_mean = (counts * obs_precision * empirical_mean) / posterior_precision
        posterior_std = np.sqrt(1.0 / posterior_precision)

        samples = rng.normal(
            loc=posterior_mean[:, None, :],
            scale=posterior_std[:, None, :],
            size=(B, num_mc_samples, num_arms),
        )
        chosen = samples.argmax(axis=-1)  # (B, num_mc_samples)
        counts_per_arm = np.stack(
            [(chosen == k).sum(axis=-1) for k in range(num_arms)], axis=-1
        )
        distributions[:, t, :] = counts_per_arm / num_mc_samples

        _running_stats_update(counts, sums, actions_taken[:, t], rewards_obtained[:, t])

    return distributions


def ucb1_distribution(
    actions_taken: np.ndarray, rewards_obtained: np.ndarray, num_arms: int, ucb_c: float = 1.0
) -> np.ndarray:
    """(B, T, K) one-hot distribution: UCB1 is deterministic given its
    history. Arms never yet pulled are forced first (infinite bonus)."""
    B, T = actions_taken.shape
    counts = np.zeros((B, num_arms))
    sums = np.zeros((B, num_arms))
    distributions = np.zeros((B, T, num_arms))
    batch_idx = np.arange(B)

    for t in range(T):
        means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
        with np.errstate(divide="ignore"):
            bonus = ucb_c * np.sqrt(2 * np.log(t + 2) / np.maximum(counts, 1e-8))
        scores = means + bonus
        scores = np.where(counts == 0, np.inf, scores)

        best = scores.argmax(axis=1)
        distributions[batch_idx, t, best] = 1.0

        _running_stats_update(counts, sums, actions_taken[:, t], rewards_obtained[:, t])

    return distributions


def epsilon_greedy_distribution(
    actions_taken: np.ndarray, rewards_obtained: np.ndarray, num_arms: int, epsilon: float = 0.1
) -> np.ndarray:
    """(B, T, K) distribution: (1-epsilon) on the empirical-mean-greedy arm,
    epsilon spread uniformly over all arms. Before any arm has been pulled,
    falls back to pure uniform (no basis for a greedy choice yet)."""
    B, T = actions_taken.shape
    counts = np.zeros((B, num_arms))
    sums = np.zeros((B, num_arms))
    distributions = np.zeros((B, T, num_arms))
    batch_idx = np.arange(B)

    for t in range(T):
        means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
        best = means.argmax(axis=1)

        dist = np.full((B, num_arms), epsilon / num_arms)
        dist[batch_idx, best] += 1.0 - epsilon

        never_pulled_any = counts.sum(axis=1) == 0
        dist[never_pulled_any] = 1.0 / num_arms
        distributions[:, t, :] = dist

        _running_stats_update(counts, sums, actions_taken[:, t], rewards_obtained[:, t])

    return distributions


def win_stay_lose_shift_distribution(
    actions_taken: np.ndarray, rewards_obtained: np.ndarray, num_arms: int, win_threshold: float = 0.0
) -> np.ndarray:
    """(B, T, K) distribution: repeat the previous arm if its reward beat
    win_threshold ("win"), otherwise spread uniformly over every OTHER arm
    ("lose"). Trial 0 has no previous action, so it's uniform over all arms."""
    B, T = actions_taken.shape
    distributions = np.zeros((B, T, num_arms))
    batch_idx = np.arange(B)

    distributions[:, 0, :] = 1.0 / num_arms
    for t in range(1, T):
        prev_action = actions_taken[:, t - 1]
        prev_reward = rewards_obtained[:, t - 1]
        win = prev_reward > win_threshold

        dist = np.full((B, num_arms), 1.0 / (num_arms - 1))
        dist[batch_idx, prev_action] = 0.0  # zero out own previous arm (lose-case default)

        win_idx = batch_idx[win]
        dist[win_idx, :] = 0.0
        dist[win_idx, prev_action[win_idx]] = 1.0

        distributions[:, t, :] = dist

    return distributions
