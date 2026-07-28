import numpy as np

from analysis.candidate_algorithms import (
    epsilon_greedy_distribution,
    thompson_sampling_distribution,
    ucb1_distribution,
    win_stay_lose_shift_distribution,
)

NUM_ARMS = 4
NUM_TRIALS = 12
BATCH = 20


def make_history(seed=0):
    rng = np.random.default_rng(seed)
    actions = rng.integers(0, NUM_ARMS, size=(BATCH, NUM_TRIALS))
    rewards = rng.normal(0, 1, size=(BATCH, NUM_TRIALS))
    return actions, rewards


def assert_valid_distribution(dist, batch, trials, arms):
    assert dist.shape == (batch, trials, arms)
    assert np.all(dist >= -1e-8)
    assert np.allclose(dist.sum(axis=-1), 1.0, atol=1e-6)


def test_thompson_sampling_shape_and_normalization():
    actions, rewards = make_history()
    dist = thompson_sampling_distribution(actions, rewards, NUM_ARMS, num_mc_samples=200)
    assert_valid_distribution(dist, BATCH, NUM_TRIALS, NUM_ARMS)


def test_thompson_sampling_concentrates_on_high_mean_arm():
    # Arm 0 gets consistently high reward, others consistently low/negative --
    # posterior should concentrate heavily on arm 0 by the end.
    actions = np.zeros((5, 30), dtype=int)
    rewards = np.zeros((5, 30))
    for t in range(30):
        actions[:, t] = t % 4  # cycle through all arms equally
        rewards[:, t] = 5.0 if actions[0, t] == 0 else -5.0
    dist = thompson_sampling_distribution(actions, rewards, 4, num_mc_samples=1000, seed=1)
    assert dist[:, -1, 0].mean() > 0.9


def test_ucb1_forces_unpulled_arms_first():
    # ucb1_distribution runs in "shadow" mode against an externally-realized
    # history (see module docstring) -- it does NOT drive its own
    # trajectory, so the *specific* arm it picks each trial depends on the
    # given (here: random and unrelated) history. What must always hold
    # regardless of that history is the invariant tested here: whenever some
    # arm has zero pulls so far, UCB1's predicted arm must be one of them
    # (infinite bonus beats any finite score).
    actions, rewards = make_history()
    dist = ucb1_distribution(actions, rewards, NUM_ARMS)
    assert_valid_distribution(dist, BATCH, NUM_TRIALS, NUM_ARMS)

    # At trial 0, every arm is unpulled -> tie-broken deterministically to arm 0.
    assert (dist[:, 0, :].argmax(axis=-1) == 0).all()

    counts = np.zeros((BATCH, NUM_ARMS))
    for t in range(NUM_TRIALS):
        predicted = dist[:, t, :].argmax(axis=-1)
        for b in range(BATCH):
            if counts[b].min() == 0:
                assert counts[b, predicted[b]] == 0
        counts[np.arange(BATCH), actions[:, t]] += 1


def test_ucb1_is_one_hot_every_trial():
    actions, rewards = make_history()
    dist = ucb1_distribution(actions, rewards, NUM_ARMS)
    assert np.allclose(dist.max(axis=-1), 1.0)


def test_epsilon_greedy_is_uniform_before_any_history():
    actions, rewards = make_history()
    dist = epsilon_greedy_distribution(actions, rewards, NUM_ARMS, epsilon=0.2)
    assert np.allclose(dist[:, 0, :], 1.0 / NUM_ARMS)


def test_epsilon_greedy_matches_formula_after_one_observation():
    actions = np.zeros((3, 5), dtype=int)
    rewards = np.ones((3, 5))
    dist = epsilon_greedy_distribution(actions, rewards, NUM_ARMS, epsilon=0.2)
    # At trial 1, arm 0 has one observation (reward 1), others have none (mean 0)
    # -> arm 0 is greedy-best.
    expected_best = 0.2 / NUM_ARMS + 0.8
    expected_other = 0.2 / NUM_ARMS
    assert np.allclose(dist[:, 1, 0], expected_best)
    assert np.allclose(dist[:, 1, 1], expected_other)


def test_win_stay_lose_shift_uniform_at_trial_zero():
    actions, rewards = make_history()
    dist = win_stay_lose_shift_distribution(actions, rewards, NUM_ARMS)
    assert np.allclose(dist[:, 0, :], 1.0 / NUM_ARMS)


def test_win_stay_lose_shift_win_case_is_one_hot_on_previous_arm():
    actions = np.array([[2, 0, 0]])
    rewards = np.array([[1.0, 0.0, 0.0]])  # trial 0 reward = +1.0 (a win)
    dist = win_stay_lose_shift_distribution(actions, rewards, NUM_ARMS, win_threshold=0.0)
    expected = np.zeros(NUM_ARMS)
    expected[2] = 1.0
    assert np.allclose(dist[0, 1, :], expected)


def test_win_stay_lose_shift_lose_case_is_uniform_over_others():
    actions = np.array([[2, 0, 0]])
    rewards = np.array([[-1.0, 0.0, 0.0]])  # trial 0 reward = -1.0 (a loss)
    dist = win_stay_lose_shift_distribution(actions, rewards, NUM_ARMS, win_threshold=0.0)
    expected = np.full(NUM_ARMS, 1.0 / (NUM_ARMS - 1))
    expected[2] = 0.0
    assert np.allclose(dist[0, 1, :], expected)


def test_all_candidates_produce_valid_distributions_on_shared_history():
    actions, rewards = make_history(seed=5)
    for dist in [
        thompson_sampling_distribution(actions, rewards, NUM_ARMS, num_mc_samples=100),
        ucb1_distribution(actions, rewards, NUM_ARMS),
        epsilon_greedy_distribution(actions, rewards, NUM_ARMS),
        win_stay_lose_shift_distribution(actions, rewards, NUM_ARMS),
    ]:
        assert_valid_distribution(dist, BATCH, NUM_TRIALS, NUM_ARMS)
