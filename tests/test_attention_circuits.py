import numpy as np

from analysis.attention_circuits import (
    AttentionDataset,
    build_attention_interpretation,
    collect_attention_dataset,
    same_arm_attention_bias,
    top_induction_head_candidates,
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


def test_collect_attention_dataset_shapes():
    policy = build_policy(TINY_CONFIG)
    dataset = collect_attention_dataset(policy, TINY_CONFIG, num_episodes=6, seed=0)
    assert len(dataset.attn_weights_per_layer) == TINY_CONFIG.n_layers
    for attn in dataset.attn_weights_per_layer:
        assert attn.shape == (6, TINY_CONFIG.n_heads, TINY_CONFIG.num_trials, TINY_CONFIG.num_trials)
    assert dataset.hist_actions.shape == (6, TINY_CONFIG.num_trials)
    assert dataset.actions_taken.shape == (6, TINY_CONFIG.num_trials)


def test_same_arm_attention_bias_detects_planted_induction_head():
    # One head (head 0) is deliberately built to place all its attention
    # mass on key positions describing the same arm the network is about to
    # pick again; head 1 attends uniformly. The bias metric must clearly
    # separate the two.
    num_arms = 2
    T = 6
    B = 1
    n_heads = 2

    hist_actions = np.array([[num_arms, 0, 1, 0, 1, 0]])  # sentinel, then actions_{t-1}
    actions_taken = np.array([[0, 1, 0, 1, 0, 1]])

    attn = np.zeros((B, n_heads, T, T), dtype=np.float32)
    for q in range(2, T):
        chosen = actions_taken[0, q]
        key_range = hist_actions[0, : q + 1]
        same_mask = key_range == chosen
        # Head 0: all attention mass split evenly across same-arm keys.
        n_same = same_mask.sum()
        attn[0, 0, q, : q + 1] = np.where(same_mask, 1.0 / n_same, 0.0)
        # Head 1: uniform attention over all valid keys.
        attn[0, 1, q, : q + 1] = 1.0 / (q + 1)

    dataset = AttentionDataset(
        attn_weights_per_layer=[attn], hist_actions=hist_actions, actions_taken=actions_taken
    )
    bias = same_arm_attention_bias(dataset, min_query_position=2)

    assert bias.shape == (1, 2)
    assert bias[0, 0] > 0.5  # head 0: strong induction-like bias
    assert abs(bias[0, 1]) < 1e-6  # head 1: uniform -> no bias
    assert bias[0, 0] > bias[0, 1]


def test_top_induction_head_candidates_ranking():
    bias = np.array([[0.1, 0.9], [0.5, 0.2]])
    top = top_induction_head_candidates(bias, top_k=2)
    assert top[0]["layer"] == 0 and top[0]["head"] == 1
    assert top[0]["same_arm_attention_bias"] == 0.9
    assert top[1]["same_arm_attention_bias"] == 0.5


def test_same_arm_attention_bias_returns_zero_when_no_history_matches():
    # No prior same-arm keys ever exist -> every query is skipped -> bias
    # stays at its initialized zero rather than raising or NaN-ing.
    hist_actions = np.full((1, 5), 2)  # sentinel-only, arm 2 never appears
    actions_taken = np.zeros((1, 5), dtype=int)
    attn = np.zeros((1, 1, 5, 5), dtype=np.float32)
    dataset = AttentionDataset(attn_weights_per_layer=[attn], hist_actions=hist_actions, actions_taken=actions_taken)
    bias = same_arm_attention_bias(dataset, min_query_position=1)
    assert np.allclose(bias, 0.0)


def test_build_attention_interpretation_reports_correct_layer_breakdown_and_extremum():
    # Reproduces the real run's shape: layer 2 dominates the positive top-k,
    # but layers 0-1 have the largest-magnitude bias, and it's negative.
    bias = np.array(
        [
            [0.004, -0.035, -0.291, -0.008],
            [-0.019, -0.200, -0.201, -0.018],
            [0.042, 0.012, 0.030, 0.014],
        ]
    )
    candidates = top_induction_head_candidates(bias, top_k=5)
    text = build_attention_interpretation(bias, candidates)

    assert "4 of 5 are in layer 2" in text
    assert "layer(s) [0]" in text
    assert "layer 0 head 2" in text  # the true global minimum, not a top-candidate layer
    assert "No clear induction-head circuit" in text
