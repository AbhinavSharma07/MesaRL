import torch

from model.transformer_policy import TransformerPolicy, TransformerPolicyConfig

NUM_ARMS = 5
MAX_TRIALS = 20


def make_policy(**overrides):
    config = TransformerPolicyConfig(
        num_arms=NUM_ARMS,
        max_trials=MAX_TRIALS,
        d_model=16,
        n_heads=2,
        n_layers=2,
        d_ff=32,
        **overrides,
    )
    return TransformerPolicy(config)


def make_history(batch, length, policy):
    prev_actions = torch.randint(0, NUM_ARMS, (batch, length))
    prev_actions[:, 0] = policy.no_prev_action
    prev_rewards = torch.randn(batch, length)
    return prev_actions, prev_rewards


def test_output_shapes():
    policy = make_policy()
    prev_actions, prev_rewards = make_history(4, 7, policy)
    logits, value, activations = policy(prev_actions, prev_rewards)
    assert logits.shape == (4, 7, NUM_ARMS)
    assert value.shape == (4, 7)
    assert activations is None


def test_return_activations_shapes():
    policy = make_policy()
    prev_actions, prev_rewards = make_history(3, 5, policy)
    _, _, activations = policy(prev_actions, prev_rewards, return_activations=True)
    assert len(activations["resid_per_layer"]) == 2  # n_layers
    assert activations["resid_per_layer"][0].shape == (3, 5, 16)
    assert len(activations["attn_weights_per_layer"]) == 2
    assert activations["attn_weights_per_layer"][0].shape == (3, 2, 5, 5)  # (B, H, L, L)
    assert activations["final_resid"].shape == (3, 5, 16)


def test_causal_masking_future_tokens_do_not_affect_past_outputs():
    """The core correctness property later interpretability phases depend on:
    output at position i must be identical regardless of what's stored at
    positions > i."""
    policy = make_policy()
    policy.eval()
    prev_actions, prev_rewards = make_history(2, 10, policy)

    with torch.no_grad():
        logits_full, value_full, _ = policy(prev_actions, prev_rewards)

    # Corrupt everything strictly after position 4.
    corrupted_actions = prev_actions.clone()
    corrupted_rewards = prev_rewards.clone()
    corrupted_actions[:, 5:] = torch.randint(0, NUM_ARMS, corrupted_actions[:, 5:].shape)
    corrupted_rewards[:, 5:] = torch.randn(corrupted_rewards[:, 5:].shape) * 100

    with torch.no_grad():
        logits_corrupt, value_corrupt, _ = policy(corrupted_actions, corrupted_rewards)

    assert torch.allclose(logits_full[:, :5], logits_corrupt[:, :5], atol=1e-5)
    assert torch.allclose(value_full[:, :5], value_corrupt[:, :5], atol=1e-5)
    # Sanity: corrupting the future should generally change at least the
    # corrupted positions' own outputs (guards against a no-op bug).
    assert not torch.allclose(logits_full[:, 5:], logits_corrupt[:, 5:], atol=1e-5)


def test_prefix_forward_matches_full_forward_at_shared_positions():
    """Running the model on a length-t prefix must give the same output at
    position t-1 as running it on the full length-L sequence -- this is what
    lets rollout collection grow the sequence one token at a time."""
    policy = make_policy()
    policy.eval()
    prev_actions, prev_rewards = make_history(2, 8, policy)

    with torch.no_grad():
        logits_full, value_full, _ = policy(prev_actions, prev_rewards)
        logits_prefix, value_prefix, _ = policy(
            prev_actions[:, :4], prev_rewards[:, :4]
        )

    assert torch.allclose(logits_full[:, :4], logits_prefix, atol=1e-5)
    assert torch.allclose(value_full[:, :4], value_prefix, atol=1e-5)


def test_act_returns_valid_action_distribution():
    policy = make_policy()
    prev_actions, prev_rewards = make_history(6, 3, policy)
    action, log_prob, value = policy.act(prev_actions, prev_rewards)
    assert action.shape == (6,)
    assert ((action >= 0) & (action < NUM_ARMS)).all()
    assert log_prob.shape == (6,)
    assert value.shape == (6,)


def test_act_greedy_is_deterministic_and_matches_argmax():
    policy = make_policy()
    policy.eval()
    prev_actions, prev_rewards = make_history(5, 4, policy)
    with torch.no_grad():
        logits, _, _ = policy(prev_actions, prev_rewards)
        expected = logits[:, -1, :].argmax(dim=-1)
        action, _, _ = policy.act(prev_actions, prev_rewards, greedy=True)
    assert torch.equal(action, expected)


def test_rejects_sequence_longer_than_max_trials():
    policy = make_policy()
    prev_actions, prev_rewards = make_history(2, MAX_TRIALS + 1, policy)
    try:
        policy(prev_actions, prev_rewards)
        assert False, "expected ValueError"
    except ValueError:
        pass
