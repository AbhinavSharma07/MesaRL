"""A small causal Transformer used as the meta-RL policy.

At trial i within an episode, the network's input is the pair (a_{i-1},
r_{i-1}) -- the action taken and reward received on the *previous* trial (a
sentinel "no previous action" token is used at i=0). Causal self-attention
means the network's output at position i depends only on positions <= i, i.e.
only on the history of (action, reward) pairs from trials 0..i-1 -- exactly
the information available to a bandit-playing agent deciding trial i's
action. Weights are shared across every meta-training task instance; any
in-episode adaptation must therefore come from computation the frozen
network performs over that history at inference time, not from further
gradient updates -- this is the emergent mesa-optimizer this project studies.

Kept intentionally small and hand-rolled (rather than nn.TransformerEncoder)
so later phases can cheaply read out the residual stream and per-head
attention weights for probing, activation patching, and induction-head
analysis.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerPolicyConfig:
    num_arms: int
    max_trials: int
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 3
    d_ff: int = 128
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TransformerPolicyConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (B, L, n_heads, head_dim)
        q = q.transpose(1, 2)  # (B, n_heads, L, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)  # (B, H, L, L)
        causal_mask = torch.triu(
            torch.ones(L, L, dtype=torch.bool, device=x.device), diagonal=1
        )
        attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, H, L, L)
        attn_weights = self.dropout(attn_weights)

        out = attn_weights @ v  # (B, H, L, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.proj(out)
        return out, attn_weights.detach()


class MLP(nn.Module):
    def __init__(self, config: TransformerPolicyConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    """Pre-LN transformer block: x + Attn(LN(x)), then x + MLP(LN(x))."""

    def __init__(self, config: TransformerPolicyConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_weights = self.attn(self.ln1(x))
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, attn_weights


class TransformerPolicy(nn.Module):
    """Causal Transformer actor-critic over (action, reward) history.

    NO_PREV_ACTION (== config.num_arms) is the sentinel action index used at
    trial 0, where there is no previous action yet.
    """

    def __init__(self, config: TransformerPolicyConfig):
        super().__init__()
        self.config = config
        self.action_embed = nn.Embedding(config.num_arms + 1, config.d_model // 2)
        self.reward_proj = nn.Linear(1, config.d_model - config.d_model // 2)
        self.pos_embed = nn.Embedding(config.max_trials, config.d_model)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.policy_head = nn.Linear(config.d_model, config.num_arms)
        self.value_head = nn.Linear(config.d_model, 1)

    @property
    def no_prev_action(self) -> int:
        return self.config.num_arms

    def forward(
        self,
        prev_actions: torch.Tensor,
        prev_rewards: torch.Tensor,
        return_activations: bool = False,
    ):
        """
        prev_actions: (B, L) long, prev_actions[:, 0] == no_prev_action.
        prev_rewards: (B, L) float, prev_rewards[:, 0] arbitrary (unused
            information-wise; the sentinel action already marks position 0).

        Returns (policy_logits (B, L, num_arms), value (B, L), activations)
        where activations is None unless return_activations=True, in which
        case it's a dict with "resid_per_layer" (list of (B, L, d_model),
        one per block, taken *after* that block) and "attn_weights_per_layer"
        (list of (B, n_heads, L, L), one per block).
        """
        B, L = prev_actions.shape
        if L > self.config.max_trials:
            raise ValueError(
                f"sequence length {L} exceeds max_trials {self.config.max_trials}"
            )

        action_emb = self.action_embed(prev_actions)  # (B, L, d_model//2)
        reward_emb = self.reward_proj(prev_rewards.unsqueeze(-1))  # (B, L, d_model - d_model//2)
        x = torch.cat([action_emb, reward_emb], dim=-1)  # (B, L, d_model)

        positions = torch.arange(L, device=prev_actions.device)
        x = x + self.pos_embed(positions).unsqueeze(0)

        resid_per_layer = [] if return_activations else None
        attn_weights_per_layer = [] if return_activations else None
        for block in self.blocks:
            x, attn_weights = block(x)
            if return_activations:
                resid_per_layer.append(x)
                attn_weights_per_layer.append(attn_weights)

        x = self.ln_f(x)
        policy_logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)

        activations = None
        if return_activations:
            activations = {
                "resid_per_layer": resid_per_layer,
                "attn_weights_per_layer": attn_weights_per_layer,
                "final_resid": x,
            }
        return policy_logits, value, activations

    def act(
        self,
        prev_actions: torch.Tensor,
        prev_rewards: torch.Tensor,
        rng: Optional[torch.Generator] = None,
        greedy: bool = False,
    ):
        """Convenience for rollout collection: samples an action for the
        *last* position in the given prefix, returning
        (action (B,), log_prob (B,), value (B,))."""
        policy_logits, value, _ = self.forward(prev_actions, prev_rewards)
        last_logits = policy_logits[:, -1, :]
        last_value = value[:, -1]
        dist = torch.distributions.Categorical(logits=last_logits)
        if greedy:
            action = last_logits.argmax(dim=-1)
        else:
            action = dist.sample() if rng is None else _sample_with_generator(dist, rng)
        log_prob = dist.log_prob(action)
        return action, log_prob, last_value


def _sample_with_generator(
    dist: torch.distributions.Categorical, rng: torch.Generator
) -> torch.Tensor:
    probs = dist.probs
    return torch.multinomial(probs, num_samples=1, generator=rng).squeeze(-1)
