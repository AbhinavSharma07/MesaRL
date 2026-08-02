# Project Details: MesaRL

## 0. Status: Feature-complete, single trained checkpoint analyzed

Every phase below (train, evaluate, compare, probe, patch, attention-circuit search, distribution shift, regime probe, audit agent, CLI, dashboard) is implemented and tested (90 tests). Findings describe one trained checkpoint (`runs/main`, committed for inspection/reproducibility), not a claim about meta-RL transformers in general.

## 1. Motivation and background

[Hubinger et al., 2019, "Risks from Learned Optimization in Advanced Machine Learning Systems"](https://arxiv.org/abs/1906.01820) introduces the concept of a **mesa-optimizer**: a learned model (produced by a "base optimizer," e.g. gradient descent) whose own forward pass implements a second, internal optimization process — with its own "mesa-objective" that can diverge from the base objective it was trained under. The paper cites meta-reinforcement-learning as one of the clearest empirically-demonstrated instances of this: train a recurrent or attention-based policy across many distinct task instances via ordinary RL, and the frozen network ends up implementing its own explore/exploit algorithm at inference time, because that's the only way to perform well across a whole task distribution ([Duan et al., 2016, "RL²: Fast Reinforcement Learning via Slow Reinforcement Learning"](https://arxiv.org/abs/1611.02779); [Wang et al., 2016, "Learning to Reinforcement Learn"](https://arxiv.org/abs/1611.05763)).

That specific effect is well-established. This project's contribution is going a step further: given that a mesa-optimizer has emerged, **what is it actually doing, mechanistically, and where does it fail?**

1. Which known bandit algorithm does its behavior resemble (or resemble least)?
2. Is its "belief" about the task linearly represented in the residual stream — and is that representation causally load-bearing, or merely correlated with behavior?
3. Does it contain anything like the induction-head circuits known to underlie in-context learning in language transformers?
4. How does it degrade under distribution shift?
5. Can a component distinguishing "training-like" from "held-out" conditions arise as a byproduct of ordinary training — a controlled, honest toy testbed for a component the same paper's *deceptive alignment* concern depends on — and if so, does it actually change behavior?

Using a Transformer specifically (rather than the LSTM/GRU architectures RL² originally used) is a deliberate choice: it's the same architecture class behind the finding that in-context learning in language models implements gradient-descent-like computation internally ([von Oswald et al., 2022, "Transformers Learn In-Context by Gradient Descent"](https://arxiv.org/abs/2212.07677)), so the interpretability tooling built here (residual-stream probing, activation patching, attention-circuit analysis) is the same toolkit used on language transformers, applied instead to sequential decision-making.

## 2. Method

### 2.1 Task family (`env/bandit_family.py`)

Each episode is a fresh k-armed Gaussian bandit: arm means ~ N(0, 1), independently per arm, per episode; rewards ~ N(arm_mean, obs_noise_std²) with obs_noise_std drawn per-episode from a narrow training range. `sample_batch(mode, ...)` supports:

- `train` / `regime_low_noise` — the training distribution.
- `ood_nonstationary` — arm means jump to a fresh draw at a random mid-episode change point.
- `ood_correlated` — arm means share a random per-episode offset instead of being drawn independently.
- `ood_wide_prior` — arm means drawn from a much wider prior.
- `regime_high_noise` — observation noise drawn from a range that never overlaps with training.

### 2.2 Model and training (`model/transformer_policy.py`, `training/train_meta_rl.py`)

A small causal Transformer (3 layers, 4 heads, d_model=64) consumes the in-episode sequence of `(previous action, previous reward)` pairs and outputs a policy distribution and value estimate at every trial. Causal masking guarantees the output at trial *i* depends only on trials `0..i-1` — exactly the information a bandit-playing agent has when choosing trial *i*'s action — and, crucially, means a single forward pass over a *completed* episode reproduces every trial's collection-time distribution, so PPO's log-prob/value recomputation only needs one forward pass per epoch, not one per timestep.

Trained via PPO + GAE across freshly sampled `train`-mode tasks. The first 500-iteration run converged on overall reward but showed *no* within-episode regret decrease (entropy stayed too high); resuming with linear entropy annealing (0.008 → 0.0005) for 1200 more iterations produced the adaptation signature reported below — a real tuning iteration, not a one-shot success (see the "genuinely evolved" commit history).

### 2.3 Analyses

- **Adaptation proof** (`eval/adaptation.py`): frozen-weight rollout on freshly sampled (hence held-out) tasks, regret compared against a memoryless random-action baseline.
- **Candidate-algorithm comparison** (`analysis/candidate_algorithms.py`, `analysis/compare.py`): Thompson sampling (posterior-sampling heuristic, not the intractable exact finite-horizon Bayes-optimal policy), UCB1, epsilon-greedy, win-stay-lose-shift — each run in "shadow mode" against the network's own realized history (never their own independent rollout, which would diverge onto a different trajectory and make a same-trial comparison meaningless). Compared via KL(candidate‖network) and action-agreement rate.
- **Belief probing** (`analysis/probes.py`): linear classifier predicting "which arm has the highest empirical mean so far" from a single layer's residual stream, checked against a majority-class baseline and a shuffled-label control. Uses an **episode-grouped train/val split** — an early bug (see §4) showed that a naive per-sample split leaks when many correlated samples come from few episodes.
- **Activation patching** (`analysis/patching.py`): a `forward_with_patch` helper re-implements the forward pass with a vector added to the residual stream at one layer/position, verified to exactly reproduce the unpatched forward pass when the patch is zero. The patch direction is `probe.weight[target] - probe.weight[current]` (a contrastive/concept-vector construction), compared against a same-magnitude random-direction control.
- **Attention-circuit search** (`analysis/attention_circuits.py`): for every (episode, query trial) pair, splits key positions into "same arm as about to be chosen again" vs. "different arm," and scores each head by the mean attention-weight gap between the two groups.
- **Distribution shift** (`shift/distribution_shift.py`): regret curves across all shift modes, plus a change-point-aligned "shock and recovery" curve for the non-stationary case.
- **Regime probe** (`regime_probe/train_deploy_probe.py`): linear probe decoding `regime_low_noise` vs. `regime_high_noise` from only the first 15 trials' activations (trial 0 excluded — it carries zero regime information, being a deterministic function of the sentinel token), paired with behavioral metrics (action entropy, distinct arms tried) to check for a qualitative behavioral divergence, not just a decodable correlate.
- **Audit agent** (`audit_agent/`): a LangGraph pipeline (ingest → draft → critique → revise) that reads every JSON result above and writes a natural-language report, with a dedicated critique stage whose job is to catch overclaiming against the raw numbers.

## 3. Results

See the README's [Key findings](README.md#key-findings) for the headline numbers with plots. In full:

| Analysis | Headline number |
|---|---|
| Adaptation | Regret 1.15 → 0.57 (trial 70), 41% reduction; random baseline flat at ~1.16 |
| Candidate comparison | Closest: epsilon-greedy (73.7% agreement); farthest: UCB1 (0.1% agreement) |
| Belief probing | 91.5% val accuracy (layer 1) vs. ~50.7% majority/shuffled control |
| Activation patching | Favored arm: +0.51 prob shift vs. -0.06 control (scale 50). Avoided arm: comparable to control at every scale ≤ 100 |
| Attention circuits | No induction-head found; early layers show *negative* same-arm bias (down to -0.29) |
| Distribution shift | Non-stationary shock: regret 0.58 → 1.14 instantly, only to 1.08 after 20 trials |
| Regime probe | 85% decodable (layer 1) vs. ~50% control; entropy/diversity gap between regimes: negligible |
| Arm-identity bias | Two of five arms played ~0.005% / ~0.13% of the time; two favorites ~44% / ~32% |

## 4. Notable debugging story: train/val leakage in the regime probe

The first regime-probe run showed a shuffled-label control at 58-65% accuracy — it should sit at chance (~50%). Root cause: with only a 15-trial window, many samples per episode are highly correlated, and the original per-sample train/val split let same-episode samples land on both sides, so a probe could key off "which episode is this" rather than genuine regime signal — and since val samples share their true label with same-episode train samples, this inflated accuracy even under label shuffling. Fixed by adding `group_aware_train_val_split` to `analysis/probes.py` (groups by episode id), verified with a synthetic test that reproduces the failure mode and confirms the fix. The *real* probe accuracy barely moved (83.8% → 85.5%) — the original signal was genuine; only the control was broken.

## 5. Known limitations

See the README's [Known limitations](README.md#known-limitations) section — repeated here for completeness: the late-episode regret anomaly (likely a PPO/GAE boundary artifact), the arm-identity bias, the non-comparability of absolute regret across distribution-shift modes with different arm-mean spread, the audit agent being untested against a real LLM in this build environment, and the single-seed/single-checkpoint scope of every finding.

## 6. Possible extensions

- Root-cause the late-episode regret anomaly directly (e.g. compare GAE with vs. without a bootstrapped terminal value).
- Retrain with an entropy floor instead of full annealing to test whether the arm-identity bias is avoidable.
- Extend the candidate-algorithm set with a true finite-horizon-optimal (dynamic-programming) policy for small arm counts, as a stronger ceiling than Thompson sampling.
- Repeat the whole pipeline across multiple seeds to check how much of each finding is seed-specific vs. robust.
