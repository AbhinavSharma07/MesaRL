# Project Details: MesaRL

## 0. Status: Feature-complete, robustness-checked across 3 seeds

Every phase below (train, evaluate, compare, probe, patch, attention-circuit search, distribution shift, regime probe, audit agent, CLI, dashboard) is implemented and tested (94 tests). Headline findings are from `runs/main` (committed for inspection/reproducibility); §4 reports which of them replicate across 2 additional independently-trained seeds (`runs/seed1`, `runs/seed2`, not committed — retrain with `python cli.py train --seed 1`/`2`).

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

## 4. Robustness across seeds

`runs/main` used a messy two-phase recipe (500 iterations at a static entropy coefficient, discovered not to adapt at all, then resumed for 1200 more with entropy annealing added — see §5.2). To check what actually generalizes, `runs/seed1` and `runs/seed2` were trained from scratch with a clean single-phase anneal (`entropy_coef_start=0.02` → `entropy_coef_end=0.0005` over the full 1700 iterations, `--seed 1`/`--seed 2`), and all 7 analyses re-run against each via `python cli.py all --run-dir runs/seedN`.

| Analysis | main | seed1 | seed2 |
|---|---|---|---|
| Cumulative regret (adaptation) | 67.3 | 31.1 | 53.3 |
| Improvement-to-best | 41.4% | 68.4% | 52.6% |
| Closest candidate algorithm | Epsilon-greedy | Epsilon-greedy | Win-stay-lose-shift |
| UCB1 agreement (always worst) | 0.1% | 5.7% | 0.7% |
| Belief-probe accuracy (best layer) | 91.6% | 88.1% | 83.2% |
| Belief-probe majority/shuffled baseline | 50.7% / 50.7% | 25.5% / 25.4% | 27.1% / 25.9% |
| Patching: prob_shift vs. control (default scale) | 0.0014 vs 0.0007 | 0.0784 vs 0.0035 | 0.0246 vs -0.0016 |
| Peak same-arm attention bias | 0.043 (layer 2) | 0.041 (layer 1) | 0.035 (layer 2) |
| Distribution-shift ordering | correlated (29.5) < train (65.9) < non-stat (89.3) < wide-prior (297.1) | 21.6 < 29.0 < 62.3 < 209.9 | 27.6 < 51.3 < 73.0 < 250.3 |
| Non-stationary shock (before → at-change → recovery) | 0.60 → 1.14 → 1.08 | 0.21 → 1.18 → 1.01 | 0.40 → 1.18 → 1.00 |
| Regime-probe accuracy | 85.5% | 80.8% | 82.5% |
| Regime-probe entropy gap (high − low noise) | -0.0086 | -0.0124 | -0.0574 |
| Regime-probe distinct-arms gap | +0.0500 | -0.1850 | -0.0150 |
| Action frequency (arms 0-4) | 22.5/45.5/0.01/32.0/0.04 % | 19.3/15.8/24.1/20.9/19.8 % | 2.8/23.9/30.2/23.3/19.9 % |

**Robust (holds in all 3 seeds):** adaptation always occurs (magnitude varies 2x+); UCB1 is always the worst-matching candidate; belief is always strongly linearly decodable (always far above that seed's own baseline); the causal steering effect always beats its control; attention bias is always weak (~0.03-0.04, no real induction head); the distribution-shift ordering and the shock/incomplete-recovery pattern always hold; regime-probe decodability and its entropy-gap *direction* always hold.

**Not robust (seed/recipe-specific):** the exact closest-candidate identity (epsilon-greedy vs. win-stay-lose-shift — both simple exploit-heavy heuristics, so this is a soft distinction anyway); the regime-probe's distinct-arms-tried gap (sign flips: +0.05, -0.19, -0.02); and most importantly, **the severe arm-identity bias** — `main`'s near-total abandonment of two arms is specific to its flawed two-phase training recipe, not the architecture or task. `seed1`, trained with a clean anneal from the start, shows almost no bias at all and consequently the *best* adaptation of the three — evidence that the bias was actively hurting performance, not a neutral quirk.

## 5. Notable debugging stories

### 5.1 Train/val leakage in the regime probe

The first regime-probe run showed a shuffled-label control at 58-65% accuracy — it should sit at chance (~50%). Root cause: with only a 15-trial window, many samples per episode are highly correlated, and the original per-sample train/val split let same-episode samples land on both sides, so a probe could key off "which episode is this" rather than genuine regime signal — and since val samples share their true label with same-episode train samples, this inflated accuracy even under label shuffling. Fixed by adding `group_aware_train_val_split` to `analysis/probes.py` (groups by episode id), verified with a synthetic test that reproduces the failure mode and confirms the fix. The *real* probe accuracy barely moved (83.8% → 85.5%) — the original signal was genuine; only the control was broken.

### 5.2 Getting the audit agent to actually run

Running `audit_agent/` against a real LLM (Groq free tier) surfaced a chain of real issues, fixed one at a time rather than papered over:

1. **413 request-too-large.** The draft prompt (all 8 JSON result files, verbatim) exceeded Groq's 8000 TPM free-tier limit. Fixed by stripping per-trial arrays (>20 elements) and the bulky `patching_sweep` field before they reach the prompt (`analysis/probes.py`'s style of trimming, applied in `audit_agent/graph.py`).
2. **Still too large at the critique/revise stages**, which carry cumulative context (metrics + draft, then + critique). Root cause: no explicit `max_tokens` cap meant a large default completion budget was reserved against the same per-minute limit. Fixed with per-stage token budgets (draft 1200, critique 400, revise 2500) via LangChain's `.bind(max_tokens=...)`, plus explicit word-count instructions in the prompts.
3. **`UnicodeEncodeError` on Windows** writing/reading the report and echoing it to the console — Windows defaults to `cp1252`, and the LLM's output contained Unicode punctuation (en/em dashes, non-breaking hyphens). Fixed by forcing `encoding="utf-8"` on every read/write of LLM-authored text, and reconfiguring stdout in `cli.py`.
4. **Real inaccuracies survived the critique stage.** An early successful run mistranscribed a 0.51 probability shift as "0.5%" (misreading a 0-1 probability delta as a percentage) and miscounted "top 5" attention heads as "all layer 2" when only 4 of 5 were. Neither is random noise — both are systematic small-model-under-tight-budget failure modes. Fixed two ways: (a) `analysis/attention_circuits.py` now computes a correct, pre-verified interpretation string (`build_attention_interpretation`) instead of making the LLM count/compare raw numbers itself, and (b) the draft/critique prompts explicitly warn about 0-1-scale values being misread as percentages. Re-running after both fixes produced a report with no factual errors found on manual line-by-line verification against the source JSON.

## 6. Known limitations

See the README's [Known limitations](README.md#known-limitations) section — repeated here for completeness: the late-episode regret anomaly (likely a PPO/GAE boundary artifact, present in all 3 seeds), the non-comparability of absolute regret across distribution-shift modes with different arm-mean spread, and the fact that 3 seeds is a first-pass robustness check, not a statistical guarantee.

## 7. Possible extensions

- Root-cause the late-episode regret anomaly directly (e.g. compare GAE with vs. without a bootstrapped terminal value).
- Run more seeds to firm up the robust/not-robust split in §4 with a larger sample.
- Extend the candidate-algorithm set with a true finite-horizon-optimal (dynamic-programming) policy for small arm counts, as a stronger ceiling than Thompson sampling.
- Speed up rollout collection with a KV-cache for the Transformer's incremental decoding — every rollout currently recomputes the full growing sequence from scratch each trial, which is why each seed took ~2.5-4 hours to train.
