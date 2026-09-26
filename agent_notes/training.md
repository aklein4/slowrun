# Training constraints

Read for changes to [loss-budget training](../research/loss_budget/README.md).

- Preserve the pinned gated architecture and distinguish it from newer root
  leaderboard entries. Test full-size dimensions for memory/stability claims.
- Keep unchanged upstream comparison scripts in root [baselines/](../baselines/README.md),
  outside experiment folders, and record their pinned commits there.
- Preserve original code structure and initialization wherever possible; keep
  the exact source available for comparison. Put new behavior at explicit
  integration points and avoid incidental renames, abstraction or cleanup.
- Keep the submission self-contained in one `train.py`; helpers for preparing
  experiments or comparing sources belong outside the implementation.
- Follow the paper’s fused MD update: positive softplus row/column gains at one,
  Frobenius matrix spheres and unit embedding-direction rows; clip averaged fused
  gradients at global norm 1 before splitting. All groups have zero weight decay.
  Only Muon directions receive noise;
  embedding/unembedding directions and all scales use AdamW without noise.
- Dropout is removed at the user's request. Preserve the original residual
  wrapper names as identities to keep the architecture diff small.
- The loss controller sees the mean over every accumulated training microbatch,
  never held-out loss. A loss budget is a feedback target, not a hard guarantee.
- Keep the experiment simple: no resume infrastructure or extra execution modes.
  Sampling times refer to the planned optimizer-step horizon, not epoch counts.
- Ensembles average categorical probabilities. Never substitute weight/logit
  averages or count repeated validation batches as independent evidence.
- Approximate Bayesian motivation is not proof of a posterior invariant measure.
  Scale priors, adaptation, momentum and geometric corrections matter.
