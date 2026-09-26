# Loss-budget Gaussian noise

This single-file submission starts from the limited-compute **Add gating per
attention head** record (`52e7441`). Compare [train.py](train.py) directly with
[the unchanged baseline](../../baselines/gated_attention.py):

```bash
git diff --no-index baselines/gated_attention.py research/loss_budget/train.py
python agent_scripts/compare_loss_budget_baseline.py --section muon
```

The original model layout, distributed optimizer structure, data loader,
CLI names and top-level training loop are retained. Muon now uses the paper’s fixed shape scale, constant momentum and no variance
renormalization. Dropout is removed. There are
1,399,476,473 parameters: 30 layers, width 1792, 14 heads and 2048-token context.
Keep this file simple and close to the baseline; resume infrastructure and
additional execution modes are deliberately outside this submission's scope.

## Run

Use the baseline CUDA dependencies, with FlashAttention 3 or the FlashAttention 2
fallback. Launch with `torchrun`, including on one GPU, because the original
optimizer uses distributed collectives. The default remains compiled; the test
environment needs `--no-compile` because its Torch/Triton versions are incompatible.
The implementation imports no local helper modules.

```bash
WANDB_MODE=disabled torchrun --standalone --nproc_per_node=1 \
  research/loss_budget/train.py --no-compile \
  --input_bin fineweb_data/fineweb_train.pt \
  --input_val_bin fineweb_data/fineweb_val.pt \
  --num-epochs 40 --loss-target 3.5 \
  --checkpoint-dir local_data/runs/loss-budget \
  --save-result local_data/runs/loss-budget/result.json
```

Use a fresh checkpoint directory for each run. `WANDB_MODE=disabled` disables
tracking uploads; otherwise the baseline's W&B logging applies. Data can use the
record's packed format or the current preparation script's flat `tokens` format.
Validation is capped at the number of available batches. Training uses the
baseline's epoch shuffling and drops incomplete device batches.

Default peak matrix/input-embedding/output-embedding LRs are .008/.003/.001.
Gains follow their associated direction LR. Adam uses betas (.9, .99), epsilon
1e-8; Muon uses constant .95 Nesterov momentum and five polar iterations.
The original residual/skip/x0 learning rates remain architecture-specific.
Every group decays linearly from its peak to an absolute LR of 1e-8, with no
warmup. The .008 matrix peak is a starting point from the larger-model sweep
(Figure 15), not a tuned optimum for this architecture. The baseline CLI
multiplier remains .25, so `--matrix-lr .032` gives an effective peak of .008.
The default loss target of 3.5
nats/token is a hypothesis for a long run, not calibrated by short smoke tests.
An unreachable target leaves noise at zero. Forty epochs need not fit the
original one-hour compute cap.

## Normalization and feedback

The normalization/scale recipe follows [Algorithm 2, Sections 4.1.1–3 and
Appendix B.1 of the paper](https://arxiv.org/pdf/2606.25971). The model sees a fused
weight `W = diag(softplus(row_scale)) @ direction @ diag(softplus(col_scale))`.
Both gains start at one. Matrix directions have fixed Frobenius radius
`sqrt(max(rows, columns))`; embedding and LM-head directions have unit row norms.
These constraints apply to the direction, not the gain-scaled effective weight.

Initialization draws Gaussian directions with standard deviation
`1/sqrt(min(rows, columns))`, then projects to the target sphere. This replaces
the baseline's zero projections/gates: finite positive gains cannot represent a
zero matrix on a nonzero sphere. The baseline's post-embedding RMSNorm already
provides unit input RMS, so no extra embedding multiplier is added.

Global fused-weight/scalar gradients are clipped to norm 1.0 **before** the
optimizer splits them into gain and direction gradients. On multiple ranks the
fused gradients are averaged before clipping; the original optimizer collectives
are retained (so this simple implementation does redundant communication).
AdamW updates raw gains and embedding directions, Muon updates other directions.
All weight decay is zero, including gains and original scalars; the CLI accepts
only `--weight-decay 0`. There is no LR warmup. Muon's fixed update multiplier is
`sqrt(max(rows/columns, columns/rows))`, matching the sphere's scale; the baseline's
variance correction is removed. No tangent projection is applied to gradients.

After the direction and gain updates, directions are projected, matrix-only
noise is injected, and the fused weights are rebuilt. Only the noise is tangent:

```text
z ~ N(0, I)
delta = s * r / sqrt(d - 1) * (z - u * <u,z> / r²)
E[||delta||²] / r² = s²
```

Here `r` is the matrix radius and `d` its number of entries. The scalar `s` is a
relative tangent RMS, not a per-entry standard deviation. At current LR
`eta`, its formal direction temperature is `s² r² / (2 eta (d - 1))`.
The loss controller still sets noise independently of LR; annealing therefore
does not hold this formal temperature constant. Noise and checkpoint ensembling
are additions to the paper’s recipe, and need separate long-run evaluation.

The controller uses the baseline's bias-corrected loss EMA (beta .9), now averaging
**all** accumulated microbatches and ranks. After `--noise-warmup 100`, every
`--noise-interval 10` steps below the target activates noise at `1e-4` or multiplies
it by `--noise-growth 1.02`, capped at `--noise-max .5`. Above target + .01 it
divides noise by the growth factor; inside that band it holds. The logged noise
is for the next update. Set `--noise-max 0` for an ablation. Validation does not
control noise. EMA lag and limited optimization capacity prevent an exact loss
guarantee.

## Checkpoint ensemble

Ten snapshots are saved from `ceil(total_steps/2)` through `total_steps`, with
nearest-integer spacing. `--snapshots 0` disables collection. The horizon must
have enough steps for distinct samples; early stopping is disabled when sampling.
Each file contains model weights, model configuration and step. There is no
optimizer/resume checkpoint. Ten full-size snapshots occupy approximately 56 GB.

After training, validation reports the equal probability-mixture NLL using
`logaddexp` of target-token log probabilities, not averaged logits or weights.
Evaluation keeps one model on the GPU and caches target log probabilities on CPU.
The result JSON includes ensemble loss and snapshot steps. The model left in
memory is the final checkpoint.

## Interpretation and evidence

For a density `q(u)` relative to spherical surface measure and a uniform prior,
`KL(q || uniform) = log(area) - H(q)`. There is no positional preference requiring
matrix-direction weight decay. This statement concerns direction distributions,
not arbitrary changes in `q`, deterministic point masses (singular KL), or priors
over the separately learned gains. Gains and embeddings are point estimates in
this experiment; no full variational objective is computed.

[SGLD](https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf) motivates
gradient/noise scaling. [Preconditioned SGLD](https://arxiv.org/abs/1512.07666)
motivates matching diffusion to mobility. Under a frozen positive diagonal
preconditioner `P` and Euclidean gradient drift `-eta P grad L`, coordinatewise
noise covariance `2 eta T P` would give a common temperature. Isotropic noise
instead gives formal local temperatures proportional to `1/P_ii`; injecting
noise before preconditioning would give a different covariance again. State
dependence requires further drift corrections. These are local interpretations,
not sampling guarantees for AdamW.

[Muon's spectral interpretation](https://arxiv.org/abs/2506.15054) motivates a
similar local singular-direction picture, but momentum, approximate polar
iterations, spherical geometry and adaptive noise here
do not establish a stationary Bayesian posterior. Mean rather than summed token
loss also changes the temperature convention by dataset size. Treat this as
loss-constrained regularization with approximate sampling motivation.
[Snapshot ensembles](https://arxiv.org/abs/1704.00109) motivate collecting multiple
predictors; uniform times alone do not ensure independent or diverse samples.

See [the full-model smoke report](../../agent_reports/2026-09-26--loss-budget/README.md)
for observed behavior and its limits. Numerical checks run with
`python -m pytest -q tests/test_loss_budget.py`.

The [simplified implementation check](../../agent_reports/2026-09-26--minimal-baseline-diff/README.md)
exercises the restored baseline optimizer and loop at full model size.

The latest [paper MD check](../../agent_reports/2026-09-26--paper-md/README.md)
validates the fused positive-gain update and clipping at full model size.
