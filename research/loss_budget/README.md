# Loss-budget Gaussian noise

Train the gated limited-compute model with Muon matrix directions, AdamW gains
and embeddings, and Gaussian noise controlled by training-loss EMA. This starts
from [“Add gating per attention head”, `52e7441`](https://github.com/qlabs-eng/slowrun/blob/52e7441f862c3295c0f5695933438dac78f7fc5b/train.py).
Its 30 layers, width 1792, 14 heads, 2048-token context, attention gates, SwiGLU,
value projections, U-Net skips and logit soft cap are retained. Dropout is removed;
the original residual wrapper names remain as identities to keep the diff small. There
are 1,399,476,473 parameters including the new gains. The root leaderboard
script remains a separate experiment.

The complete training implementation is the single, self-contained [train.py](train.py).
It imports no local modules. Preparation and comparison scripts are development
tools and are not needed to execute the submission.

The [exact original script](../../baselines/gated_attention.py) is checked in unchanged. Model
source retains the original layout, with only dropout removal, a parameterization
hook at the end of initialization, and optimizer routing changed. Nonzero initial
weights preserve the original draws; zero matrices get arbitrary directions
multiplied by zero gains. The Muon numerical kernel is copied unchanged, with
compilation selected by the caller. Review the focused diffs with:

```bash
python agent_scripts/compare_loss_budget_baseline.py --section model
python agent_scripts/compare_loss_budget_baseline.py --section muon
python agent_scripts/compare_loss_budget_baseline.py --section trainer
```

The last command includes all training-orchestration changes. Preserving original
structure and avoiding incidental changes is a [repository rule](../../AGENTS.md).

## Run

Use a CUDA PyTorch environment with FlashAttention 2 installed; tests used
PyTorch 2.7.0, CUDA 12.8 and flash-attn 2.7.4.post1 on one GH200. CUDA training
uses bf16 autocast with fp32 master parameters and optimizer state. CPU execution
uses SDPA for numerical tests. The data helper also needs `pyarrow`,
`huggingface_hub` and `tiktoken` (included by the root data dependencies).
Eager mode is the tested default. `--compile` requires compatible Torch/Triton
versions; the local compiler check failed on an existing version mismatch.

From the repository root, using data from `prepare_data.py` or the original
record's packed `.pt` files:

```bash
python research/loss_budget/train.py \
  --output local_data/runs/loss-budget \
  --train-data fineweb_data/fineweb_train.pt \
  --val-data fineweb_data/fineweb_val.pt \
  --num-epochs 40 --loss-target 3.5
```

`--help` lists controls. The default effective batch is 524,288 tokens, accumulated
on one GPU. Matrix/gain/embedding/output-embedding LRs are .02/.01/.002/.02.
LR warms up for 100 updates, then stays constant for exploration. The default
loss target 3.5 nats/token is a starting hypothesis for a long run, **not calibrated
by the short smoke tests**. Choose a training-loss budget the model can reach;
an unreachable budget leaves noise at zero. This is not an assertion that the
40-epoch run meets the original one-hour compute cap.

Bounded full-model exercise, using a small pinned FineWeb slice:

```bash
python agent_scripts/prepare_loss_budget_smoke.py \
  --output local_data/smoke-data \
  --revision 9bb295ddab0e05d785b879661af7260fed5140fc
python research/loss_budget/train.py \
  --output local_data/runs/smoke \
  --train-data local_data/smoke-data/train.pt \
  --val-data local_data/smoke-data/val.pt \
  --steps 32 --total-batch-size 2048 --eval-batches 2 \
  --lr-warmup-steps 2 --loss-target 10 --loss-ema-beta .9 \
  --noise-warmup-steps 4 --noise-interval 1 --noise-growth 1.2
```

These smoke-only settings exercise feedback early; they do not tune a mature
language model's temperature. Raw token files are split into consecutive
`sequence_len + 1` rows, dropping the tail, like the gated record's preprocessing.
Legacy packed files retain their valid rows. Training reshuffles rows each epoch;
validation traverses its finite rows once without wrapping. Incomplete device
batches are dropped and counted in `run.json`. Separate paths/hashes guard against
byte-identical input files; callers remain responsible for document-level separation in
custom datasets. The preparation helper separates splits at document boundaries.

## Update and feedback

Every effective weight is `diag(row_scale) @ direction @ diag(col_scale)`.
Matrix directions live on the Frobenius sphere of radius `sqrt(output_width)`;
token/output embedding directions have unit row norms. Scales are signed, so
zero row gains exactly preserve initially zero residual projections and gates
while the direction remains nonzero. The forward pass applies gains without
renormalizing weights. Muon updates all other linear directions, including gate
and value-projection matrices; AdamW updates all gains, embedding directions and
residual/skip scalars. The original scalar LR .125 is retained (residual/skip
use .00125; x0 uses .125 with beta1 .96). Only Muon parameters receive noise. All direction weight
decay is zero; optional decay applies only to gains.

The matrix update keeps the record's momentum, Polar Express and variance
reduction and aspect-ratio LR adjustment unchanged, then retracts to the sphere,
adds a tangent Gaussian, and retracts again. Embeddings are normalized
after AdamW. The direction/magnitude split is inspired by
[the supplied paper](https://arxiv.org/abs/2606.25971); this implementation uses
explicit autograd gains, signed zero-capable scales and fixed radii rather than
its fused positive-gain implementation.

For direction `u` with `d` entries, radius `r`, reference LR `eta0` and current
LR `eta`, the noise before final retraction is

```text
z ~ N(0, I)
P_u z = z - u * <u,z> / r²
delta = s * sqrt(eta/eta0) * r/sqrt(d-1) * P_u z
E[||delta||²] / r² = s² * eta/eta0
```

Thus `s` is a dimension-independent relative tangent RMS at the base LR, not a
per-entry standard deviation. The Gaussian is isotropic within the tangent
space; the retracted weights themselves are not Gaussian. Its coordinate
diffusion coefficient corresponds formally to
`T_direction = s² r² / (2 eta0 (d-1))` in `sqrt(2 eta T)` units.

The controller bias-corrects an EMA of mean training CE across **all** accumulated
microbatches. After 100 warmup steps, every 10 updates below the target activates
noise at `1e-4` or multiplies it by `1.02`, capped at `.5`. Above target + `.01`
it divides noise by `1.02`; within that tolerance it holds. Noise is zero before
activation. The current batch's loss sets the next parameter perturbation.
Validation is never a controller input. `--no-noise` supplies a matched ablation.
Feedback cannot enforce an exact loss constraint: EMA lag, stochastic loss,
limited optimization capacity and the noise cap can all prevent tracking.

## Checkpoints and prediction

The default saves exactly ten distinct snapshots between `ceil(total_steps/2)`
and `total_steps`, including both endpoints, with nearest-integer spacing.
Too-short horizons fail explicitly; `--snapshots 0` disables this for ablations.
No early stopping silently truncates the requested ensemble. Predictive averaging
uses `logsumexp(log_softmax(logits)) - log(K)`, an equal categorical mixture.
`ensemble_log_probs` in [train.py](train.py) provides full distributions;
validation caches only target-token log probabilities on CPU and keeps one model
on the GPU. It reports the ensemble NLL and individual member NLLs.

`snapshots.json` records planned/saved steps and completion. `metrics.jsonl` records
the controller trajectory; `run.json` records settings, dataset hashes and
environment. `latest.pt` contains full model/optimizer/controller/RNG state and
the exact data cursor. Checkpoints are written by atomic replacement. At full
size, budget approximately 56 GB for ten fp32 snapshots plus 12 GB for `latest.pt`,
and transient space for atomic replacement. Repeated snapshot loading favors
bounded memory over inference latency.

```bash
# Pause at step 200 while preserving the original planned training horizon:
python research/loss_budget/train.py --output local_data/runs/paused --stop-after 200
# Resume with the same training/data options and original output directory:
python research/loss_budget/train.py --output local_data/runs/paused \
  --resume local_data/runs/paused/latest.pt
# Re-evaluate saved predictive samples without training:
python research/loss_budget/train.py --output local_data/runs/loss-budget --eval-only
```

Use `--save-every N` for recovery during a long run; final/pause saves are automatic.
`--steps` changes the planned horizon, whereas `--stop-after` only pauses it.
Resume rejects changed optimization/data/horizon settings and inconsistent
snapshot histories. CPU tests verify identical continuation with matrix noise; GPU kernels may have numerical nondeterminism.

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
iterations, variance normalization, spherical geometry and adaptive noise here
do not establish a stationary Bayesian posterior. Mean rather than summed token
loss also changes the temperature convention by dataset size. Treat this as
loss-constrained regularization with approximate sampling motivation.
[Snapshot ensembles](https://arxiv.org/abs/1704.00109) motivate collecting multiple
predictors; uniform times alone do not ensure independent or diverse samples.

See [the full-model smoke report](../../agent_reports/2026-09-26--loss-budget/README.md)
for observed behavior and its limits. Numerical checks run with
`python -m pytest -q tests/test_loss_budget.py`.
