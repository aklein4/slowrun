# Loss-budget training: full-model smoke evidence

The final no-dropout implementation completed 64 updates over eight repeats of
eight FineWeb sequences using the full 1,399,476,473-parameter model and 2048-token
context. All ten planned snapshots were saved and combined as a probability
mixture. Peak allocated GPU memory was **26.136 GiB**. This establishes execution,
normalization and checkpoint plumbing; it does **not** establish constant-loss
tracking, posterior correctness, or generalization on the benchmark.

## Method and reproduction

See the [plan and explicit revisions](plan.md), [runnable reproduction](reproduce.sh),
[aggregate results](results.json), [recorded trajectories](trajectories.json),
and [final verification](verification.json).
Run `bash agent_reports/2026-09-26--loss-budget/reproduce.sh` from the repository
root with CUDA PyTorch, FlashAttention 2, PyArrow, tiktoken and huggingface_hub.
The helper fetches pinned FineWeb content; raw inputs and checkpoints are local
prerequisites, not checked in. Exact commands also live in ignored agent logs.

- Hardware: one NVIDIA GH200, about 96 GiB GPU memory; Python 3.10.12,
  Torch 2.7.0/CUDA 12.8, flash-attn 2.7.4.post1. No tracker/API uploads.
- Base checkout: `c6f0fa1b415994e0c5cf163990fe253fe1622883`, dirty implementation.
  The starting record is `52e7441f862c3295c0f5695933438dac78f7fc5b`; its preserved
  source SHA256 is `ed92a47e4cbf44b831cf9b5669f0e6e19a4f099802882a47c993db2ee4c95a35`.
- FineWeb `sample-10BT`, revision `9bb295ddab0e05d785b879661af7260fed5140fc`.
  Validation takes 16,392 tokens from 39 documents, training starts at the next
  document and takes 524,544 tokens from 753 documents. The repeated smoke slice
  uses the first eight complete 2049-token training rows; the evaluation uses the
  first two validation rows (4096 predicted tokens). There is no split filling
  or validation wraparound. This slice differs from the leaderboard validation.
- Seed 42; one sequence per optimizer step. Final run uses matrix LR .02, gain
  LR .01, embedding LR .002, output embedding LR .02, and the original scalar
  LR .125 (residual/skip multiplier .01 and x0 beta1 .96). Two-step LR warmup.
- Smoke controller: target 5, EMA beta .9, warmup 4, interval 1, seed noise .01,
  growth 1.2, maximum .5. These deliberately differ from the conservative long-run
  feedback timescale. No training-epoch/validation result was fed to the controller.

`results.json` preserves each run's config, hashes, environment and results;
`analyze.py` recreates it from local run directories. Source hashes added to the
later runs identify their executed files. The eight-epoch no-dropout validation
preceded the requested single-file consolidation; its model/update algebra was
then copied into the single file without algorithm changes. Separate CPU tests,
a full-size standalone launch and saved-ensemble re-evaluation check that assembly.
Earlier prototypes are explicitly labeled superseded, and are not claimed to be
reproducible from the final script with their old argument lists.

## Observations

For the final eight-epoch no-dropout run:

| Measurement | Observed |
| --- | ---: |
| Initial validation CE | 10.826016 |
| Final validation CE | 8.640197 |
| Ten-member mixture CE | 8.177210 |
| Best individual member CE | 8.165007 |
| Final training-loss EMA | 4.427053 |
| First nonzero noise step | 55 |
| Final relative noise RMS | .0515978 |
| Maximum matrix relative norm error | 1.20e-7 |
| Maximum embedding row norm error | 1.79e-7 |
| Peak allocated GPU memory | 26.136 GiB |

Snapshot steps were `[32, 36, 39, 43, 46, 50, 53, 57, 60, 64]`. The mixture improves
over the final checkpoint but does not beat the best earlier member. These are
single-seed descriptive numbers; repeated epochs and correlated snapshots are
not independent replicates. Training plus checkpoint/validation work before the
ensemble took about 75 seconds, excluding initial model setup. The reported
per-step timings exclude checkpoint I/O and validation.

Early 12-step prototypes exposed an output-embedding LR mismatch after row
normalization: a .001 direction LR under-updated an effective weight initialized
at .001 per coordinate. Using .02 for its unit-row direction, .002 for input
embeddings and .003 gains lowered validation CE from 10.691 to 9.266; .01 gains
gave 8.514. The first change alters multiple settings and is not a causal ablation.
The gain comparison used the same prototype/data/seed. The final implementation
then restored the original initialization and Muon kernel, restored original
scalar groups, and removed dropout in response to user instructions; its final
full-size check establishes stability of the resulting choices, not optimality.

The initial .03 noise ceiling saturated while loss continued to fall. A bounded
128-step prototype confirmed finite training and sphere constraints up to .5,
which became the configurable ceiling. Even that run did not hold its target 5:
EMA finished at 3.273. The final run's shorter trajectory increased noise but
never needed a decrease; CPU tests independently exercise both feedback branches.
The ceiling, EMA lag and unperturbed learned embeddings/gains limit what the
controller can guarantee. The default long-run target 3.5 remains uncalibrated.

## Failures and checks

Development failures are retained in local logs. A CUDA device without an index
and a gain multiplication that promoted FlashAttention inputs to fp32 failed
before training; both were fixed. Three data-preparation attempts wrote correct,
matching hashes but exited with an Arrow worker shutdown error. The final helper
uses synchronous Parquet column reads and exited successfully with identical
token-file hashes.

The separate ensemble CLI check caught a missing parameterization hook when
constructing a fresh evaluation model. Initialization now installs its gains
before loading samples; the resume/inference regression test exercises this path.

The optional full-model `--compile` attempt failed before its first update:
Torch 2.7's Inductor could not import `triton_key` from installed Triton 3.7.0.
The working default is eager mode. No claim is made that this environment's
compiled backend is validated, and no global CUDA dependency was downgraded.

Nine CPU tests cover optimizer ownership, absence of dropout, preservation of
baseline effective weights/outputs, Gaussian tangent size, sphere normalization,
both controller branches, snapshot spacing, probability mixtures, both token
formats, correct accumulation/final partial batches, and identical checkpoint
continuation with noise. Baseline comparison checks show only the intended model
changes; the Muon kernel differs solely in where compilation is selected.
Documentation links and whitespace are checked separately.

No 100M-token multi-epoch benchmark, seed sweep, stationary sampling analysis,
calibration claim or limited-track record is established by these smoke tests.
