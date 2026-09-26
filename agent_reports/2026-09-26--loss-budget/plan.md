# Full-model smoke study plan

Question: does the pinned 30-layer, 1792-wide gated model train with normalized
matrix/embedding directions and learned gains, does the noise controller operate,
and can ten snapshots be saved and combined on one GH200?

Use seed 42, 2048-token sequences, and a bounded FineWeb sample-10BT slice at
revision 9bb295ddab0e05d785b879661af7260fed5140fc. Validation consumes the first
16,392 tokens; training starts at the next document and consumes 524,544 tokens.
No claims about population uncertainty or final generalization: the experimental
unit is one short run/seed. Compare early training behavior of matrix LR .02,
gain LR .001 against at most two local changes. Use held-out loss descriptively,
not as a controller input. Exercise noise with an explicitly elevated smoke-only
loss target, and test checkpoint mixtures on a finite held-out subset.

Bounds: one GPU at a time, each launch <=10 minutes, at most 100 optimizer steps
across the initial comparisons, model dimensions unchanged, >=1 full 2048-token
sequence per step, <96 GiB GPU memory and <150 GB new disk outputs. Small CPU
fixtures separately verify exact resume, routing, tangent variance and mixtures.
Record commands, failures, packages, input hashes, memory, time, losses and sphere
errors. Adjust this plan explicitly if evidence warrants more work.

## Evidence-driven extension

The first 100 full-model optimizer steps completed, including 64 updates over
eight repeats of eight training sequences with all ten snapshots. A .03 angular
noise ceiling saturated at step 27 while EMA fell well below the smoke target 9.
Extend by one 128-step run (16 epochs on the same eight sequences), no snapshots,
target 5, beta .9, noise seed .01, growth 1.2, ceiling .5. Question: can the
feedback increase AND decrease noise around an attainable loss budget without
breaking normalization? Keep the same per-launch 10-minute and total 150-GB disk
limits. This extension is about feedback behavior, not generalization tuning.

## Preserve the original implementation

The user explicitly requested that similarity to the original be a core repo
principle. The implementation now retains the model definitions verbatim except
for a post-initialization parameterization hook and optimizer factory; nonzero
weights preserve the original initialization draws. The Muon numerical kernel is
copied unchanged (compilation becomes optional), replacing the prototype's
explicit update-norm rescaling. Keep prototype observations labeled separately.
Validate the final revision with 64 full-size steps/eight repeated epochs and ten
snapshots using ceiling .5, then a two-step compile check if needed. Extend total
bounds to 300 optimizer updates and 250 GB disk to preserve earlier evidence.

## Remove dropout

The user then requested dropout removal. Both residual dropout modules are now
identities, retaining their original names; the dropout CLI/config field is gone.
Restore original residual/skip/x0 scalar optimizer hyperparameters as well.
Revalidate the final revision with another 64-step/ten-snapshot run. Keep all
preceding runs as explicitly superseded observations. Bounds become 400 total
optimizer updates and 300 GB disk, one GPU and <=10 minutes per launch unchanged.

## Single-file consolidation and final verification

The user's final structural constraint moved every training dependency into one
self-contained train.py. Preparation and comparison tools moved to agent_scripts.
A two-step full-model eager launch with active Gaussian noise passed; the optional
compiled attempt failed before updates due to Torch 2.7.0/Triton 3.7.0 mismatch.
Standalone evaluation of all ten no-dropout snapshots exactly matched the saved
mixture result. Final checks and hashes are in verification.json. No further
training beyond the bounded smoke runs was launched.
