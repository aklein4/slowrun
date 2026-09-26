# Minimal baseline diff: execution check

Question: does restoring the original gated record's optimizer and top-level
training loop retain the requested normalization, feedback and ten snapshots?
This follows the earlier [smoke study](../2026-09-26--loss-budget/README.md).
The user requested fewer abstractions and less edge-case machinery; resume,
custom optimizer wrappers and alternate execution modes have been removed.

A single GH200 ran the full 1,399,476,473-parameter model, 30 layers, width 1792,
14 heads and 2048-token context. Environment: PyTorch 2.7.0, CUDA 12.8,
FlashAttention 2.7.4.post1, eager mode. Compilation was bypassed for the previously
reported Torch/Triton mismatch. Distributed world size was one, W&B disabled.

The experimental unit was one run (model seed 42, noise seed 43), using the first
four 2049-token rows of the earlier pinned FineWeb training slice and first row
of its separate validation slice. Five epochs gave 20 updates at 2048 tokens per
update. Noise warmup was 2 steps, interval 1, growth 1.2, target 10. These settings
exercise feedback; they are not mature-model hyperparameter recommendations.
Inputs/checkpoints are in ignored `local_data/minimal-diff-check/`; command/output
logs are under the `minimal-baseline-diff` task in `agent_logs/`.

Observed: initial validation CE 10.825354; final CE 9.184397; probability ensemble
CE 8.522369. Noise activated at step 10 and reached 0.000619174 for the next update.
All ten scheduled snapshots were saved and evaluated. Peak allocated GPU memory
was 38,725.90 MiB; wall time was 54.31 seconds. Machine-readable outcomes are in
[results.json](results.json); [reproduce.sh](reproduce.sh) records the command.

Three CPU numerical tests check effective initial weights, parameter routing,
sphere norms, matrix-only noise, feedback, sampling times and probability mixing.
This is an execution check with tiny repeated data and one seed. It does not
establish generalization, long-run stability, multi-GPU or compiled correctness.
The original optimizer kernels and distributed optimizer class are retained.
