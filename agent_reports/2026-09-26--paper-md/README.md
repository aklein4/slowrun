# Paper MD alignment

This follow-up matches the normalization and optimizer strategy in
[arXiv:2606.25971v2, Algorithm 2 and Appendix B.1](https://arxiv.org/pdf/2606.25971).
The question was whether fused softplus gains, fixed direction spheres,
clipping before gradient splitting, and the revised optimizer settings execute
at the full gated-baseline size. The noise controller and constant LR exploration
remain experiment-specific; this is not an architectural reproduction of the paper.

One GH200 ran the 1,399,476,473-parameter model for 20 updates over five epochs,
using the same four training sequences and one separate validation sequence as
the [previous check](../2026-09-26--minimal-baseline-diff/README.md). Inputs were
unchanged. Model seed 42, noise seed 43; PyTorch 2.7.0/CUDA 12.8,
FlashAttention 2.7.4.post1, eager mode, one distributed rank, no W&B upload.
The recorded pre-run bound was two minutes, 60GB snapshot storage and 96GB GPU.

Initial validation CE was 11.317924; final CE 10.364526; best CE 9.652829;
ten-checkpoint ensemble CE 9.822068. Noise activated at step 5 and reached
0.001540702 for the next update. Final training EMA was 2.985847. Peak memory
was 37,711.78 MiB; wall time 53.75s. [results.json](results.json) stores outcomes.
The repeated tiny training slice visibly overfits: these are execution checks,
not evidence of better generalization or calibrated hyperparameters.

Four CPU tests cover sphere constraints, all parameter groups, zero decay,
positive gains, embedding/gain learning rates, clipping before splitting,
chain-rule agreement with independent autograd, matrix-only noise, feedback,
snapshot timing and probability mixtures. A four-update full-size check after
removing unused variance-state allocations verifies the final optimizer interface.
Multi-GPU and compilation remain untested. Multi-GPU clipping averages fused
gradients before the retained optimizer reductions, at a redundant communication cost.

Commands and output are logged in the `paper-md` task under ignored `agent_logs/`.
[reproduce.sh](reproduce.sh) contains the 20-update command. Inputs and snapshots
are in `local_data/minimal-diff-check/` and `local_data/paper-md-check/`.
The paper defaults replace exact-zero gate/projection initializations; previous
checkpoints using signed explicit gains are incompatible with this parameterization.
