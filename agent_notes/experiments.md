# Evidence index

See [report procedure](../agent_reports/README.md). Add findings with their limits
only when they affect future choices.

- [Loss-budget full-model smoke study](../agent_reports/2026-09-26--loss-budget/README.md):
  full gated model, no dropout, normalized Muon directions and ten snapshots fit
  in about 26.1 GiB. Noise increases and mixtures execute; constant-loss tracking
  and benchmark generalization remain unproven. Includes preliminary LR evidence.

[Minimal baseline diff check](../agent_reports/2026-09-26--minimal-baseline-diff/README.md)
verifies the simplified original-loop implementation: 20 full-size updates,
matrix noise activation and ten-checkpoint probability evaluation on one GPU.

[Paper MD alignment](../agent_reports/2026-09-26--paper-md/README.md) checks the
fused positive-gain parameterization, paper optimizer settings and clipping at
full size. Tiny-data overfitting is observed; no generalization claim is made.

[Paper-informed LR defaults](../agent_reports/2026-09-26--paper-lr/README.md)
checks peak matrix LR .008 with full-run linear decay to 1e-8 over 20 full-size
updates; this is an execution check, not an optimal-LR determination.
