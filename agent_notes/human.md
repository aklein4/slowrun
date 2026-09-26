# Human intent

Explicit request, 2026-09-26 (human-owned; change only on explicit request):

- Start from limited-compute “Add gating per attention head”, commit `52e7441`.
- Increase Gaussian matrix noise when training-loss EMA is below a threshold,
  over many epochs. Use Muon for matrices and AdamW for embeddings and scales.
- Normalize directions after optimizer updates; learn magnitudes separately.
  Embeddings also need normalization and learned row/column scales.
- Ensemble predictions from ten checkpoints evenly spaced in the second half.
- Adapt the agent filesystem/protocols from `~/meta-data`, consulting OpenAI and
  Anthropic guidance. Use the one available GPU for short full-model tests.

The present conversation takes precedence over recorded intent.

Explicit follow-up, 2026-09-26: keep implementations as similar to the original
as possible so a diff clearly shows the changes. This is a core development
principle throughout this repository.

Explicit follow-up, 2026-09-26: remove dropout from this implementation.

Explicit follow-up, 2026-09-26: keep the implementation in a single hackable file,
like the other submissions, as part of minimizing the diff.

Explicit follow-up, 2026-09-26: simplify the submission generally; avoid excessive
robustness and edge-case handling. Preserve the baseline top-level script and
make only the changes needed for this experiment.

Explicit follow-up, 2026-09-26: match the normalization/scale strategy, embeddings,
weight decay and gradient clipping to the normalized parameter paper
(arXiv:2606.25971), while retaining the simple single-file implementation.

Explicit follow-up, 2026-09-26: adjust hyperparameters based on the paper’s findings.
