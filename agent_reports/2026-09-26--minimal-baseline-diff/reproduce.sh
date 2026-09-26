#!/usr/bin/env bash
set -euo pipefail
# Run from the repo root after preparing the pinned data in the previous report.
mkdir -p local_data/minimal-diff-check
python - <<'PY'
import torch
for split, rows in [('train', 4), ('val', 1)]:
    data = torch.load(f'local_data/loss-budget-data-sync/{split}.pt', weights_only=True)
    torch.save({'tokens': data['tokens'][:rows * 2049]}, f'local_data/minimal-diff-check/{split}.pt')
PY
WANDB_MODE=disabled torchrun --standalone --nproc_per_node=1 research/loss_budget/train.py \
  --no-compile --input_bin local_data/minimal-diff-check/train.pt \
  --input_val_bin local_data/minimal-diff-check/val.pt --num-epochs 5 \
  --device-batch-size 1 --total-batch-size 2048 --noise-warmup 2 \
  --noise-interval 1 --noise-growth 1.2 --loss-target 10 \
  --checkpoint-dir local_data/minimal-diff-check/checkpoints \
  --save-result local_data/minimal-diff-check/result.json
