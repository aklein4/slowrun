#!/usr/bin/env bash
set -euo pipefail
# Run from repo root using inputs from the linked previous check.
mkdir -p local_data/paper-md-check
WANDB_MODE=disabled torchrun --standalone --nproc_per_node=1 research/loss_budget/train.py \
  --no-compile --input_bin local_data/minimal-diff-check/train.pt \
  --input_val_bin local_data/minimal-diff-check/val.pt --num-epochs 5 \
  --device-batch-size 1 --total-batch-size 2048 --noise-warmup 2 \
  --noise-interval 1 --noise-growth 1.2 --loss-target 10 \
  --checkpoint-dir local_data/paper-md-check/checkpoints \
  --save-result local_data/paper-md-check/result.json
