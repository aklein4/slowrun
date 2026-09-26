#!/usr/bin/env bash
# Run from the repo root. Fresh output paths are required; no upload/tracking.
set -euo pipefail
python agent_scripts/prepare_loss_budget_smoke.py \
  --output local_data/reproduce-loss-budget-data \
  --revision 9bb295ddab0e05d785b879661af7260fed5140fc
python - <<'PY'
import torch
path = 'local_data/reproduce-loss-budget-data/'
data = torch.load(path + 'train.pt', weights_only=True)
torch.save({'tokens': data['tokens'][:8 * 2049]}, path + 'train-eight-sequences.pt')
PY
timeout 600 python research/loss_budget/train.py \
  --train-data local_data/reproduce-loss-budget-data/train-eight-sequences.pt \
  --val-data local_data/reproduce-loss-budget-data/val.pt \
  --output local_data/reproduce-loss-budget-run \
  --num-epochs 8 --total-batch-size 2048 --lr-warmup-steps 2 \
  --eval-batches 2 --loss-target 5 --loss-ema-beta .9 \
  --noise-warmup-steps 4 --noise-interval 1 --noise-seed-std .01 \
  --noise-growth 1.2 --eval-every 16
