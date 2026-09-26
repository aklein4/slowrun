# Baselines

Unmodified single-file snapshots from `qlabs-eng/slowrun`. Preserve source and
formatting byte for byte so experiment diffs show only intentional changes.

| File | Role | Pinned upstream source |
| --- | --- | --- |
| [gated_attention.py](gated_attention.py) | Our starting point: “Add gating per attention head” | [52e7441](https://github.com/qlabs-eng/slowrun/blob/52e7441f862c3295c0f5695933438dac78f7fc5b/train.py) |
| [one_hour_record.py](one_hour_record.py) | Current one-hour record: first-order meta-gradient, 3.183 validation loss in 59.3 minutes | [45bf3fd](https://github.com/qlabs-eng/slowrun/blob/45bf3fdf229c2079f276142073838b924a333ba1/train.py) |

The record was checked against the [upstream leaderboard](https://github.com/qlabs-eng/slowrun#limited-compute-track-1-hour)
on 2026-09-26. This is a pinned snapshot, not an automatically updated copy.
Both scripts retain their original dependencies and launch requirements.

Compare [our experiment](../research/loss_budget/train.py) with the gated base:

```bash
diff -u baselines/gated_attention.py research/loss_budget/train.py
python agent_scripts/compare_loss_budget_baseline.py --section model
python agent_scripts/compare_loss_budget_baseline.py --section muon
python agent_scripts/compare_loss_budget_baseline.py --section trainer
```

These snapshots retain the repository's MIT license and original attribution.
