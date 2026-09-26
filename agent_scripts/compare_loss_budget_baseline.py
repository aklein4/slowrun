"""Show a direct diff against the unchanged gated record."""

import argparse
import difflib
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--section", choices=["all", "model", "muon", "trainer"], default="all"
)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
before = (root / "baselines/gated_attention.py").read_text()
after = (root / "research/loss_budget/train.py").read_text()
bounds = {
    "model": ("# GPT Model", "# Optimizer:"),
    "muon": ("def muon_step_fused", "class DistMuonAdamW"),
    "trainer": ("# Compute init", None),
}
if args.section != "all":
    start, end = bounds[args.section]
    before, after = (
        s[s.index(start) : s.index(end) if end else None] for s in (before, after)
    )
print(
    "".join(
        difflib.unified_diff(
            before.splitlines(True),
            after.splitlines(True),
            fromfile="baselines/gated_attention.py",
            tofile="research/loss_budget/train.py",
        )
    ),
    end="",
)
