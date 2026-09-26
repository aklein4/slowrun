"""Show the implementation delta against the exact, checked-in gated record."""

import argparse
import difflib
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--section", choices=["all", "model", "muon", "trainer"], default="all"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / "research/loss_budget"
    baseline = (root.parents[1] / "baselines/gated_attention.py").read_text()
    model = (
        (root / "train.py")
        .read_text()
        .split("# BEGIN BASELINE MODEL\n")[1]
        .split("# END BASELINE MODEL")[0]
    )
    original_model = baseline[
        baseline.index("@dataclass\nclass GPTConfig:") : baseline.index(
            "# =============================================================================\n# Optimizer:"
        )
    ]
    original_muon = baseline[
        baseline.index(
            "@torch.compile(dynamic=False, fullgraph=True)\ndef muon_step_fused"
        ) : baseline.index("class DistMuonAdamW")
    ]
    muon = (root / "train.py").read_text()
    muon = muon[muon.index("def muon_step_fused") : muon.index("# END BASELINE MUON")]
    original_training = baseline[baseline.index("# Compute init") :]
    training = (root / "train.py").read_text()
    sections = {
        "model": (original_model, model, "train.py (model)"),
        "muon": (original_muon, muon, "train.py (muon)"),
        "trainer": (original_training, training, "train.py"),
    }
    for name, (before, after, filename) in sections.items():
        if args.section in ("all", name):
            print(
                "".join(
                    difflib.unified_diff(
                        before.splitlines(True),
                        after.splitlines(True),
                        fromfile=f"52e7441/train.py ({name})",
                        tofile=filename,
                    )
                ),
                end="",
            )


if __name__ == "__main__":
    main()
