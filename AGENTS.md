# Agent instructions

Start with [agent_notes/README.md](agent_notes/README.md), its core notes, and
only the task routes you need. Preserve existing edits and historical records.
Keep implementations as close to their chosen original as possible: preserve
structure, names, initialization and update algebra unless the requested change
requires otherwise. Make the behavioral delta directly reviewable against the
pinned source; avoid incidental refactors and formatting churn.
Keep each submission's training implementation in one self-contained, hackable
file, following the original submissions. Put development tools and experiment
evidence outside that file, not pieces required to run the trainer.
Shared behavior and usage belong in [docs/](docs/README.md); evidence belongs in
[agent_reports/](agent_reports/README.md), temporary recovery state in ignored
`agent_logs/`. Follow the [maintenance procedure](agent_notes/documentation.md)
when changing these instructions.
