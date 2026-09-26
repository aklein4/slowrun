# Agent helpers

Run from the repository root. These helpers handle agent logging and note
maintenance; human pipeline commands remain in their components.

For this fork, [compare_loss_budget_baseline.py](compare_loss_budget_baseline.py)
shows focused model/Muon/training diffs against the preserved gated record.
[prepare_loss_budget_smoke.py](prepare_loss_budget_smoke.py) prepares a bounded,
pinned FineWeb slice for the [full-model study](../agent_reports/2026-09-26--loss-budget/README.md).
Neither is imported by the single-file training submission.

## Progress and logged commands

```bash
agent_scripts/append_progress.sh --user-message 'Request summary' TASK_ID 'Outcome, constraints, plan, starting state.'
agent_scripts/run_logged.sh TASK_ID docs-links -- python3 agent_scripts/check_docs.py
agent_scripts/append_progress.sh TASK_ID 'Finding, evidence path, next action.'
```

Replace `TASK_ID` with one stable task name. IDs and run names must start with a
letter/digit and contain only letters, digits, dots, underscores or hyphens.
Use distinct run names to make outcomes easy to locate.

| Helper | Contract |
| --- | --- |
| [append_progress.sh](append_progress.sh) | Creates/reuses a task directory and appends UTC-stamped entries to `_progress.md`, using `flock` for writes. |
| [run_logged.sh](run_logged.sh) | Displays and saves combined stdout/stderr, returns the command's status, and appends start/outcome entries if progress already exists. Does not change the working directory. |
| [log_names.sh](log_names.sh) | Shared implementation for task directories and reverse-time names; not a user entry point. |

Logs live in ignored
`agent_logs/REVERSE_TIME--TASK_ID/REVERSE_TIME--RUN_NAME.log`, newest sorting
first. Existing old-style task directories are supported. A log-path collision
fails rather than overwriting; use a new run name. If multiple task directories
match, inspect them and recover the correct task context before rerunning.

Initialize progress before long commands so the wrapper records outcomes.
After interruption, inspect the actual process and output state; an incomplete
progress entry is not evidence that the process stopped.
Use the [recovery entry](../agent_notes/workflow.md#resume) at useful milestones.

The Bash helpers target Linux and need `flock`, `date`, `tee` and `tr`.
They record commands/output **without redaction**; load credentials via an env
file, keep secrets out of arguments/output, and do not use shell tracing.

## Documentation checks

```bash
python3 agent_scripts/check_docs.py
python3 agent_scripts/check_docs.py agent_notes docs
```

[check_docs.py](check_docs.py) uses only Python's standard library. Defaults
cover maintained root instructions, agent notes/helper docs, shared guides,
pipeline README entry points and the report index. Explicit files/directories
replace that selection; directories are searched recursively for `*.md`.

Checks local inline Markdown links (including images), file/directory existence,
and ATX heading anchors in Markdown targets. Ignores fenced code and external
URLs. Reports `file:line` errors and exits nonzero on failure. It does not fetch
URLs or validate reference-style links, HTML anchors, prose claims or commands.
Use inline links and ATX headings in maintained notes. Historical reports are
checked only when explicitly selected; their original paths may be historical.

After moves, also search for incoming references outside the default set.
Keep fixes scoped: old experiment records should retain original commands and
provenance. See [note maintenance](../agent_notes/documentation.md).
