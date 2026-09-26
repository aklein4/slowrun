# Workflow

Work from the root; inspect status/diffs and nested instructions first. For
multi-step work initialize one progress file using [logged commands](../agent_scripts/README.md),
record the outcome, constraints, plan and starting state. Wrap project programs,
checks and environment changes; bounded inspection needs no wrapper.

Make the change, run [appropriate checks](validation.md), inspect the final diff
and `git diff --check`. Record results and limitations in progress. Save durable
guidance only when it changes a future decision.

## Resume

Read the existing task progress and current diff before repeating work. Check
actual process/output state; a missing completion entry does not prove failure.
At milestones record: outcome/constraints, changed paths/decisions, checks and
limits, running processes/output paths, and next action. Preserve failed runs.
