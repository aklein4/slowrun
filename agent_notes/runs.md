# Bounded runs

Before a download or GPU experiment inspect the entry point and establish
token/step, time, memory and disk bounds within the user's authorization. Use
fresh ignored `local_data/` outputs and [logging](../agent_scripts/README.md).
Keep one GPU process at a time. Pin inputs and record environment and commands.
No service publication or tracking upload is implied by a local training test.

After interruption inspect process/output state before rerunning. Preserve
failed attempts and distinguish synthetic plumbing checks, real-data smoke
tests, and generalization evidence. Stop only owned processes after use.
