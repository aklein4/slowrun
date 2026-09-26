# Validation

Use [logging](../agent_scripts/README.md). Format new helper code with isort then
Black. Preserve formatting in copied baseline code so diffs expose behavior
changes, as explicitly requested by the user. Run `python -m pytest -q tests`, `python agent_scripts/check_docs.py`, and
`git diff --check` for this trainer. Focus numerical tests on parameter routing,
sphere constraints, loss control, checkpoint timing and probability
mixtures. CPU tests do not establish GPU stability or throughput.

Read [runs](runs.md) before full-size tests. Record actual checks and failures;
short runs establish execution and local behavior, not final generalization.
