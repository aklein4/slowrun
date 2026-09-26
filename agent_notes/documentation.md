# Note maintenance

Keep one authoritative home: explicit intent in `human.md`, routing here and in
the index, shared usage in `docs/`, implementation in source/config, evidence in
dated reports, temporary recovery in ignored logs. Keep mandatory context small;
load narrow task modules on demand. Save constraints with reasons and evidence,
not routine completion histories. Repair incoming links when moving guidance.
Completed reports are immutable; add corrections in a new dated report.

For note changes, walk code, docs and recovery routes and run
`python agent_scripts/check_docs.py` through the logging wrapper. Link checks do
not validate prose or commands. Do not add permission gates beyond user scope.

Adapted from `~/meta-data` on 2026-09-26. Official guidance checked:
[OpenAI AGENTS.md](https://learn.chatgpt.com/docs/agent-configuration/agents-md)
for scoped instructions and
[Anthropic long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
for explicit progress artifacts and verification between sessions.
