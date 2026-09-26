#!/usr/bin/env bash

set -uo pipefail

usage() {
    cat >&2 <<'EOF'
Usage: agent_scripts/run_logged.sh <conversation-id> <run-name> [--] <command> [args...]

Runs a command, displays its combined stdout/stderr, and saves that output to
agent_logs/<reverse-time>--<conversation-id>/<reverse-time>--<run-name>.log.
If that conversation already has an _progress.md, appends the shortened command
and its eventual outcome there.
EOF
}

if (( $# < 3 )); then
    usage
    exit 2
fi

conversation_id=$1
run_name=$2
shift 2

if [[ ${1-} == -- ]]; then
    shift
fi

if (( $# == 0 )); then
    usage
    exit 2
fi

if [[ ! $conversation_id =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    printf 'Invalid conversation ID: %s\n' "$conversation_id" >&2
    exit 2
fi

if [[ ! $run_name =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    printf 'Invalid run name: %s\n' "$run_name" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
source "$script_dir/log_names.sh"
log_dir=$(conversation_log_dir "$repo_root" "$conversation_id") || exit 2
log_name="$(reverse_timestamp)--$run_name.log"
log_file="$log_dir/$log_name"
progress_file="$log_dir/_progress.md"
progress_helper="$script_dir/append_progress.sh"

mkdir -p -- "$log_dir"

if [[ -e $log_file ]]; then
    printf 'Refusing to overwrite existing log: %s\n' "$log_file" >&2
    exit 2
fi

command_summary=
printf -v command_summary '%q ' "$@"
command_summary=${command_summary% }
max_command_length=240
if (( ${#command_summary} > max_command_length )); then
    command_summary="${command_summary:0:max_command_length-3}..."
fi

progress_enabled=false
if [[ -f $progress_file ]]; then
    progress_enabled=true
    progress_update=$(printf 'Started command for `%s`:\n\n    %s' \
        "$log_name" "$command_summary")
    if ! "$progress_helper" "$conversation_id" "$progress_update"; then
        printf 'Warning: could not append command start to %s\n' \
            "$progress_file" >&2
    fi
fi

"$@" 2>&1 | tee -- "$log_file"
command_status=${PIPESTATUS[0]}

if [[ $progress_enabled == true ]]; then
    if (( command_status == 0 )); then
        outcome="Command for \`$log_name\` completed successfully (exit status 0)."
    elif (( command_status > 128 && command_status <= 255 )); then
        signal_number=$((command_status - 128))
        signal_name=$(kill -l "$signal_number" 2>/dev/null || printf 'unknown signal')
        outcome="Command for \`$log_name\` stopped with exit status $command_status (signal $signal_name)."
    else
        outcome="Command for \`$log_name\` exited with status $command_status."
    fi

    if ! "$progress_helper" "$conversation_id" "$outcome"; then
        printf 'Warning: could not append command outcome to %s\n' \
            "$progress_file" >&2
    fi
fi

exit "$command_status"
