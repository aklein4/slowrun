#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat >&2 <<'EOF'
Usage: agent_scripts/append_progress.sh [--user-message <summary>] <conversation-id> <update>

Appends a timestamped Markdown update to the conversation's _progress.md,
creating a reverse-time-prefixed directory and the progress file when needed.
When supplied, the user message summary is added as a Markdown header before
the update.
EOF
}

user_message=
user_message_set=false

if [[ ${1-} == --user-message ]]; then
    if (( $# < 2 )); then
        usage
        exit 2
    fi
    user_message=$2
    user_message_set=true
    shift 2
fi

if [[ ${1-} == -- ]]; then
    shift
fi

if (( $# != 2 )); then
    usage
    exit 2
fi

conversation_id=$1
update=$2

if [[ ! $conversation_id =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    printf 'Invalid conversation ID: %s\n' "$conversation_id" >&2
    exit 2
fi

if [[ -z $update ]]; then
    printf 'Progress update must not be empty.\n' >&2
    exit 2
fi

if [[ $user_message_set == true && -z $user_message ]]; then
    printf 'User message summary must not be empty.\n' >&2
    exit 2
fi

if [[ $user_message_set == true ]]; then
    user_message=${user_message//$'\r'/ }
    user_message=${user_message//$'\n'/ }
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
source "$script_dir/log_names.sh"
log_dir=$(conversation_log_dir "$repo_root" "$conversation_id") || exit 2
progress_file="$log_dir/_progress.md"
timestamp=$(TZ=UTC date '+%Y-%m-%d %H:%M:%S %Z')

mkdir -p -- "$log_dir"

exec 9>>"$progress_file"
flock 9

if [[ ! -s $progress_file ]]; then
    printf '# Progress\n' >&9
fi

if [[ $user_message_set == true ]]; then
    printf '\n## User message: %s\n' "$user_message" >&9
fi

printf '\n### %s\n\n%s\n' "$timestamp" "$update" >&9
