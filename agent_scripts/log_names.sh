#!/usr/bin/env bash

# Encode a non-negative integer as fixed-width, lowercase base 36.
base36() {
    local value=$1
    local width=$2
    local digits=0123456789abcdefghijklmnopqrstuvwxyz
    local encoded=
    local remainder

    while (( value > 0 )); do
        remainder=$((value % 36))
        encoded="${digits:remainder:1}$encoded"
        value=$((value / 36))
    done
    printf '%0*s%s' "$((width - ${#encoded}))" '' "$encoded" | tr ' ' 0
}

# Return a compact, fixed-width, reverse-sortable timestamp in seconds.
reverse_timestamp() {
    local seconds
    seconds=$(date '+%s')
    base36 "$((9999999999 - seconds))" 7
    printf '\n'
}

# Print the existing directory for a conversation, or its newly allocated
# timestamp-prefixed path. Legacy unprefixed directories remain supported.
conversation_log_dir() {
    local repo_root=$1
    local conversation_id=$2
    local legacy_dir="$repo_root/agent_logs/$conversation_id"
    local candidate
    local -a matches=()

    if [[ -d $legacy_dir ]]; then
        printf '%s\n' "$legacy_dir"
        return
    fi

    shopt -s nullglob
    matches=("$repo_root"/agent_logs/*--"$conversation_id")
    shopt -u nullglob

    for candidate in "${matches[@]}"; do
        [[ -d $candidate ]] || continue
        if (( ${#matches[@]} > 1 )); then
            printf 'Multiple log directories found for conversation ID %s.\n' \
                "$conversation_id" >&2
            return 1
        fi
        printf '%s\n' "$candidate"
        return
    done

    printf '%s/agent_logs/%s--%s\n' \
        "$repo_root" "$(reverse_timestamp)" "$conversation_id"
}
