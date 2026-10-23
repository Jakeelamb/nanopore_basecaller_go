#!/usr/bin/env bash
set -u
if (( $# < 2 )); then
  echo 'usage: record-command.sh NEW_RECEIPT_DIRECTORY COMMAND [ARG...]' >&2
  exit 2
fi
receipt=$1
shift
if ! mkdir "$receipt"; then
  echo 'The receipt directory must be new; existing attempts are preserved.' >&2
  exit 2
fi
receipt=$(cd "$receipt" && pwd)
pwd > "$receipt/cwd.txt"
printf '%q ' "$@" > "$receipt/command.sh"
printf '\n' >> "$receipt/command.sh"
started=$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)
printf '%s\n' "$started" > "$receipt/start.txt"
/usr/bin/time -v -o "$receipt/resources.txt" "$@" > "$receipt/stdout" 2> "$receipt/stderr"
status=$?
finished=$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)
printf '%s\n' "$finished" > "$receipt/end.txt"
printf '%s\n' "$status" > "$receipt/exit.txt"
sha256sum "$receipt/stdout" "$receipt/stderr" > "$receipt/output-sha256.txt"
printf '{"started":"%s","finished":"%s","exit":%d}\n' "$started" "$finished" "$status" > "$receipt/receipt.json"
printf '%s\n' "$receipt"
exit "$status"
