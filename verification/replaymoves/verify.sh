#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
if (( $# < 2 )); then
  echo "usage: $0 DORADO_SOURCE CAPTURE_MANIFEST [CAPTURE_MANIFEST ...]" >&2
  exit 2
fi
source_path=$(realpath "$1")
shift
captures=()
for capture in "$@"; do
  capture=$(realpath "$capture")
  if [[ "$capture" == *:* ]]; then echo "capture paths cannot contain a colon" >&2; exit 2; fi
  captures+=("$capture")
done
artifact_root=${REPLAY_MOVES_ARTIFACT_ROOT:-artifacts/replay-moves}
mkdir -p "$artifact_root"
attempt=$(mktemp -d "$artifact_root/attempt-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")
attempt=$(realpath "$attempt")
export DORADO_SOURCE="$source_path"
export REPLAY_MOVES_CAPTURE="${captures[0]}"
export REPLAY_MOVES_CAPTURES
REPLAY_MOVES_CAPTURES=$(IFS=:; echo "${captures[*]}")
export REPLAY_MOVES_OUTPUT="$attempt/replay"
{
  printf 'started=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)"
  printf 'DORADO_SOURCE=%q\nREPLAY_MOVES_CAPTURE=%q\nREPLAY_MOVES_CAPTURES=%q\nREPLAY_MOVES_OUTPUT=%q\n' "$DORADO_SOURCE" "$REPLAY_MOVES_CAPTURE" "$REPLAY_MOVES_CAPTURES" "$REPLAY_MOVES_OUTPUT"
  printf 'command=go test -count=1 -v ./verification/replaymoves/...\n'
} > "$attempt/command.txt"
set +e
go test -count=1 -v ./verification/replaymoves/... > "$attempt/stdout.txt" 2> "$attempt/stderr.txt"
status=$?
set -e
printf 'finished=%s\nexit_code=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)" "$status" >> "$attempt/command.txt"
sha256sum "$attempt/stdout.txt" "$attempt/stderr.txt" > "$attempt/log-sha256.txt"
cat "$attempt/stdout.txt"
cat "$attempt/stderr.txt" >&2
printf 'retained attempt %s\n' "$attempt"
exit "$status"
