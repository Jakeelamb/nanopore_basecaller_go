#!/usr/bin/env bash
set -euo pipefail
: "${CALLER_CAPTURE_MANIFEST:?set the retained genuine caller capture manifest}"
: "${CALLER_STOCK_FASTQ:?set the retained official CPU FASTQ control}"
root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root"
receipt=$(mktemp -d "$root/artifacts/caller/check-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")
printf '%s\n' "$receipt"
run(){
    local name=$1
    shift
    printf '%q ' "$@" > "$receipt/$name.command"
    printf '\n' >> "$receipt/$name.command"
    date -u +%Y-%m-%dT%H:%M:%SZ > "$receipt/$name.started"
    set +e
    "$@" > "$receipt/$name.stdout" 2> "$receipt/$name.stderr"
    local status=$?
    set -e
    printf '%s\n' "$status" > "$receipt/$name.exit"
    date -u +%Y-%m-%dT%H:%M:%SZ > "$receipt/$name.finished"
    return "$status"
}
run revision git rev-parse HEAD
run source-pins bash -c '{ find internal/caller internal/decode cmd/basecaller verification/caller -type f -print0; printf "%s\0" docs/caller.md; } | sort -z | xargs -0 sha256sum'
run cached-pins sha256sum "$CALLER_CAPTURE_MANIFEST" "$CALLER_STOCK_FASTQ"
run race env DORADO_SOURCE="${DORADO_SOURCE:-$root/external/dorado-source}" CALLER_METADATA_ARTIFACT_DIR="$receipt/metadata-oracle" DECODE_ARTIFACT_DIR="$receipt/decode-oracle" CALLER_CAPTURE_MANIFEST="$CALLER_CAPTURE_MANIFEST" CALLER_STOCK_FASTQ="$CALLER_STOCK_FASTQ" go test -count=1 -race -v ./internal/caller ./internal/decode ./cmd/basecaller ./verification/caller/...
run nocgo env CGO_ENABLED=0 go test -count=1 -v ./internal/caller ./internal/decode ./cmd/basecaller ./verification/caller/...
run vet go vet ./internal/caller ./internal/decode ./cmd/basecaller ./verification/caller/...
run build go build -o "$receipt/basecaller" ./cmd/basecaller
run replay "$receipt/basecaller" replay --manifest "$CALLER_CAPTURE_MANIFEST"
run compare go run ./verification/caller/cmd --go-fastq "$receipt/replay.stdout" --stock-fastq "$CALLER_STOCK_FASTQ" --manifest "$CALLER_CAPTURE_MANIFEST"
run raw-cmp cmp "$receipt/replay.stdout" "$CALLER_STOCK_FASTQ"
run changed-field awk 'NR == 1 { sub(/\tch:i:[0-9]+/, "\tch:i:1") } { print }' "$receipt/replay.stdout"
if run reject-changed-field go run ./verification/caller/cmd --go-fastq "$receipt/changed-field.stdout" --stock-fastq "$CALLER_STOCK_FASTQ" --manifest "$CALLER_CAPTURE_MANIFEST"; then
    printf '%s\n' 'changed header field unexpectedly passed' >&2
    exit 1
fi
test "$(cat "$receipt/reject-changed-field.exit")" = 1
if command -v strace >/dev/null; then
    run replay-trace strace -f -e trace=execve,openat -o "$receipt/replay.strace" "$receipt/basecaller" replay --manifest "$CALLER_CAPTURE_MANIFEST"
    run trace-output-cmp cmp "$receipt/replay-trace.stdout" "$CALLER_STOCK_FASTQ"
fi
run cached-pins-after sha256sum "$CALLER_CAPTURE_MANIFEST" "$CALLER_STOCK_FASTQ"
run immutable-inputs cmp "$receipt/cached-pins.stdout" "$receipt/cached-pins-after.stdout"
cat "$receipt/compare.stdout"
