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
run source-pins bash -c '{ find internal/caller cmd/basecaller verification/caller -type f -print0; printf "%s\0" docs/caller.md; } | sort -z | xargs -0 sha256sum'
run cached-pins sha256sum "$CALLER_CAPTURE_MANIFEST" "$CALLER_STOCK_FASTQ"
run race env CALLER_CAPTURE_MANIFEST="$CALLER_CAPTURE_MANIFEST" CALLER_STOCK_FASTQ="$CALLER_STOCK_FASTQ" go test -count=1 -race -v ./internal/caller ./cmd/basecaller ./verification/caller/...
run nocgo env CGO_ENABLED=0 go test -count=1 -v ./internal/caller ./cmd/basecaller ./verification/caller/...
run vet go vet ./internal/caller ./cmd/basecaller ./verification/caller/...
run build go build -o "$receipt/basecaller" ./cmd/basecaller
run replay "$receipt/basecaller" replay --manifest "$CALLER_CAPTURE_MANIFEST"
run compare go run ./verification/caller/cmd --go-fastq "$receipt/replay.stdout" --stock-fastq "$CALLER_STOCK_FASTQ" --manifest "$CALLER_CAPTURE_MANIFEST"
cat "$receipt/compare.stdout"
