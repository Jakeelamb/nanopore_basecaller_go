#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root"
mkdir -p artifacts/caller
receipt=$(mktemp -d "$root/artifacts/caller/default-check-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")
printf '%s\n' "$receipt"
run(){
    local name=$1
    shift
    printf '%q ' "$@" > "$receipt/$name.command"
    printf '\n' >> "$receipt/$name.command"
    date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$receipt/$name.started"
    set +e
    "$@" > "$receipt/$name.stdout" 2> "$receipt/$name.stderr"
    local status=$?
    set -e
    printf '%s\n' "$status" > "$receipt/$name.exit"
    date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$receipt/$name.finished"
    sha256sum "$receipt/$name.stdout" "$receipt/$name.stderr" > "$receipt/$name.streams.sha256"
    return "$status"
}
run revision git rev-parse HEAD
run runtime go version
run source-before bash -c 'find internal/caller internal/postcall cmd/basecaller verification/caller -type f -print0 | sort -z | xargs -0 sha256sum'
run ordinary go test -count=1 -v ./internal/caller ./internal/postcall ./cmd/basecaller ./verification/caller/...
run race go test -count=1 -race ./internal/caller ./internal/postcall ./cmd/basecaller ./verification/caller/...
run nocgo env CGO_ENABLED=0 go test -count=1 ./internal/caller ./internal/postcall ./cmd/basecaller ./verification/caller/...
run vet go vet ./internal/caller ./internal/postcall ./cmd/basecaller ./verification/caller/...
run metadata env DORADO_SOURCE="${DORADO_SOURCE:-$root/external/dorado-source}" CALLER_METADATA_ARTIFACT_DIR="$receipt/metadata-oracle" go test -count=1 -v -run 'TestPinnedNativeFASTQMetadata|TestNativeFASTQSourceMutationRejected' ./internal/caller
run cached-cli go run ./verification/caller/defaultcmd --output "$receipt/cached-cli"
run source-after bash -c 'find internal/caller internal/postcall cmd/basecaller verification/caller -type f -print0 | sort -z | xargs -0 sha256sum'
run source-freeze cmp "$receipt/source-before.stdout" "$receipt/source-after.stdout"
cat "$receipt/cached-cli.stdout"
