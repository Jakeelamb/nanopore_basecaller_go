#!/usr/bin/env bash
set -euo pipefail
: "${NANOPORE_POD5_LIBRARY:?set the pinned POD5 0.3.39 shared-library path}"
: "${POD5_FIXTURE_DIR:?set the independent producer fixture directory}"
root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root"
receipt=$(mktemp -d "$root/artifacts/pod5/verify-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")
printf '%s\n' "$receipt"
run() {
    local name=$1
    shift
    printf '%q ' "$@" > "$receipt/$name.command"
    printf '\n' >> "$receipt/$name.command"
    date -u +%Y-%m-%dT%H:%M:%SZ > "$receipt/$name.started"
    set +e
    "$@" > "$receipt/$name.log" 2>&1
    local status=$?
    set -e
    printf '%s\n' "$status" > "$receipt/$name.exit"
    date -u +%Y-%m-%dT%H:%M:%SZ > "$receipt/$name.finished"
    cat "$receipt/$name.log"
    return "$status"
}
run pins sha256sum "$NANOPORE_POD5_LIBRARY" "$POD5_FIXTURE_DIR/manifest.json" internal/pod5/pod5_format/c_api.h internal/pod5/pod5_format/pod5_format_export.h verification/pod5/generate.py
run revision git rev-parse HEAD
run source-diff git diff -- internal/pod5 verification/pod5 docs/pod5-reader.md
run source-pins bash -c '{ find internal/pod5 verification/pod5 -type f -print0; printf "%s\0" docs/pod5-reader.md; } | sort -z | xargs -0 sha256sum'
run native env NANOPORE_POD5_LIBRARY="$NANOPORE_POD5_LIBRARY" POD5_FIXTURE_DIR="$POD5_FIXTURE_DIR" go test -count=1 -race -v ./internal/pod5
run nocgo env CGO_ENABLED=0 go test -count=1 -v ./internal/pod5
run vet go vet ./internal/pod5
