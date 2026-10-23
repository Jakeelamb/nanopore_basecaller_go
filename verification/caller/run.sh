#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root"
stock=${STOCK_ROOT:-/home/jake/Projects/nanopore_basecaller}
tensor=${TENSOR_LIBRARY:-$root/artifacts/tensor/attempt-004/libnanopore_tensor.so}
pod5=${POD5_LIBRARY:-$root/external/backend/pod5/lib/libpod5_format.so}
control=${STOCK_CONTROL:-$root/artifacts/native-cpu-control/stock-hac-dev1-001/stdout}
receipt=$(mktemp -d "$root/artifacts/caller/attempt-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")
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
run build go build -o "$receipt/basecaller" ./cmd/basecaller
run revision git rev-parse HEAD
run pins sha256sum "$tensor" "$pod5" "$control" "$receipt/basecaller"
run source-pins bash -c 'find internal/caller cmd/basecaller verification/caller -type f -print0 | sort -z | xargs -0 sha256sum'
run call timeout 600 "$receipt/basecaller" \
  --model "$stock/models/dna_r10.4.1_e8.2_400bps_hac@v6.0.0" \
  --pod5 "$stock/data/raw/HG002/PAW70337/PAW70337_66b2eea5_de8117b1_33.pod5" \
  --read-id bb76c31a-36f0-48a0-abd2-107a10e1ade0 \
  --tensor-library "$tensor" --pod5-library "$pod5" --device cpu \
  --chunksize 12288 --overlap 600 --no-trim --disable-read-splitting \
  --capture "$receipt/capture"
run compare go run ./verification/caller/cmd \
  --go-fastq "$receipt/call.stdout" --stock-fastq "$control" \
  --manifest "$receipt/capture/bb76c31a-36f0-48a0-abd2-107a10e1ade0/manifest.json"
cat "$receipt/compare.stdout"
