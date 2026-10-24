#!/usr/bin/env bash
# Native rejection-control mutation run for P02. Each mutation removes one guard
# from verification/tensor/cuda_storage.inc, rebuilds the backend, runs the
# storage-koi capture, and records the observed failure. The source is restored
# after every attempt, and every failed attempt is preserved under artifacts.
set -u
cd "$(dirname "$0")/../../.." || exit 1
ROOT=$(pwd)
T=/home/jake/Projects/nanopore_basecaller/software/source-build-qualification-20261023/dependencies/torch-2.9.0.0-ont-CUDA-13.0-linux-x64-cxx11-abi/libtorch
A=/home/jake/Projects/nanopore_basecaller/software/source-build-qualification-20261023/archives/torch-2.9.0.0-ont-CUDA-13.0-linux-x64-cxx11-abi.zip
SRC=verification/tensor/cuda_storage.inc
BACKUP=/tmp/opencode/p02-cuda_storage.inc.orig
mkdir -p /tmp/opencode
cp "$SRC" "$BACKUP"

restore() { cp "$BACKUP" "$SRC"; }
trap restore EXIT

mutate() {
  local name=$1 pattern=$2 replacement=$3
  restore
  python3 - "$SRC" "$pattern" "$replacement" <<'PY'
import sys
path, pattern, replacement = sys.argv[1:4]
body = open(path).read()
if pattern not in body:
    raise SystemExit("mutation pattern absent: " + pattern)
open(path, "w").write(body.replace(pattern, replacement, 1))
PY
  local build=artifacts/tensor/p02-mutant-$name
  local capture=artifacts/tensor/p02-mutant-$name-capture
  rm -rf "$build" "$capture"
  go run ./verification/tensor/build -torch "$T" -archive "$A" \
    -dorado-source /home/jake/Projects/nanopore_basecaller_go/external/dorado-source \
    -cuda-family 13.0 -cuda-root /opt/cuda -out "$build" >/tmp/opencode/p02-$name-build.log 2>&1
  local built=$?
  if [ $built -ne 0 ]; then
    echo "$name: BUILD FAILED (see /tmp/opencode/p02-$name-build.log)"
    return
  fi
  go run ./verification/tensor/cuda -backend "$build/libnanopore_tensor.so" -unit storage-koi \
    -runtime-admitted -ordinal 0 -out "$capture" >/tmp/opencode/p02-$name-run.log 2>&1
  local ran=$?
  local observed="capture unexpectedly passed"
  if [ $ran -ne 0 ]; then
    observed=$(grep -o 'cuda_storage_device_test.go:[0-9]*: [^"]*' "$capture"/storage-koi-device.stdout 2>/dev/null | tail -1)
    [ -z "$observed" ] && observed=$(grep -o '"Output":"[^"]*"' "$capture"/storage-koi-device.stdout 2>/dev/null | tail -2 | head -1)
    [ -z "$observed" ] && observed="capture exit $ran"
  fi
  echo "$name: $observed"
}

mutate overlap 'cuda_require(a->end <= b->offset || b->end <= a->offset, NT_CUDA_LAYOUT, why);' 'if (false) cuda_require(a->end <= b->offset || b->end <= a->offset, NT_CUDA_LAYOUT, why);'
mutate owner-release 'if (item.second->scale_owner == id) {' 'if (false && item.second->scale_owner == id) {'
mutate scale-dtype 'cuda_require(p->dtype == NT_DTYPE_I8, NT_CUDA_DTYPE, "Koi scale owner requires an I8 tensor");' 'if (false) cuda_require(p->dtype == NT_DTYPE_I8, NT_CUDA_DTYPE, "Koi scale owner requires an I8 tensor");'
mutate contiguity 'cuda_require(p->tensor.is_contiguous(), NT_CUDA_LAYOUT, why);' 'if (false) cuda_require(p->tensor.is_contiguous(), NT_CUDA_LAYOUT, why);'
mutate view-bounds 'cuda_require(end <= parent->end, NT_CUDA_LAYOUT, "storage view exceeds parent interval");' 'if (false) cuda_require(end <= parent->end, NT_CUDA_LAYOUT, "storage view exceeds parent interval");'
mutate dtype-copy 'cuda_require(dst->dtype == src->dtype, NT_CUDA_DTYPE, "storage copy dtype differs");' 'if (false) cuda_require(dst->dtype == src->dtype, NT_CUDA_DTYPE, "storage copy dtype differs");'
mutate fill-width 'cuda_require(pattern && count == size, NT_CUDA_LAYOUT, "storage fill pattern length differs from element size");' 'if (false) cuda_require(pattern && count == size, NT_CUDA_LAYOUT, "storage fill pattern length differs from element size");'
