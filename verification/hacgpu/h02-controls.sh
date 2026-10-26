#!/usr/bin/env bash
# H02 native rejection-control run.
#
# Each mutation removes exactly one guard from verification/tensor/hac_smallconv.inc,
# rebuilds the ABI4 backend, and re-runs the guard probe. The probe calls
# Device.SmallConvF16Unchecked, which bypasses every Go pre-check, so the outcome
# it prints can only come from the native boundary.
#
# A control passes when the mutated build changes the observed outcome in the
# direction the mutation predicts. The source is restored after every attempt and
# every failed attempt is preserved under artifacts/tensor.
set -euo pipefail
cd "$(dirname "$0")/../.." || exit 1
ROOT=$(pwd)
T=/home/jake/Projects/nanopore_basecaller/software/source-build-qualification-20261023/dependencies/torch-2.9.0.0-ont-CUDA-13.0-linux-x64-cxx11-abi/libtorch
A=/home/jake/Projects/nanopore_basecaller/software/source-build-qualification-20261023/archives/torch-2.9.0.0-ont-CUDA-13.0-linux-x64-cxx11-abi.zip
K=/home/jake/Projects/nanopore_basecaller/software/source-build-qualification-20261023/dependencies/libkoi-0.7.5-Linux-x86_64-cuda-13.0/libkoi-0.7.5-Linux-x86_64-cuda-13.0
SRC=verification/tensor/hac_smallconv.inc
OUT=${OUT:-artifacts/tensor/h02-controls}
ONLY=${1:-}
case "$ONLY" in ""|omitted-input-padding|wrong-output-stride|unknown-activation|non-multiple-stride|non-f16-storage) ;; *) echo "unknown control: $ONLY" >&2; exit 2 ;; esac
mkdir -p "$(dirname "$OUT")"
mkdir "$OUT" || { echo "attempt already exists: $OUT" >&2; exit 1; }
BACKUP=$OUT/source-original.inc
[ -f "$SRC" ] || { echo "missing $SRC (wrong working directory?)"; exit 1; }
cp "$SRC" "$BACKUP" || exit 1
: >"$OUT/transcript.txt"

restore() { cp "$BACKUP" "$SRC"; }
trap restore EXIT

build() {
  local dir=$1
  [ ! -e "$dir" ] || { echo "attempt already exists: $dir" >&2; return 1; }
  go run ./verification/tensor/build -torch "$T" -archive "$A" \
    -dorado-source external/dorado-source -koi "$K" \
    -cuda-family 13.0 -cuda-root /opt/cuda -out "$dir" >"$dir.build.log" 2>&1
}

probe() {
  local backend=$1 log=$2
  go run ./verification/hacgpu/h02/cmd/guards -backend "$backend" -runtime-admitted >"$log" 2>&1
  return $?
}

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
}

# A control names one probe case and the outcome its mutation must produce.
# Passing a name as the script's first argument runs only that control, so a
# corrected expectation does not require rebuilding every other mutated backend.
control() {
  local name=$1 case=$2 pattern=$3 replacement=$4 want=$5
  if [ -n "${ONLY:-}" ] && [ "$ONLY" != "$name" ]; then
    return
  fi
  mutate "$name" "$pattern" "$replacement" || { echo "$name: MUTATION ABSENT" | tee -a "$OUT/transcript.txt"; return 1; }
  local dir=$OUT/build-$name
  if ! build "$dir"; then
    echo "$name: BUILD FAILED (see immutable build log under $OUT)" | tee -a "$OUT/transcript.txt"
    return 1
  fi
  local log=$OUT/probe-$name.txt
  probe "$dir/libnanopore_tensor.so" "$log"
  local observed
  observed=$(grep -F "$case:" "$log" | tail -1)
  if [ -z "$observed" ]; then
    observed="$case: probe produced no line (exit $?)"
  fi
  echo "$name: $observed" | tee -a "$OUT/transcript.txt"
  case "$observed" in
    *"$want"*) echo "  -> control held: $name produced the predicted outcome" | tee -a "$OUT/transcript.txt" ;;
    *) echo "  -> CONTROL FAILED: expected to contain '$want'" | tee -a "$OUT/transcript.txt"; return 1 ;;
  esac
}

restore
if [ -z "$ONLY" ] && ! build "$OUT/build-baseline"; then
  echo "baseline: BUILD FAILED (see immutable build log under $OUT)"
  exit 1
fi
if [ -z "$ONLY" ]; then
  probe "$OUT/build-baseline/libnanopore_tensor.so" "$OUT/probe-baseline.txt"
  echo "baseline:" | tee -a "$OUT/transcript.txt"
  cat "$OUT/probe-baseline.txt" | tee -a "$OUT/transcript.txt"
fi

# 1. Omitted input padding: ConvStack.cpp:223 passes winlen/2. The request also
#    carries a wrong output stride so removing the padding guard advances to the
#    next guard rather than reaching the kernel with an out-of-bounds read.
control omitted-input-padding padding \
  'cuda_require(padding == window / 2, NT_CUDA_ARGUMENT,' \
  'if (false) cuda_require(padding == window / 2, NT_CUDA_ARGUMENT,' \
  'output batch stride differs from the view'

# 2. The cropped view keeps the parent's batch stride. Removing the guard lets the
#    kernel run with a stride that is not the view's, which is a wrong convolution.
control wrong-output-stride stride \
  'cuda_require(ot.stride(0) == out_n_stride && ot.stride(1) == c_out && ot.stride(2) == 1,' \
  'if (false) cuda_require(ot.stride(0) == out_n_stride && ot.stride(1) == c_out && ot.stride(2) == 1,' \
  'ok'

# 3. An activation outside get_koi_activation's range is not a different
#    convolution. Removing the throw lets the kernel run with SWISH instead.
control unknown-activation activation \
  'throw CUDAFailure(NT_CUDA_UNSUPPORTED, "unknown Koi activation for the small convolution");' \
  'return KOI_SWISH;' \
  'ok'

# 4. T_in must be a whole number of strides.
control non-multiple-stride stride-multiple \
  'cuda_require(t_out > 0 && t_out * stride == int64_t(t_in),' \
  'if (false) cuda_require(t_out > 0 && t_out * stride == int64_t(t_in),' \
  'input shape differs from N,T,C_in'

# 5. All four storages must be F16.
control non-f16-storage dtype \
  'cuda_require(p->dtype == NT_DTYPE_F16, NT_CUDA_DTYPE,
                         "small convolution requires F16 storage");' \
  'if (false) cuda_require(p->dtype == NT_DTYPE_F16, NT_CUDA_DTYPE,
                         "small convolution requires F16 storage");' \
  'ok'

restore
echo "source restored: $(sha256sum "$SRC" | cut -d' ' -f1)"
