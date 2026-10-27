#!/usr/bin/env bash
# H03 native rejection-control run.
#
# Two kinds of control, both preserved under artifacts/tensor:
#
#   * Native guard mutations. Each removes exactly one guard from
#     verification/tensor/hac_lastconv.inc, rebuilds the ABI4 backend and re-runs
#     the guard probe. The probe calls Device.LastConvKoiUnchecked /
#     LastConvLinearUnchecked, which bypass every Go pre-check, so the outcome it
#     prints can only come from the native boundary. A control holds when the
#     mutated build changes the observed outcome for that case.
#
#   * Replay mutants. Each changes one input to the maintained differential
#     replay and must make the conv2 comparison fail:
#       - forced-fallback: koi_conv=false selects host_linear, which is a
#         different kernel and does not reproduce the capture;
#       - f32-accum: koi_conv_f32=true changes the arithmetic;
#       - i8-corrupt: one converted output byte is replaced after the primitive.
#
# The source is restored after every attempt and every failed attempt is kept.
set -euo pipefail
cd "$(dirname "$0")/../.." || exit 1
ROOT=$(pwd)
T=/home/jake/Projects/nanopore_basecaller/software/source-build-qualification-20261023/dependencies/torch-2.9.0.0-ont-CUDA-13.0-linux-x64-cxx11-abi/libtorch
A=/home/jake/Projects/nanopore_basecaller/software/source-build-qualification-20261023/archives/torch-2.9.0.0-ont-CUDA-13.0-linux-x64-cxx11-abi.zip
K=/home/jake/Projects/nanopore_basecaller/software/source-build-qualification-20261023/dependencies/libkoi-0.7.5-Linux-x86_64-cuda-13.0/libkoi-0.7.5-Linux-x86_64-cuda-13.0
SRC=verification/tensor/hac_lastconv.inc
OUT=${OUT:-artifacts/tensor/h03-controls}
CAPTURE=${CAPTURE:-/home/jake/.local/share/opencode/worktree/e3db79/H01/artifacts/tensor/hacgpu-capture-008/stages}
MODEL=${MODEL:-/home/jake/Projects/nanopore_basecaller/models/dna_r10.4.1_e8.2_400bps_hac@v6.0.0}
ONLY=${1:-}
case "$ONLY" in ""|padding|input-rows|input-strides|output-strides|weight-shape|output-type|activation|accumulation|overlap|forced-fallback|f32-accum|i8-corrupt) ;; *) echo "unknown control: $ONLY" >&2; exit 2 ;; esac
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
  go run ./verification/hacgpu/h03/cmd/guards -backend "$backend" -runtime-admitted >"$log" 2>&1
  return $?
}

replay() {
  local backend=$1 out=$2 log=$3
  shift 3
  go run ./verification/hacgpu/cmd -unit last-conv \
    -capture "$CAPTURE" -model "$MODEL" -backend "$backend" \
    -out "$out" -runtime-admitted "$@" >"$log" 2>&1
  return $?
}

# Disable the cuda_require whose message contains $1, by prefixing the nearest
# preceding `cuda_require(` with `if (false) `. Anchoring on the message instead
# of the whole statement keeps the mutation immune to formatting.
disable_guard() {
  restore
  python3 - "$SRC" "$1" <<'PY'
import sys
path, anchor = sys.argv[1:3]
body = open(path).read()
if anchor not in body:
    raise SystemExit("guard anchor absent: " + anchor)
idx = body.index(anchor)
start = body.rindex("cuda_require(", 0, idx)
open(path, "w").write(body[:start] + "if (false) " + body[start:])
PY
}

replace_once() {
  restore
  python3 - "$SRC" "$1" "$2" <<'PY'
import sys
path, pattern, replacement = sys.argv[1:4]
body = open(path).read()
if pattern not in body:
    raise SystemExit("mutation pattern absent: " + pattern)
open(path, "w").write(body.replace(pattern, replacement, 1))
PY
}

# A guard control names one probe case, the guard to disable and, optionally, the
# outcome the mutation must produce. Without a `want` the control holds when the
# observed line differs from the baseline line for the same case.
control() {
  local name=$1 case=$2 anchor=$3 want=${4:-}
  if [ -n "${ONLY:-}" ] && [ "$ONLY" != "$name" ]; then
    return
  fi
  if ! disable_guard "$anchor"; then
    echo "$name: MUTATION ABSENT ($anchor)" | tee -a "$OUT/transcript.txt"
    return 1
  fi
  local dir=$OUT/build-$name
  if ! build "$dir"; then
    echo "$name: BUILD FAILED (see immutable build log under $OUT)" | tee -a "$OUT/transcript.txt"
    return 1
  fi
  local log=$OUT/probe-$name.txt
  probe "$dir/libnanopore_tensor.so" "$log"
  local observed baseline
  # Anchor on the whole line prefix: "weight-shape:" is a suffix of
  # "fallback-weight-shape:", so a plain substring match picks the wrong case.
  observed=$(grep -E "^$case: " "$log" | tail -1)
  baseline=$(grep -E "^$case: " "$OUT/probe-baseline.txt" | tail -1)
  [ -n "$observed" ] || observed="$case: probe produced no line"
  echo "$name: baseline [$baseline]" | tee -a "$OUT/transcript.txt"
  echo "$name: mutated  [$observed]" | tee -a "$OUT/transcript.txt"
  if [ -n "$want" ]; then
    case "$observed" in
      *"$want"*) echo "  -> control held: $name produced the predicted outcome" | tee -a "$OUT/transcript.txt" ;;
      *) echo "  -> CONTROL FAILED: expected to contain '$want'" | tee -a "$OUT/transcript.txt"; return 1 ;;
    esac
    return
  fi
  if [ "$observed" = "$baseline" ]; then
    echo "  -> CONTROL FAILED: removing the guard changed nothing" | tee -a "$OUT/transcript.txt"
    return 1
  else
    echo "  -> control held: removing the guard changed the observed outcome" | tee -a "$OUT/transcript.txt"
  fi
}

# A replacement control changes one statement rather than disabling a
# cuda_require, for guards that live in a helper.
control_replace() {
  local name=$1 case=$2 pattern=$3 replacement=$4 want=${5:-}
  if [ -n "${ONLY:-}" ] && [ "$ONLY" != "$name" ]; then
    return
  fi
  if ! replace_once "$pattern" "$replacement"; then
    echo "$name: MUTATION ABSENT ($pattern)" | tee -a "$OUT/transcript.txt"
    return 1
  fi
  local dir=$OUT/build-$name
  if ! build "$dir"; then
    echo "$name: BUILD FAILED (see immutable build log under $OUT)" | tee -a "$OUT/transcript.txt"
    return 1
  fi
  local log=$OUT/probe-$name.txt
  probe "$dir/libnanopore_tensor.so" "$log"
  local observed baseline
  # Anchor on the whole line prefix: "weight-shape:" is a suffix of
  # "fallback-weight-shape:", so a plain substring match picks the wrong case.
  observed=$(grep -E "^$case: " "$log" | tail -1)
  baseline=$(grep -E "^$case: " "$OUT/probe-baseline.txt" | tail -1)
  [ -n "$observed" ] || observed="$case: probe produced no line"
  echo "$name: baseline [$baseline]" | tee -a "$OUT/transcript.txt"
  echo "$name: mutated  [$observed]" | tee -a "$OUT/transcript.txt"
  if [ -n "$want" ]; then
    case "$observed" in
      *"$want"*) echo "  -> control held: $name produced the predicted outcome" | tee -a "$OUT/transcript.txt" ;;
      *) echo "  -> CONTROL FAILED: expected to contain '$want'" | tee -a "$OUT/transcript.txt"; return 1 ;;
    esac
    return
  fi
  if [ "$observed" = "$baseline" ]; then
    echo "  -> CONTROL FAILED: removing the guard changed nothing" | tee -a "$OUT/transcript.txt"
    return 1
  else
    echo "  -> control held: removing the guard changed the observed outcome" | tee -a "$OUT/transcript.txt"
  fi
}

# A replay mutant must make the differential comparison fail, and the receipt
# must name the primitive or the digest that changed.
replay_control() {
  local name=$1 want=$2
  shift 2
  if [ -n "${ONLY:-}" ] && [ "$ONLY" != "$name" ]; then
    return
  fi
  local dir=$OUT/replay-$name log=$OUT/replay-$name.txt
  [ ! -e "$dir" ] || { echo "attempt already exists: $dir" >&2; return 1; }
  if replay "$BACKEND" "$dir" "$log" "$@"; then
    echo "$name: replay PASSED (control did not fire)" | tee -a "$OUT/transcript.txt"
    return 1
  fi
  local error
  error=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['error'])" "$dir/receipt.json" 2>/dev/null)
  echo "$name: ${error:-no receipt (see $log)}" | tee -a "$OUT/transcript.txt"
  case "$error" in
    *"$want"*) echo "  -> control held: $name made the comparison fail" | tee -a "$OUT/transcript.txt" ;;
    *) echo "  -> CONTROL FAILED: expected the error to contain '$want'" | tee -a "$OUT/transcript.txt"; return 1 ;;
  esac
  [ -f "$dir/receipt.json" ] || return
  python3 - "$dir/receipt.json" <<'PY' | tee -a "$OUT/transcript.txt"
import json, sys
r = json.load(open(sys.argv[1]))
print("     dispatch:", json.dumps(r.get("dispatch")))
for s in r.get("stages", []):
    print("     stage:", s["name"], "equal", s["equal"], "go", s["go_written_sha256"][:16], "capture", s["capture_written_sha256"][:16])
PY
}

restore
if ! build "$OUT/build-baseline"; then
  echo "baseline: BUILD FAILED (see immutable build log under $OUT)"
  exit 1
fi
BACKEND=$OUT/build-baseline/libnanopore_tensor.so
probe "$BACKEND" "$OUT/probe-baseline.txt"
echo "baseline:" | tee -a "$OUT/transcript.txt"
cat "$OUT/probe-baseline.txt" | tee -a "$OUT/transcript.txt"
if [ -z "${ONLY:-}" ]; then
  if replay "$BACKEND" "$OUT/replay-baseline" "$OUT/replay-baseline.txt"; then
    echo "baseline replay: lastconv_passed" | tee -a "$OUT/transcript.txt"
  else
    echo "baseline replay: FAILED" | tee -a "$OUT/transcript.txt"
    cat "$OUT/replay-baseline.txt" | tee -a "$OUT/transcript.txt"
    exit 1
  fi
fi

# 1. ConvStack.cpp:222-223 passes `winlen / 2`. Removing the guard advances to
#    the allocation-rows guard rather than reaching the kernel with a window that
#    does not match the declared padding.
control padding padding 'last convolution padding must be window/2' \
  'last convolution input rows are not T_in plus two paddings'

# 2. The input is the contiguous [N, T_in + 2*padding, C_in] allocation. Removing
#    the rows guard lets T_in disagree with the allocation.
control input-rows input-rows 'last convolution input rows are not T_in plus two paddings'

# 3. The input strides are `(T_alloc*C_in, C_in, 1)`. Removing the guard lets the
#    kernel read through a different batch stride.
control input-strides input-strides 'last convolution input strides differ from the view'

# 4. The output view is the transposed CUTLASS view, so its strides are not the
#    contiguous pair. Removing the guard lets the kernel write through a
#    different stride than the declared one.
control output-strides output-strides 'last convolution output strides differ from the view'

# 5. The weights are [C_out, W*C_in]. Removing the guard lets a short weight
#    tensor reach the primitive.
control weight-shape weight-shape 'last convolution weights shape differs from C_out,W*C_in'

# 6. An activation outside get_koi_activation's range is not a different
#    convolution. Replacing the throw with a default lets the kernel run.
control_replace activation activation \
  'throw CUDAFailure(NT_CUDA_UNSUPPORTED, "unknown Koi activation for the last convolution");' \
  'return KOI_SWISH;'

# 7. The declared Koi output type must agree with the storage's dtype. Removing
#    the guard lets an I8 request write into an F16 buffer.
control_replace output-type output-type \
  'lastconv_type_matches(out, out_type, d, "last convolution output type differs from its storage");' \
  'if (false) lastconv_type_matches(out, out_type, d, "last convolution output type differs from its storage");'

# 8. Accumulation is F16 or F32 (ConvStack.cpp:288). Removing the guard lets an
#    I8 accumulation reach the primitive.
control accumulation accumulation 'last convolution accumulation must be F16 or F32'

# 9. The output view must not overlap the input. Removing the guard lets the
#    kernel write over the input it is reading.
control_replace overlap overlap \
  'storage_overlap(p, out, d, "last convolution output overlaps an input interval");' \
  'if (false) storage_overlap(p, out, d, "last convolution output overlaps an input interval");'

# Replay mutants run against the unmutated baseline backend.
restore
BACKEND=$OUT/build-baseline/libnanopore_tensor.so
replay_control forced-fallback 'differs from the capture' -koi-conv=false
replay_control f32-accum 'differs from the capture' -koi-conv-f32

# 10. The I8 conversion is load-bearing: replacing one converted output byte
#     after the primitive must be caught by the byte-for-byte comparison. The
#     write goes through cudaMemcpyAsync because data_ptr() is device memory.
if [ -z "${ONLY:-}" ] || [ "$ONLY" = "i8-corrupt" ]; then
  replace_once 'bt.data_ptr(), act, koi_in, koi_out, koi_accum);' \
'bt.data_ptr(), act, koi_in, koi_out, koi_accum);
            if (*koi_result == 0) {
                static const unsigned char corrupt[2] = {0x00, 0xff};
                cudaMemcpyAsync(static_cast<unsigned char*>(ot.data_ptr()), corrupt, 2,
                                cudaMemcpyHostToDevice, s->stream);
            }'
  if build "$OUT/build-i8-corrupt"; then
    BACKEND=$OUT/build-i8-corrupt/libnanopore_tensor.so
    replay_control i8-corrupt 'differs from the capture'
  else
    echo "i8-corrupt: BUILD FAILED (see immutable build log under $OUT)" | tee -a "$OUT/transcript.txt"
    return 1
  fi
fi

restore
echo "source restored: $(sha256sum "$SRC" | cut -d' ' -f1)"
