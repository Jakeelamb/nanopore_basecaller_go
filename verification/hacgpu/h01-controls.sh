#!/usr/bin/env bash
# H01 rejection controls. Each control copies an admitted capture, applies one
# mutation, and requires `-unit verify` to reject it. The capture itself is never
# modified, and every failed verification is preserved under artifacts.
#
# Usage: verification/hacgpu/h01-controls.sh [capture-dir]
set -u
cd "$(dirname "$0")/../.." || exit 1
ROOT=$(pwd)
SRC=${1:-artifacts/tensor/hacgpu-capture-006/stages}
WORK=artifacts/tensor/h01-controls
LOG=/tmp/opencode/h01-controls
mkdir -p "$WORK" "$LOG"

if [ ! -d "$SRC" ]; then
  echo "no admitted capture at $SRC" >&2
  exit 1
fi

fail=0
control() {
  local name=$1
  shift
  local dir=$WORK/$name
  rm -rf "$dir"
  cp -a "$SRC" "$dir"
  python3 - "$dir" "$name" <<'PY' || { echo "$name: MUTATION FAILED"; fail=1; return; }
import json, sys
dir, name = sys.argv[1:3]
path = dir + "/capture.json"
m = json.load(open(path))
if name == "corrupt-payload":
    p = dir + "/stage-flstm3.bin"
    body = bytearray(open(p, "rb").read())
    body[0] ^= 0xFF
    open(p, "wb").write(body)
elif name == "missing-payload":
    import os
    os.remove(dir + "/stage-conv1.bin")
elif name == "extra-stage":
    m["stages"].append(dict(m["stages"][-1], name="linear2", kind="projection"))
elif name == "stack-digest-drift":
    m["stack_sha256"]["convs"] = "0" * 64
elif name == "scores-digest-drift":
    m["scores_sha256"] = "0" * 64
elif name == "batch-not-granular":
    m["batch"] = 63
    for s in m["stages"]:
        s["N"] = 63
elif name == "layer-relabelled":
    m["stages"][3]["layer"] = 9
elif name == "t-out-not-tiled":
    m["t_out"] = 48
elif name == "graph-shape-drift":
    m["linear2_present"] = True
elif name == "unwritten-drift":
    m["stages"][4]["unwritten_t"] = [[30, 20]]
elif name == "foreign-source":
    m["source_commit"] = "0" * 40
else:
    raise SystemExit("unknown control " + name)
json.dump(m, open(path, "w"), indent=2)
PY
  go run ./verification/hacgpu/cmd -unit verify -out "$dir" >"$LOG/$name.log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then
    echo "$name: ACCEPTED (control did not fire)"
    fail=1
  else
    echo "$name: rejected -> $(grep -v '^exit status' "$LOG/$name.log" | tail -1)"
  fi
}

control corrupt-payload
control missing-payload
control extra-stage
control stack-digest-drift
control scores-digest-drift
control batch-not-granular
control layer-relabelled
control t-out-not-tiled
control graph-shape-drift
control unwritten-drift
control foreign-source

if [ $fail -ne 0 ]; then
  echo "H01 controls failed" >&2
  exit 1
fi
echo "H01 controls: 11 mutations, all rejected"
