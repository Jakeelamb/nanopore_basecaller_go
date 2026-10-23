# SUP QKV half-weight preparation

`internal/model/koi_qkv.go` owns the fixed SUP host-weight preparation in Go.
It accepts 786,432 binary16 bit patterns in contiguous `[1536,512]` order.
The private result owns two fresh buffers, independent of the input, each other,
and every other invocation.

The unrounded buffer has shape `[3,8,64,512]`. Q and K interleave the lower and
upper halves of D. V keeps its original order. This preserved buffer is the input
required by a future int8 preparation unit. No int8 quantization runs here.

The final buffer has shape `[3,8,4,64,16,8]`, strides
`[262144,32768,8192,128,8,1]`, and tags
`[KOI_DIM_MAT_Q_K_V,H,D,C,d,c]`. Its values use modular half-bit rounding,
`(bits + 8) & 0xfff0`. Every bit pattern is admitted, including signed zero,
infinities, and NaN payloads. Rounding can turn `0x7bff` into `0x7c00`.
The input remains unchanged. A wrong element count returns a typed
`qkvWeightShapeError` and two nil result slices before allocation.

This module is not wired into the model graph or caller. Tensor ABI 3 is unchanged.
The endpoint excludes int8 scales, FP8, activations, RoPE, attention, Koi dispatch,
GPU execution, score tensors, decoder output, FASTQ, biological accuracy,
performance, and default SUP GPU behavior.

## Native evidence

The oracle compiles only a CPU storage adapter and three verbatim blocks from
`dorado/nn/TxModules.cpp` at Dorado revision
`8b8fc5d36a9c0baab262a743cb175b5878e38ca6`.
Its whole-file SHA256 is
`9e73e81b38c6b6e36a17566a8dfc16e2da2e54fab44dfdbcb85485d77998b701`.

| Inclusive lines | Operation | Extracted block SHA256 |
| --- | --- | --- |
| 99-106 | Half-bit rounding | `0244424d9c3a9058e4341d126025f2b9b983249205b4e06acaabfab6deff4c28` |
| 445-452 | Q/K interleave and V copy | `f6d271c9acedd154200144a979d61169c29ed34634293c20db80a177a00de0e6` |
| 462-464 | Half tiling | `27e0ca2485571bd6119b72a25848f82d42a29374c164b0f91da419e2cab22163` |

The adapter supplies minimal tensor storage for `self_attn->wqkv->weight` and
`wqkv_weights_f16.t`. It saves the interleaved buffer before tiling, saves the tile
before rounding, then calls the extracted `apply_rounding` with four removed bits.
Synthetic inputs copy raw half bytes into CPU half storage without conversion.
This qualifies extracted source-block computation, not the original module ABI.
The full native graph, `model.Open`, and `modeloracle.BuildSUP` do not run.

The runner reuses the existing distribution from
`artifacts/tensor/attempt-005-sup-abi3/manifest.json`. Its archive SHA256 is
`8680441dbe3b0990fa7a99a70319affe25bc51f0cd6e105c3e07d522d7355929`.
It verifies the archive, all 4,553 declared extracted members, qualified runtime
pins, and compiler hash. It records a fresh compiler version, flags, depfile,
generated source, executable, command clocks, exit status, and separate stdout
and stderr hashes. `/proc/self/maps` supplies observed mapped paths after the
producer runs. The manifest hashes those files and requires the qualified
`libtorch_cpu.so` and `libc10.so` paths. No Koi or CUDA library is linked.

The genuine fixture loads exactly one CPU float32 `[1536,512]` tensor from
`transformer_encoder.0.self_attn.Wqkv.weight.tensor`. Its archive SHA256 is
`b2260c11cadaf34d4bc14f0a9875829a51ecad7f4f6268604b2d3ae8d848f47d`,
checked before and after loading. One CPU half cast produces the saved input.
The float32 snapshot SHA256 is
`340546a13a60d28fa5b665f1f79bad4b9ffbe1dbede60c45d09c6715ca6651e7`.
The saved half input SHA256 is
`46d2edaa4b905fa34f611c9311ec9de2236c3f8ed29fed5c358084a57f59e2c2`.
The Go endpoint starts at those half bytes. This establishes no CPU-versus-GPU
cast claim.

## Reproduce capture and replay

Run from the Go port module root. The runner checks module identity before
creating outputs. Each output must be a fresh direct child of
`artifacts/koiweights`. Existing attempt directories are rejected.

```sh
go run ./verification/koiweights/cmd \
	-source "$PWD/external/dorado-source" \
	-tensor-manifest "$PWD/artifacts/tensor/attempt-005-sup-abi3/manifest.json" \
	-weight /home/jake/Projects/nanopore_basecaller/models/dna_r10.4.1_e8.2_400bps_sup@v5.2.0/transformer_encoder.0.self_attn.Wqkv.weight.tensor \
	-out "$PWD/artifacts/koiweights/attempt-qkv-f16-NEW"

go run ./verification/koiweights/cmd \
	-replay "$PWD/artifacts/koiweights/attempt-qkv-f16-002/manifest.json" \
	-out "$PWD/artifacts/koiweights/replay-qkv-f16-NEW"
```

Capture opens the already-existing stock `results/stock/host.lock` read-only,
without create or truncate, and acquires nonblocking `LOCK_EX`. It holds admission
through native compilation and CPU Torch execution. Contention preserves an
`admission_blocked` manifest, exits with code 3, and launches no compiler or
producer command. The `-admission-control` mode deliberately holds that original
lock and requires this blocked result. It takes the same capture arguments.

Replay launches only the Go consumer. It validates all declared producer pins,
saved bytes, fixed metadata, and current verification sources. It records current
consumer source hashes and requires `TestSUPQKVF16Native` to run and pass rather
than skip. A consumer-only edit requires a new comparison receipt, while a native
capture whose producer pins still match remains reusable. A failed consumer
command does not invalidate an otherwise complete retained native capture.

## Accepted checks and retained receipts

The final CPU capture on 2026-10-23 and its producer-free replay passed. Both Go
result arrays matched all three complete fixtures bitwise. Each QKV fixture has
786,432 elements. Fixtures contain axis-varying patterns, repeated complete half
patterns, and the genuine first-layer weights. The saved native pre-round tile
also matches the corresponding permutation of the preserved Go result. Exhaustive
rounding checks cover all 65,536 half bit patterns.

Six actual Go source mutants fail behavioral tests. They omit Q/K interleaving,
interleave V, change tile order, round the preserved buffer early, saturate, or
use round-to-even. `-mutants` runs these through Go source overlays and retains
payloads as `.go.txt`, their hashes, configurations, and raw test logs.

Capture and replay each reject 16 controls covering malformed dtype, endian,
shape, foreign architecture shape, byte count, output metadata, modified input and
output bytes, copied source mutation, wrong archive or library pin, and a held
lock. The separate original-lock control confirms capture records no launched
command when admission is blocked.

| Receipt beneath `artifacts/koiweights` | Result |
| --- | --- |
| `attempt-qkv-f16-002/manifest.json` | Final CPU native capture and Go comparison passed |
| `replay-qkv-f16-002/manifest.json` | Fresh Go comparison passed; its sole command is `go test` |
| `attempt-mutants-002/manifest.json` | All six altered implementations rejected by behavioral tests |
| `attempt-admission-002/manifest.json` | Original lock held; capture blocked before commands or generated outputs |
| `attempt-checks-002/results.tsv` | All seven final checks exited 0 |
| `core-parity-qkv-f16-002` | Maintained core parity passed its listed contracts |

The exact final checks, their raw logs, clocks, exit codes, and source hashes are
retained in `attempt-checks-002`. The commands were:

```sh
go test -count=1 ./...
CGO_ENABLED=0 go test -count=1 ./...
go test -count=1 -race ./...
go vet ./...
NANOPORE_KOI_QKV_MANIFEST="$PWD/artifacts/koiweights/attempt-qkv-f16-002/manifest.json" \
	go test -count=1 -race ./internal/model -run '^TestSUPQKVF16'
CGO_ENABLED=0 NANOPORE_KOI_QKV_MANIFEST="$PWD/artifacts/koiweights/attempt-qkv-f16-002/manifest.json" \
	go test -count=1 ./internal/model -run '^TestSUPQKVF16'
./verification/verify.sh --source "$PWD/external/dorado-source" \
	--artifacts "$PWD/artifacts/koiweights/core-parity-qkv-f16-002" --refresh
```

Ordinary tests explicitly skip the native fixture test when its environment
variable is absent. CPU CI and core parity remain separate from this optional
native weight evidence.

`attempt-checks-001` preserves the first broad-check failures. Go discovered
ignored `.go` artifact snapshots, including this unit's first mutation payloads.
The runner now saves `.go.txt` payloads. The coordinator added the nested
`artifacts/go.mod` boundary while preserving the older snapshots. The final broad
checks pass after those corrections. Earlier captures, replay, mutants, and lock
receipts remain retained. `attempt-incident-001/receipt.json` records the corrected
wrong-directory invocation and exact file ownership allowlist. No existing file
in the original experiment was changed by that invocation.
