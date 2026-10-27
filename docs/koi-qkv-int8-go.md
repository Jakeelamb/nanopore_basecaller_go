# Go SUP QKV int8 cache arithmetic

The bounded host differential passes against the saved CUDA S01 observations.
The production Go path quantizes already interleaved, unrounded half
`[3,8,64,512]` weights into contiguous int8 `[3,8,4,32,16,16]` tiles and float32
inverse scales `[3,8,64]`. This is cache arithmetic only. It does not execute a
SUP graph, GPU kernel, basecaller, or biological comparison.

## Production path

`internal/model/koi_qkv_int8.go` implements `BuildSUPQKVInt8` and the row operation
`QuantizeSUPQKVRows`. The latter accepts complete nonempty 512-value rows and
returns int8 rows plus one float32 inverse scale per row. The full cache builder
requires exactly 1536 rows.

Each row must contain only finite half values, and its absolute maximum must have
half bits `0x1801..0x7bff`, inclusive. Individual zero and subnormal values are
allowed. `QKVRowError` identifies the row and one of `nonfinite`, `zero`, or
`scale-overflow`. Rejection returns no partial output.

The quantization reuses `QuantizeFLSTMRows`, which ports the same
`utils::quantize_tensor` source used by QKV. This preserves one arithmetic
implementation for the half scale, half product, ties-to-even integer rounding,
and clipping to `[-127,127]`. A separate QKV quantizer would duplicate those
rules. QKV then takes the float32 reciprocal of the widened half scale and
reuses `permuteContiguous` for the source's tile permutation.

The pinned source is Dorado `8b8fc5d36a9c0baab262a743cb175b5878e38ca6`.
`TxModules.cpp:449-465` establishes the input boundary and cache layouts.
Lines 455-461 quantize along the last axis, invert the scale, and transform
`[3,8,4,16,32,16]` with permutation `[0,1,2,4,3,5]`.
`tensor_utils.cpp:291-298` supplies the arithmetic. The maintained S01 extractor
checks those source pins before this differential runs.

## Retained differential

Run the standalone consumer from the repository root with a fresh output path:

```sh
env GOMAXPROCS=2 GOMEMLIMIT=3GiB CGO_ENABLED=0 GOPROXY=off GOSUMDB=off \
  go run ./verification/koiquant/s02/cmd -out artifacts/koiquant/NEW-S02-ATTEMPT
```

The consumer requires the accepted CUDA import SHA256
`b65fc01a95b46530cfd73f25f8762c2b67752281edc89257278afefbf74e29e8` and CPU/CUDA
comparison SHA256
`13466965436bd283913a24587167a9ee1e67d823accd5ceeb5193c1fd3008339`.
It checks the raw CUDA manifest pin, all saved stage descriptors and payload
hashes, device metadata, and the native row-classification counts. It does not
rerun the native producer or repeat the accepted S01 runtime audit.

The consumer runs production arithmetic against every admitted row. It checks
int8 values, exact float32 reciprocal bits, and the final tile positions. It
separately calls the production row operation on every excluded row and requires
the correct typed rejection. Any fixture containing an excluded row must also
fail the full cache builder.

To exercise full-shape tiling for mixed fixtures, excluded rows are replaced
with a finite filler row only after their rejection is established. Output
directories retain `admitted-mask.bin`, `tiled-i8.bin`, and
`inverse-f32le.bin`, each with a manifest hash. Only the original admitted rows
are compared against the native outputs. Filler values are not parity evidence.

The completed attempt is
`artifacts/koiquant/attempt-s02-host-20261024-003/differential/manifest.json`.

| Check | Observed result |
| --- | --- |
| Fixtures and rows accounted for | 31 and 47616 |
| Admitted rows | 34992 |
| Explicitly rejected rows | 12624 |
| Rejection classes | 3120 nonfinite, 1536 zero, 7968 scale overflow |
| Exact int8 elements compared | 17915904 |
| Exact tile elements compared | 17915904 |
| Exact float32 reciprocal bit patterns compared | 34992 |
| Production mismatches | 0 |

The finite-range fixtures cover all 25599 admitted distinct positive half row
maxima and 513 padding rows at the largest range. The genuine fixture contributes
1536 admitted rows. These fixtures do not exhaust arbitrary row combinations.

## Rejection controls and execution evidence

Verification-only mutants operate outside the production package. All six
changed saved outputs on admitted inputs.

| Mutant | Changed int8 elements | Changed inverse-scale bits | Rows rejected after mutation |
| --- | --- | --- | --- |
| Reciprocal direction | 0 | 33455 | 0 |
| Float32 product without half rounding | 44440 | 0 | 0 |
| Truncation instead of ties to even | 1215626 | 0 | 0 |
| Clipping to 128 | 14470274 | 0 | 0 |
| Missing tile-axis transpose | 2867437 | 0 | 0 |
| Early input half-bit rounding | 485026 | 28465 | 2088 |

The early-rounding mutant applies `(bits+8)&0xfff0` before quantization. Some
admitted rows then become nonfinite or acquire an overflowing scale. Those
rejections are recorded separately and do not count as output mismatches.

The outer `attempt-s02-host-20261024-003/receipt.json` records exact arguments,
clocks, exits, logs, the Go executable hash, and the consumer binary hash.
`dependency-source-sha256.json` retains 880 Go, assembly, header, C, embedded
source, and module pins selected by the actual dependency graph. The receipt
also pins the new model test file, for 881 source pins in total. No source
changed between the source inventory and completed checks.

Each command ran in a recorded user systemd service with effective
`MemoryMax=3221225472`, `CPUQuotaPerSecUSec=2s`, `TasksMax=64`, and
`RuntimeMaxUSec=2min`. No host lock or native command was used.

The differential and consumer build exited 0. A changed accepted import was
rejected with exit 1, and an existing output directory was rejected with exit 1
without changing the saved successful manifest. The three S02 tests ran and
passed with zero skips under
`artifacts/koiquant/attempt-s02-checks-20261024-001` using:

```sh
go test -count=1 -json ./internal/model \
  -run '^(TestSUPQKVRowArithmetic|TestSUPQKVRowDomain|TestSUPQKVTile)$'
```

The earlier broad test expression also selected an existing environment-gated
native test, which skipped. It is not S02 acceptance evidence. The exact test
receipt above excludes that unrelated test. The failed wrapper attempt
`attempt-s02-host-20261024-001` remains preserved. Its source inventory command
failed because the transient service initially lacked the repository working
directory. No arithmetic ran in that attempt.

The independent review in `artifacts/parity-resume-review-20261024/s02-review.md` accepts the bounded S02 implementation and current-source evidence. Historical attempt `002` remains preserved; attempt `003` includes the final model observation interface in its source closure. Full
`go test ./...`, SUP graph integration, and general or end-to-end parity remain
separate coordinator gates. No performance claim is made.
