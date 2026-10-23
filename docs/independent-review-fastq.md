PASS for the bounded FASTQ formatter and retained HAC replay delta. No actionable correctness finding in the assigned paths. This verdict does not qualify SUP or its tensor backend.

The reviewed checkout is `/home/jake/Projects/nanopore_basecaller_go` at HEAD `086f2361b7b4900ddcf0ceb1cdbfad1cf7a9c1e2`, with working-tree FASTQ changes. The comparison base is `7a0f19352f6f9bdb62036718e2710cb53127be6a`. There is no committed difference between those revisions in the scoped paths. HEAD alone does not identify the reviewed implementation. `current-scope-manifest/stdout` identifies the reviewed files, including untracked source. `scoped-source.tar.gz` preserves those bytes. `artifact-identity/stdout` identifies the independently built executable, SHA256 `b83cbc7ad933cc234f4d1a8521bf012bf3b78c7443e4548438b6386ce85205b8`.

During review, another owner changed the example tensor-library path in `docs/caller.md:103` from `attempt-004` to `attempt-005-sup-abi3`. No scoped implementation or acceptance source changed. Reversing that one documentation substitution recreates the original reviewed document hash exactly. See `docs-drift-reconstruction/stdout`. The replacement path exists, but this review did not execute inference with that library. `review-final-identity` verifies the refreshed final source manifest.

Observed behavior and evidence are as follows.

| Check | Observed result | Evidence under this directory |
| --- | --- | --- |
| Race-enabled caller, decoder, CLI, and caller verification packages | Exit 0. Every test passed. No skips in the cgo suite. | `race/command`, `race/stdout`, `race/stderr`, `race/exit` |
| Extracted native metadata oracle | 153 records. Raw FASTQ and float32 mean-quality bits exact. Native and Go FASTQ SHA256 `15c8d7adeabc3de21770a9637ef4815be5e18c6009415c6b6237dcf4571eef9c`. | `metadata-oracle/` |
| Existing decoder differential | 1465 cases. Sequence, quality, moves, states, final score bits, and quality-data bits exact. Zero differing floats. | `decode-oracle/` |
| Independent direct oracle comparisons | `cmp` of metadata FASTQ, `diff` of extracted mean-bit lines, and `cmp` of decoder output all exited 0. Independent native output also matches worker-retained native bytes. | `independent-raw-oracles/`, `metadata-native-evidence/` |
| Explicit no-cgo package tests | Exit 0, including `errors.Is(MeanQScore("I"), ErrCGORequired)`. | `nocgo/` |
| Built no-cgo CLI replay | Exit 1 with the explicit native float libm requirement. Zero stdout. | `build-nocgo/`, `nocgo-replay/` |
| Vet and cgo CLI build | Both exited 0. | `vet/`, `build/` |
| Real CLI replay and stock comparator | Both exited 0 for the authoritative capture. Exactly one 1653-base record. Entire raw FASTQ SHA256 `53ec2ca57acff8ce0041576a6bade2724a623110421867827032990d8df860cd`, matching unchanged stock bytes. | `replay/`, `compare/`, `raw-cmp/` |
| Changed channel negative case | Comparator exited 1. Canonical endpoint remained true, while raw FASTQ, header, and raw capture replay fields were false. This establishes that the raw check detects a header-only failure. | `changed-channel/`, `reject-changed-channel/` |
| Traced CLI replay | Exit 0 and traced output `cmp` exit 0. Exactly one `execve`, for the reviewed executable. The successful native library opens are libc and libm. No tensor, POD5, CUDA library, model-weight, or producer executable open. The loader made unsuccessful libm/libc search attempts under inherited CUDA search paths. | `replay.strace`, `trace-inspection/`, `trace-cmp/` |
| Clock boundary controls on copied genuine captures | Both endpoints, -9223372036854 and 9223372036854 ms, exit 0. Adjacent outside values each exit 1 with native clock range error and zero stdout. | `clock-*/`, `capture-controls/` |
| Malformed metadata on copied genuine captures | Newline in RunID or FlowCellID each exits 1 with invalid metadata text and zero stdout. | `newline-RunID/`, `newline-FlowCellID/` |
| Changed new output metadata with valid updated payload hash | Exit 1 with replayed endpoint differs from capture, zero stdout. | `changed-output-metadata/` |
| Original capture and stock immutability | Every file in the authoritative read capture and stock stdout has the same before and after SHA256. `cmp` exits 0. | `cached-pins-before/`, `cached-pins-after/`, `immutable-originals/` |

The exact positive package commands, run from the reviewed checkout, were:

```sh
env DORADO_SOURCE=/home/jake/Projects/nanopore_basecaller_go/external/dorado-source CALLER_METADATA_ARTIFACT_DIR=/tmp/nanopore-go-fastq-review/metadata-oracle DECODE_ARTIFACT_DIR=/tmp/nanopore-go-fastq-review/decode-oracle CALLER_CAPTURE_MANIFEST=/home/jake/Projects/nanopore_basecaller_go/artifacts/caller/attempt-20261023T023928Z-Wr38Jl/capture/bb76c31a-36f0-48a0-abd2-107a10e1ade0/manifest.json CALLER_STOCK_FASTQ=/home/jake/Projects/nanopore_basecaller_go/artifacts/native-cpu-control/stock-hac-dev1-001/stdout go test -count=1 -race -v ./internal/caller ./internal/decode ./cmd/basecaller ./verification/caller/...
env CGO_ENABLED=0 go test -count=1 -v ./internal/caller ./internal/decode ./cmd/basecaller ./verification/caller/...
go vet ./internal/caller ./internal/decode ./cmd/basecaller ./verification/caller/...
go build -o /tmp/nanopore-go-fastq-review/basecaller ./cmd/basecaller
/tmp/nanopore-go-fastq-review/basecaller replay --manifest /home/jake/Projects/nanopore_basecaller_go/artifacts/caller/attempt-20261023T023928Z-Wr38Jl/capture/bb76c31a-36f0-48a0-abd2-107a10e1ade0/manifest.json
cmp /tmp/nanopore-go-fastq-review/replay/stdout /home/jake/Projects/nanopore_basecaller_go/artifacts/native-cpu-control/stock-hac-dev1-001/stdout
go run ./verification/caller/cmd --go-fastq /tmp/nanopore-go-fastq-review/replay/stdout --stock-fastq /home/jake/Projects/nanopore_basecaller_go/artifacts/native-cpu-control/stock-hac-dev1-001/stdout --manifest /home/jake/Projects/nanopore_basecaller_go/artifacts/caller/attempt-20261023T023928Z-Wr38Jl/capture/bb76c31a-36f0-48a0-abd2-107a10e1ade0/manifest.json
strace -f -e trace=execve,openat -o /tmp/nanopore-go-fastq-review/replay.strace /tmp/nanopore-go-fastq-review/basecaller replay --manifest /home/jake/Projects/nanopore_basecaller_go/artifacts/caller/attempt-20261023T023928Z-Wr38Jl/capture/bb76c31a-36f0-48a0-abd2-107a10e1ade0/manifest.json
```

The exact header negative commands were:

```sh
awk 'NR == 1 { sub(/\tch:i:[0-9]+/, "\tch:i:1") } { print }' /tmp/nanopore-go-fastq-review/replay/stdout
go run ./verification/caller/cmd --go-fastq /tmp/nanopore-go-fastq-review/changed-channel/stdout --stock-fastq /home/jake/Projects/nanopore_basecaller_go/artifacts/native-cpu-control/stock-hac-dev1-001/stdout --manifest /home/jake/Projects/nanopore_basecaller_go/artifacts/caller/attempt-20261023T023928Z-Wr38Jl/capture/bb76c31a-36f0-48a0-abd2-107a10e1ade0/manifest.json
```

`run.sh` is only an evidence recorder. Each command directory preserves the actual command, cwd, start and finish clocks, stdout, stderr, and exit status. `capture-controls.sh` preserves exact copied-fixture mutations and command receipts. No implementation or maintained acceptance check was edited. The maintained `verification/caller/verify.sh` was inspected, and its relevant commands were rerun individually so all new evidence stays in the assigned `/tmp` directory. No GPU, Dorado producer, Go POD5 inference, or download ran.

Source tracing confirms the following.

- `internal/caller/fastq.go:29` derives acquisition clock plus integer-truncated sample delta and checks multiplication and addition overflow. `fastq.go:40` admits only the pinned int64 nanosecond clock range. `fastq.go:51` validates the complete record before calling the writer.
- `fastq.go:81` uses the quality suffix only when length is greater than MeanQStart. Native `messages.cpp:419` has the same boundary. Native `config/BasecallModelConfig.cpp:23` defaults missing model-specific mean-quality start to 60, and its config parser uses that default when the qscore field is absent. The pinned HAC config has no override. The current Go HAC spec has MeanQStart 60.
- `internal/decode/mean_quality.go:9` uses native `powf`, float32 lookup values, float32 accumulation and division, native `log10f`, and a 1-to-50 clamp. `internal/decode/math_native.go:12` and `:13` provide the native float operations. This matches the extracted source function rather than a float64 calculation followed by a cast.
- `fastq.go:88` formats the float with six significant digits. The fixed score range remains in ordinary decimal notation. Tags preserve selected BAM insertion order, qs, ch, st, optional RG, optional PU, DT, tm. RunID is the protocol run ID, not acquisition ID. The empty model fallback is unknown, empty run and flow-cell omit their optional tags, and disabled trim emits none.
- Native `ReadToBamTypeNode` carries protocol clock and flow-cell metadata to HtsData. `WriterNode` dispatches to HtsFileWriter, whose process calls prepare_item before the stream or structured writer. The native selected-tag list filters BAM tags while HTSlib emits insertion order. The native system clock conversion is compiled unchanged in the oracle.
- `internal/caller/capture.go:396` derives replay metadata from pinned input and model data. Lines 405 through 410 derive absent legacy metadata without rewriting files and compare new metadata exactly. Both CLI producer output and CLI replay call `WriteFASTQ` through `cmd/basecaller/main.go:35`.

`source-trace/stdout` retains source line excerpts and hashes. The native source checkout is exactly `8b8fc5d36a9c0baab262a743cb175b5878e38ca6` with no working-tree changes. The oracle's VerifySources also passed its mutation rejection control. I inspected raw worker receipts at `artifacts/caller/check-20261023T030726Z-Eki21a`, then independently reran their relevant checks. I did not rely on a worker summary.

The earlier failed year-9999 attempt is preserved and verified in `retained-first-failure/`. Its native process exited 0 while its raw comparison and Go race command failed. Go rendered year 9999, while the native intermediate clock wrapped to 1816. The current range rejects that unsupported input. Valid endpoint cases match the extracted native formatter, and adjacent rejection is separately checked above.

Uncovered limits remain.

- The 153 metadata cases are finite fixtures, not an exhaustive numerical proof. The oracle uses shimmed data containers and system HTSlib 1.24. It does not establish the stock executable's linked HTSlib identity or portability across a different libc, libm, compiler, or operating system.
- The stock raw byte check qualifies one retained development HAC read. There is no stock move-table output to compare. No claim follows about all reads, biological accuracy, throughput, GPU execution, SUP, adapter trimming, read splitting, barcodes, RNA, or BAM output.
- Replay validates payload integrity and derived new metadata. The manifest format does not authenticate an adversarial manifest. Absence of FASTQ metadata remains a deliberate schema-1 compatibility case.
- The latest producer path was inspected, not rerun. Retained replay demonstrates the changed formatter without fresh tensor or POD5 inference. Other owners' model and tensor changes remain outside this verdict.
- The review applies to the frozen file hashes and reviewed executable. A later integrated commit can inherit this evidence only if these scoped bytes match, with a separate review of any dependency or documentation delta.

All required local verification tools worked. The full repository suite was outside this bounded review and was not rerun. Upstream delegated how and Comment Sicko workflows were not run because this assignment explicitly forbids child agents. Prove It Works shaped the choice to rebuild, replay, inspect raw bytes, and run failing controls. No source, acceptance check, stock input, or authoritative capture was changed by this reviewer.
