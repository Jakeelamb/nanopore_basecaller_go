PASS for the bounded caller MUX integration delta over `9cd1030ef57cb175fea77b266b93f7542a269f65`. No functional finding in the seven assigned files. This verdict identifies the reviewed working tree by `scoped-files.sha256`, with identical before/after hashes. `reviewed-artifact/` preserves every assigned final file and the tracked diff. The untracked align, ends, and splitter work is excluded.

Dorado source HEAD is `8b8fc5d36a9c0baab262a743cb175b5878e38ca6`, with a clean source checkout. The local toolchain is `go1.27.3-X:nodwarf5 linux/amd64`, cgo enabled. No model execution, Dorado process, download, GPU activity, source edit, maintained test edit, commit, or child delegation occurred.

- `internal/caller/caller.go:246-256` stitches before MUX and finishes the capture after MUX. `internal/caller/capture.go:396-418` reconstructs the same boundary before the complete endpoint comparison. Thus chunk captures remain pre-MUX and the final capture is post-MUX.
- `internal/caller/mux.go:16` uses numeric enum values 1 and 2. `internal/pod5/loader.c:156` resolves the POD5 enum, and `internal/pod5/native_cgo.go:95` stores it in `Read.EndReason`. The header assigns 1 to mux change, 2 to unblock mux change, and 3 to data-service unblock mux change. This matches the exact two-value source condition in `DataLoader.cpp:235-237`.
- `BasecallerNode.cpp:253,270` stitches and invokes MUX unconditionally before the downstream nodes, including with no-trim. The Go delta has the same position and no optional trimming gate.
- `NumSamples` preserves original POD5 count and `Trimmed` preserves the scaler prefix count. `TailTrimmedSamples` is normalized input length minus the returned current signal length. No identity, clock, or mean-Q-start field is reassigned. Normalized signal and chunk payloads retain the original capture content.
- Native MUX resets current MinKNOW events and pre-trim sequence length and shortens current signal. `CalledRead` does not expose those current internal fields or raw signal. Capture input intentionally keeps the POD5 originals. The admitted FASTQ tag selection in `hts_file.cpp:29-30` omits ns, du, ts, and me. `WriteFASTQ` computes qs from the final qstring with the unchanged mean-Q-start offset. These native internal-field differences do not create a supported FASTQ discrepancy. A future BAM, event, or signal consumer needs its own contract.

Fresh rerun evidence has exact commands, UTC start/finish clocks, exit status, raw stdout/stderr, and log hashes in adjacent files named by each check.

| Check | Observed result | Evidence |
| --- | --- | --- |
| Focused caller tests | Exit 0. MUX end reasons/tail accounting, invalid stride/map rejection, capture tamper and shape/path/count/model rejection, metadata compatibility, and FASTQ validation pass. Genuine endpoint explicitly skipped because this invocation has no environment selection. | `focused-caller.stdout` |
| Supplemental positive capture and legacy replay | Exit 0. A source-config-pinned synthetic capture contains 100 actual Go decoder/stitch outputs. Seventy high-Q chunks and thirty low-Q chunks produce 100 pre-MUX bases/moves. Final output has 69 bases, 69 moves, 414 current normalized samples, tail count 186, original count 610, and prefix count 10. All 507 capture files are unchanged by successful replay. The exact FASTQ header, sequence, and quality bytes pass. Zero tail is omitted and old non-MUX schema-1 output replays unchanged. | `overlay-capture.stdout`, `review_mux_evidence_test.go`, `overlay.json`, `synthetic-positive/` |
| Semantic negative controls | Each rehash-valid missing-tail field, wrong tail count, and pre-MUX expected output is rejected with `replayed endpoint differs from capture`. These controls pass the payload hash boundary and exercise endpoint comparison. | `overlay-capture.stdout` |
| No-cgo MUX caller tests | Exit 0. Both MUX integration tests run and pass. | `no-cgo-caller.stdout` |
| Fresh race-enabled HAC replay and helper tests | Exit 0. The actual retained capture replays and formats byte-for-byte as the unchanged stock raw FASTQ, 1,653 bases. MUX helper positive and invalid-map checks pass. | `fresh-hac-replay.stdout` |
| Fresh race-enabled SUP replay and helper tests | Exit 0. The actual retained capture replays and formats byte-for-byte as the unchanged stock raw FASTQ, 1,650 bases. MUX helper positive and invalid-map checks pass. | `fresh-sup-replay.stdout` |
| Actual retained artifact audit | Exit 0. Parent receipt start/end/exit files agree and ordered clocks parse. All parent stdout/stderr hashes verify. Both genuine captures have enum 4 signal_positive, original 19,856 samples, normalized 19,846, prefix 10, and omitted tail field. The declared run/input/output/tensor payloads and both signals hash correctly. Stock raw FASTQ SHA256 values match the retained controls. | `actual-retained-artifact-audit.stdout`, `retained-receipt-hashes.stdout`, `audit-receipts.go` |
| CI-shaped native evidence audit | The parent invocation exited 0 after 892 ordinary synthetic and three binary32 cases. Four source alterations were rejected. I checked actual native inputs/output/stderr hashes against native receipts and used cmp on native versus Go outputs for all four batches. The trailing no-tests-to-run warning comes after the parent test body executes and its sole optional retained subtest is excluded. No retained biological identity is claimed by that invocation. | `actual-retained-artifact-audit.stdout`, `retained-receipt-hashes.stdout`, repository `artifacts/caller/mux-ci-synthetic/`, `artifacts/readtrim/attempt-20261023T041413Z-663450456/` |
| Scoped diff and final identity | Exit 0. Diff check passes. All seven before/after hashes match. | `diff-and-source-check.stdout`, `freeze-reviewed-artifact.stdout`, `initial-pins.stdout`, `final-pins.stdout` |

The supplemental fixture uses constructed score/guide/posterior arrays and does not claim model inference or biological meaning. Its numeric end reason intentionally conflicts with its string name. Its FASTQ metadata includes a nonzero start sample and acquisition/protocol timestamps. The expected header independently fixes the final timestamp and channel/run/model/flow tags. The final quality is Q50 after removal of the low-Q tail.

The older parent generic receipts retain actual commands, clocks, exits, and output hashes, but do not themselves pin production source files. They are therefore supporting retained execution evidence, not standalone revision-keyed proof. The fresh focused, positive capture, no-cgo, HAC, and SUP runs are bound to the identical reviewed before/after working-tree hashes. The native readtrim implementation was already committed at the supplied base and independently reviewed; this review did not repeat its entire native fixture gate.

One nonblocking documentation follow-up is outside the assigned delta. `docs/readtrim.md:5` still says caller integration is pending. The parent was told to replace that statement and crosslink the now-integrated caller after acceptance.

Unproven contracts remain positive biological MUX endpoints, a live inference-producing Caller.Call positive MUX capture, hosted GitHub Actions on the configured Go 1.24 toolchain, official binary move-table output, general ATen behavior, GPU behavior, all-input processing, biological accuracy, performance, BAM/sample/event metadata parity, adapter or primer trimming, and splitting. The source call position of Caller.Call was inspected; only replay and the pure helper were executed for positive MUX behavior. Both genuine HAC/SUP controls are non-MUX.

No required tool or fixture was unavailable. Initial investigative lookups used obsolete source directories, a nonexistent Python glob, and summary.json rather than comparison.json; those read-only lookups failed and were corrected using the actual file inventory. They did not execute acceptance checks. No failed acceptance execution was discarded.

The Prove It Works principle changed the review by adding a positive capture-to-FASTQ replay and semantic rejection controls instead of accepting passing helper tests or parent summaries alone. The no-child and no-source-write scope overrides delegated skill steps.

Exact fresh commands follow. Each label also has `.started`, `.finished`, `.exit`, `.stdout`, `.stderr`, and `.logs.sha256` receipts. The command recorder is `run-command.sh`.

`focused-caller`

```sh
go test -count=1 -v -run \^\(TestMux\|TestReplay\|TestCapture\|TestFASTQ\) ./internal/caller 
```

`overlay-capture`

```sh
go test -overlay /tmp/nanopore-go-mux-integration-review/overlay.json -count=1 -v -run \^TestReviewMux ./internal/caller 
```

`no-cgo-caller`

```sh
env CGO_ENABLED=0 go test -count=1 -v -run \^TestMux ./internal/caller 
```

`fresh-hac-replay`

```sh
env CALLER_CAPTURE_MANIFEST=/home/jake/Projects/nanopore_basecaller_go/artifacts/caller/attempt-20261023T023928Z-Wr38Jl/capture/bb76c31a-36f0-48a0-abd2-107a10e1ade0/manifest.json CALLER_STOCK_FASTQ=/home/jake/Projects/nanopore_basecaller_go/artifacts/native-cpu-control/stock-hac-dev1-001/stdout go test -count=1 -race -v -run \^\(TestMux.\*\|TestCapturedNativeEndpoint\)\$ ./internal/caller 
```

`fresh-sup-replay`

```sh
env CALLER_CAPTURE_MANIFEST=/home/jake/Projects/nanopore_basecaller_go/artifacts/caller/attempt-20261023T033758Z-xTMh1T/capture/bb76c31a-36f0-48a0-abd2-107a10e1ade0/manifest.json CALLER_STOCK_FASTQ=/home/jake/Projects/nanopore_basecaller_go/artifacts/native-cpu-control/stock-sup-dev1-001/stdout go test -count=1 -race -v -run \^\(TestMux.\*\|TestCapturedNativeEndpoint\)\$ ./internal/caller 
```

`actual-retained-artifact-audit`

```sh
go run /tmp/nanopore-go-mux-integration-review/audit-receipts.go 
```

`retained-receipt-hashes`

```sh
bash -c for\ check\ in\ mux-ci-synthetic\ mux-integration-focused\ mux-nocgo\ mux-hac-retained\ mux-sup-retained\;\ do\ printf\ \"%s\\n\"\ \"\$check\"\;\ sha256sum\ -c\ \"artifacts/caller/\$check/output-sha256.txt\"\ \|\|\ exit\;\ test\ \"\$\(cat\ \"artifacts/caller/\$check/exit.txt\"\)\"\ =\ 0\ \|\|\ exit\;\ cat\ \"artifacts/caller/\$check/receipt.json\"\;\ done\;\ for\ dir\ in\ artifacts/readtrim/attempt-20261023T041413Z-663450456\ artifacts/readtrim/attempt-20261023T041413Z-663450456/binary32-\*\;\ do\ cmp\ \"\$dir/go-output.bin\"\ \"\$dir/native.stdout\"\ \|\|\ exit\;\ cat\ \"\$dir/comparison.json\"\;\ done 
```

`source-and-wiring-inspection`

```sh
bash -c nl\ -ba\ external/dorado-source/dorado/read_pipeline/nodes/BasecallerNode.cpp\ \|\ sed\ -n\ \"245\,278p\"\;\ nl\ -ba\ external/dorado-source/dorado/data_loader/DataLoader.cpp\ \|\ sed\ -n\ \"208\,244p\"\;\ nl\ -ba\ external/dorado-source/dorado/read_pipeline/base/read_utils.cpp\ \|\ sed\ -n\ \"110\,166p\"\;\ nl\ -ba\ internal/caller/mux.go\;\ nl\ -ba\ internal/caller/caller.go\ \|\ sed\ -n\ \"240\,264p\"\;\ nl\ -ba\ internal/caller/capture.go\ \|\ sed\ -n\ \"395\,425p\"\;\ nl\ -ba\ internal/caller/fastq.go\ \|\ sed\ -n\ \"77\,105p\" 
```

`diff-and-source-check`

```sh
bash -c git\ diff\ --check\ --\ .github/workflows/cpu-parity.yml\ README.md\ docs/caller.md\ internal/caller/caller.go\ internal/caller/capture.go\;\ git\ -C\ external/dorado-source\ status\ --porcelain\ --untracked-files=all\;\ sha256sum\ external/dorado-source/dorado/read_pipeline/nodes/BasecallerNode.cpp\ external/dorado-source/dorado/data_loader/DataLoader.cpp\ external/dorado-source/dorado/read_pipeline/base/read_utils.cpp\ external/dorado-source/dorado/read_pipeline/base/messages.cpp\ internal/pod5/pod5_format/c_api.h\ internal/pod5/loader.c\ internal/pod5/native_cgo.go 
```

`initial-pins`

```sh
bash -c git\ rev-parse\ HEAD\;\ git\ status\ --short\;\ sha256sum\ .github/workflows/cpu-parity.yml\ README.md\ docs/caller.md\ internal/caller/caller.go\ internal/caller/capture.go\ internal/caller/mux.go\ internal/caller/mux_test.go\;\ git\ -C\ external/dorado-source\ rev-parse\ HEAD\;\ go\ version\;\ go\ env\ CGO_ENABLED\ GOOS\ GOARCH 
```

`final-pins`

```sh
bash -c git\ rev-parse\ HEAD\;\ git\ status\ --short\;\ sha256sum\ .github/workflows/cpu-parity.yml\ README.md\ docs/caller.md\ internal/caller/caller.go\ internal/caller/capture.go\ internal/caller/mux.go\ internal/caller/mux_test.go\;\ git\ -C\ external/dorado-source\ rev-parse\ HEAD\;\ go\ version\;\ go\ env\ CGO_ENABLED\ GOOS\ GOARCH 
```

`freeze-reviewed-artifact`

```sh
/tmp/nanopore-go-mux-integration-review/freeze-reviewed-artifact.sh 
```
