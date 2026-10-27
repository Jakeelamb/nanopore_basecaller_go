# SUP QKV int8 arithmetic capture

`verification/koiquant` prepares, captures, imports, and compares the native QKV int8 cache boundary. The retained CPU producer completed on 2026-10-23. CUDA capture completed on 2026-10-25 using that exact executable. Both 31-fixture captures pass host import and replay, including genuine metadata and corruption rejection. Independent review remains pending.

This unit has no production Go quantizer. Its replay checks retained raw observations and metadata on the host. Passing replay does not establish Go arithmetic parity, Koi kernel use, biological accuracy, or performance.

The input is the saved unrounded half buffer `[3,8,64,512]`. Both captures verify final int8 `[3,8,4,32,16,16]` and float32 reciprocal scales `[3,8,64]` storage. This observes source arithmetic, not a production Go implementation.

## Retained CPU evidence

The raw producer manifest is `artifacts/koiquant/attempt-native-cpu-001/raw-manifest.json`, SHA256 `61a028d0837e966019054291427c39a424a0e512b4566475e175f7b9dd8f1f09`. Host import revalidated its source, runtime, command and payload pins without repeating the producer. The exhaustive table covers 31,743 unique finite positive half ranges, with 25,599 finite scales and 6,144 overflowing scales. The Go diagnostic half-scale and reciprocal checks have zero mismatches on this CPU capture. These diagnostics do not qualify a production quantizer or CUDA arithmetic.

| Attempt | Result |
| --- | --- |
| `artifacts/resume-koiquant-import-20261024-001` | Rejected output-directory name before import; retained |
| `artifacts/resume-koiquant-import-20261024-002` | All 31 observations imported and validated |
| `artifacts/resume-koiquant-replay-20261024-001` | Genuine bridge passed; corruption test failed on a cross-filesystem hard link |
| `artifacts/resume-koiquant-replay-20261024-002` | Genuine bridge and corruption rejection passed after copying test payloads |

The replay receipt is `artifacts/koiquant/attempt-resume-replay-cpu-20261024-002/replay-manifest.json`. Both required tests passed with no skips. The corruption test now checks that every mutation changes the metadata and selects the observed device string for both CPU and CUDA captures. The saved producer and failed replay remain unchanged.

## CUDA comparison

`artifacts/koiquant/attempt-native-cuda-resume-20261024-001/raw-manifest.json` retains 470 output pins and six successful child commands. The producer reused the CPU executable with SHA256 `4a5c636212df97fa924d878eaa3aac0621335223d0cd60700916789b5e006f9c`; it did not recompile or run a model. The canonical host lock covers the invocation under the existing program-wide grant in `provenance/gpu-admission-grant.json`.

Only the host replay test had changed since preparation. The recorded invocation temporarily restored that test's saved producer version to preserve the exact producer source closure, then restored the corrected consumer test. Both source hashes match before and after. The script and hashes are retained in `artifacts/koiquant/resume-cuda-command-20261024/`.

The CUDA replay receipt is `artifacts/koiquant/attempt-resume-replay-cuda-20261024-001/replay-manifest.json`. The comparison receipt is `artifacts/koiquant/attempt-resume-comparison-20261024-001/manifest.json`, covering 434 stage pairs. The genuine buffer and all exhaustive finite-range fixtures match at every stage. Both runtimes classify 25,599 finite scales and 6,144 overflowing scales across the 31,743 positive finite half ranges, with zero Go diagnostic scale or reciprocal mismatches. All 1,536 genuine rows have finite positive scales and reciprocal values.

Four fixtures differ in NaN representations at intermediate boundaries: `all-half-patterns`, `infinities`, `nans`, and `signed-zero`. These differences remain recorded. Int8 tile bytes match on every fixture, while nonfinite reciprocal bits differ on the NaN fixtures. Zero ranges, nonfinite inputs and overflowing half scales remain outside the proposed finite-scale production domain. A future Go quantizer must reject those rows explicitly and pass its own differential against these saved CUDA stages.

## Prepare source and retained inputs

Run from the canonical Go module root. Use fresh attempt names.

```sh
CGO_ENABLED=0 go test -count=1 ./verification/koiquant/...
go vet ./verification/koiquant/...
CGO_ENABLED=0 go build -o /tmp/koiquant ./verification/koiquant/cmd

/tmp/koiquant -device cpu \
	-input-capture "$PWD/artifacts/koiweights/attempt-qkv-f16-002/manifest.json" \
	-out "$PWD/artifacts/koiquant/attempt-prepare-cpu-NEW"

/tmp/koiquant -device cuda \
	-input-capture "$PWD/artifacts/koiweights/attempt-qkv-f16-002/manifest.json" \
	-out "$PWD/artifacts/koiquant/attempt-prepare-cuda-NEW"

/tmp/koiquant -check "$PWD/artifacts/koiquant/attempt-prepare-cpu-NEW/manifest.json"
```

Preparation and checking launch no external command. They reuse unchanged `koiweights.ValidateCapture` and `ReadHalf`. The original capture manifest must hash to `3faad1885b81146b604eea6e2028767ba204a6f62b71a782add4a268d7b5aae1`. Its genuine unrounded input must contain 786432 half elements and hash to `cb27b916510b9bf98576a247db446d842ef7a199c74b54da32bae031258879f0`. Legacy validation reads retained archive pins. It never deserializes an archive or runs a model.

Each preparation records its command, clocks, grounding documents, current source closure, extracted blocks, generated source, fixture list, inputs, and expected stages. Source changes require a fresh preparation. Existing attempts and symlink outputs are rejected. Historical ten-fixture preparations remain preserved.

The grounding dependencies remain the durable design at `artifacts/koiweights/int8-design/design.md` and the report and source list under `/tmp/nanopore-go-sup-int8-contract`. Their fixed hashes remain enforced by preparation.

## Native source and fixtures

The adapter extracts the unchanged `ScaledTensor` declaration, `quantize_tensor`, and QKV quantize, reciprocal, and tile block. It reuses the existing raw half storage reader and writer. It contains no model graph, archive loader, or Koi kernel call.

An untouched helper and cache block run on a fresh input copy. A companion records each materialization on a separate copy. Raw transfer bytes and both final outputs must match exactly. Native exceptions remain per-fixture observations. Partial outputs and exception text are retained. Missing outputs are not fabricated.

| Stages | Expected storage |
| --- | --- |
| `input`, `transfer`, `abs`, `hproduct`, `hround`, `hclip` | Half `[3,8,64,512]` |
| `amax`, `hscale` | Half `[3,8,64]` |
| `char` | Int8 `[3,8,64,512]` |
| `fscale`, `freciprocal`, `untouched-freciprocal` | Float32 `[3,8,64]` |
| `i8tile`, `untouched-i8tile` | Int8 `[3,8,4,32,16,16]` |

The fixture list contains 31 full-size inputs. The original ten cover the genuine buffer, axis variation, round and clip boundaries, neighboring scales, finite extremes, all half patterns, signed zero, tiny ranges, infinities, and NaNs. Twenty-one additional fixtures cover every positive finite nonzero half row range, from bits `0001` through `7bff`. Their 32256 rows contain 31743 distinct ranges and 513 padding rows at the largest finite range. Selection depends only on the range index. Genuine model values do not select the numeric domain.

Classification reports finite, zero-range, subnormal, and nonfinite input rows. It separately records native scale infinities, finite-input overflow, NaNs, zero scales, reciprocal classes, and potential domain admission. Differences in native reduction, widening, Go diagnostic half division, and Go diagnostic reciprocal bits are retained. Go diagnostics reuse `internal/signal/half.go`. They are not a production implementation or an assumed CUDA contract. The exhaustive table indexes the original input range even if a native reduction differs.

## Native admission and reproduction

The historical reference/retrieval hold preceded the program-wide grant recorded in `provenance/gpu-admission-grant.json`. Every new invocation still needs current process/resource intake and canonical lock ownership. A free lock or an admission flag alone does not override another owner's active reservation.

Capture requires the exact accepted basic runtime receipt and its matching CUDA13 build. Their SHA256 values are `15ca5c4a0cda12af18a430cf7763efa2bf78590636aa501892b9106d53c9beae` and `f41faaed7ba863f3d16c5a5f08cb373bf588794ec73fe25d1620359863b9d0dd`. The backend pin is `d70b612b442e8449b2357d12284705bd82b30cdfa7316358a29ad1c800c9e3d7`. Admission trusts this accepted receipt, rather than repeating historical runtime tests.

Before every compiler, producer, or GPU query, the command requires explicit admission, exact qualification, and the existing stock lock. It opens `/home/jake/Projects/nanopore_basecaller/results/stock/host.lock` without creation or truncation and takes `LOCK_EX | LOCK_NB`. It holds the file through compilation, diagnostics, capture, and raw receipt storage. Missing admission, invalid qualification, or a held lock launch zero external commands.

The completed sequence compiled one executable during CPU capture. CUDA reused that exact binary, dependency closure, and source pins. Each declared fixture ran once per runtime. The build uses the accepted Torch CUDA13 archive and `/opt/cuda`, whose actual toolkit version and headers are recorded separately. This is not stock CUDA12 qualification. Reuse the retained captures above for downstream work.

```sh
/tmp/koiquant \
	-capture "$PWD/artifacts/koiquant/attempt-prepare-cpu-NEW/manifest.json" \
	-basic-runtime "$PWD/artifacts/tensor/replay-abi4-cuda-001/manifest.json" \
	-build-manifest "$PWD/artifacts/tensor/attempt-abi4-cuda-004/manifest.json" \
	-runtime-admitted \
	-out "$PWD/artifacts/koiquant/attempt-native-cpu-NEW"

/tmp/koiquant \
	-replay "$PWD/artifacts/koiquant/attempt-native-cpu-NEW/raw-manifest.json" \
	-out "$PWD/artifacts/koiquant/attempt-replay-cpu-NEW"

/tmp/koiquant \
	-capture "$PWD/artifacts/koiquant/attempt-prepare-cuda-NEW/manifest.json" \
	-basic-runtime "$PWD/artifacts/tensor/replay-abi4-cuda-001/manifest.json" \
	-build-manifest "$PWD/artifacts/tensor/attempt-abi4-cuda-004/manifest.json" \
	-reuse-binary "$PWD/artifacts/koiquant/attempt-native-cpu-NEW/raw-manifest.json" \
	-ordinal 0 -runtime-admitted \
	-out "$PWD/artifacts/koiquant/attempt-native-cuda-NEW"

/tmp/koiquant \
	-replay "$PWD/artifacts/koiquant/attempt-native-cuda-NEW/raw-manifest.json" \
	-out "$PWD/artifacts/koiquant/attempt-replay-cuda-NEW"

/tmp/koiquant \
	-compare-cpu "$PWD/artifacts/koiquant/attempt-replay-cpu-NEW/manifest.json" \
	-compare-cuda "$PWD/artifacts/koiquant/attempt-replay-cuda-NEW/manifest.json" \
	-out "$PWD/artifacts/koiquant/attempt-compare-NEW"
```

The CPU replay must import the genuine fixture and reject copied corrupt metadata before the CUDA invocation. If import fails, retain the raw output and repair the consumer. Use `-import RAW -out FRESH` or `-replay RAW -out FRESH` to reuse the original producer output. Do not repeat a producer to repair an observation or importer failure. Compilation failures and incomplete capture receipts remain failed attempts and require coordinator assessment.

## Receipt contracts

`raw-manifest.json` uses `nanopore-sup-qkv-i8/raw-v1`. It records these fields.

- Native status, clocks, invocation, planned and observed device, and ordinal.
- Exact preparation, basic runtime, build, archive, executable, compiler, and optional binary-origin pins.
- A frozen copy of every declared producer source, plus generated source and input pins.
- Compiler tools, depfile headers, archive member and symlink pins, linked libraries, and actual mapped files before and after execution. Mapping segments are deduplicated before hashing. Device mappings are preserved in raw maps and are not read as files. The dynamic loader must appear in the mapped closure.
- Effective CUDA and loader environment with explicit presence and value, including unset versus empty `CUDA_VISIBLE_DEVICES`, `CUDA_DEVICE_ORDER`, and `CUBLAS_WORKSPACE_CONFIG`.
- GPU UUID and driver bookends, native runtime and toolkit configuration, raw argv, exits, clocks, stdout, stderr, and hashes.
- Every raw observation file, including partial stages and exceptions.

The importer uses `nanopore-sup-qkv-i8/import-v1`. It requires exact field presence, including zero-valued fields. Duplicate, unknown, case-variant, and missing fields fail. It checks the closed fixture and stage schemas, actual devices, sizes, strides, payload hashes, transfer parity, untouched-output parity, source snapshots, resource closure, raw commands, and bookends. The imported manifest pins current consumer sources separately from frozen producer sources. A consumer fix can reimport the retained raw output without requiring the old producer source to match the current checkout.

Raw half, int8, and float32 bytes remain outside Git. CPU and CUDA observations stay separate. Comparison emits a mismatch count and the first differing element and raw bits for each fixture and stage. Nonfinite values use raw bit comparisons.

Replay builds a Go test executable with `CGO_ENABLED=0` and runs exactly `TestCaptureMetadataBridge` and `TestCapturedRawRejection`. Both must run and pass once, without skips. Replay records the binary and current source pins separately and initializes no GPU. Ordinary tests use explicitly synthetic storage metadata. An admitted replay supplies the retained raw capture, imports its genuine fixture, and corrupts a temporary metadata copy for rejection. Those tests validate capture handoff and rejection, not production Go arithmetic.

The Model the Domain principle keeps fixtures, stages, storage widths, and receipt identities in closed tables and types. The independent review at `artifacts/parity-resume-review-20261024/s01-review.md` accepts the bounded source observations and test correction. Its admitted production candidate domain requires finite row values and absolute row maximum half bits `0x1801..0x7bff`. Production Go arithmetic, domain rejection and its own differential remain open. Exhaustive row maxima do not exhaust arbitrary 512-value row combinations.
