# SUP QKV int8 arithmetic capture

`verification/koiquant` prepares, captures, imports, and compares the native QKV int8 cache boundary. Native execution is currently held for the reference and retrieval priority pass. The new C++ adapter remains uncompiled. No CPU or CUDA arithmetic capture exists for this unit.

This unit has no production Go quantizer. Its replay checks retained raw observations and metadata on the host. Passing replay does not establish Go arithmetic parity, Koi kernel use, biological accuracy, or performance.

The input is the saved unrounded half buffer `[3,8,64,512]`. The intended final outputs are int8 `[3,8,4,32,16,16]` and float32 reciprocal scales `[3,8,64]`. Expected stage dtypes remain source hypotheses until native execution.

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

## Native admission and held commands

Do not run these commands while the priority hold remains active. The coordinator must first release the bounded native invocation after current process and resource intake. A free lock or an admission flag alone does not lift that hold.

Capture requires the exact accepted basic runtime receipt and its matching CUDA13 build. Their SHA256 values are `15ca5c4a0cda12af18a430cf7763efa2bf78590636aa501892b9106d53c9beae` and `f41faaed7ba863f3d16c5a5f08cb373bf588794ec73fe25d1620359863b9d0dd`. The backend pin is `d70b612b442e8449b2357d12284705bd82b30cdfa7316358a29ad1c800c9e3d7`. Admission trusts this accepted receipt, rather than repeating historical runtime tests.

Before every compiler, producer, or GPU query, the command requires explicit admission, exact qualification, and the existing stock lock. It opens `/home/jake/Projects/nanopore_basecaller/results/stock/host.lock` without creation or truncation and takes `LOCK_EX | LOCK_NB`. It holds the file through compilation, diagnostics, capture, and raw receipt storage. Missing admission, invalid qualification, or a held lock launch zero external commands.

The planned sequence compiles one executable during CPU capture. CUDA reuses that exact binary, dependency closure, and source pins. Each declared fixture runs once per runtime. The build uses the accepted Torch CUDA13 archive and `/opt/cuda`, whose actual toolkit version and headers are recorded separately. This is not stock CUDA12 qualification.

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

The Model the Domain principle keeps fixtures, stages, storage widths, and receipt identities in closed tables and types. The source checkpoint is preparation, gated execution code, retained-output import, metadata rejection, and pure Go checks. Native compilation, real handoff, CPU arithmetic, CUDA arithmetic, finite-range observations, and the production arithmetic decision remain open.
