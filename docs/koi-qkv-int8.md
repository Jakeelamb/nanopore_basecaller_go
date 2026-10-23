# SUP QKV int8 oracle preparation

`verification/koiquant` prepares source and raw half inputs for the next cache boundary. Native execution remains held. This package implements no production arithmetic, native runner, capture importer, or Go cache consumer.

The boundary consumes the saved unrounded `[3,8,64,512]` half buffer. It does not consume the rounded half tile. The intended native outputs are int8 `[3,8,4,32,16,16]` and float32 reciprocal scales `[3,8,64]`.

The pinned design is `artifacts/koiweights/int8-design/design.md`. Its SHA256 is `9115492cd9dd2a5c0b279c8c78ad95625c2bbc9b5a1e8aacd89a53e0040696b6`. Preparation also requires the pinned contract report and source list in `/tmp/nanopore-go-sup-int8-contract`. Those temporary files remain explicit dependencies until their owner supplies a durable copy.

## Prepare and check source and inputs

Run these commands from the Go module root with fresh attempt names.

```sh
CGO_ENABLED=0 go test -count=1 ./verification/koiquant/...

CGO_ENABLED=0 go run ./verification/koiquant/cmd \
	-input-capture "$PWD/artifacts/koiweights/attempt-qkv-f16-002/manifest.json" \
	-device cpu -out "$PWD/artifacts/koiquant/attempt-prepare-cpu-NEW"

CGO_ENABLED=0 go run ./verification/koiquant/cmd \
	-check "$PWD/artifacts/koiquant/attempt-prepare-cpu-NEW/manifest.json"

CGO_ENABLED=0 go run ./verification/koiquant/cmd \
	-input-capture "$PWD/artifacts/koiweights/attempt-qkv-f16-002/manifest.json" \
	-device cuda -out "$PWD/artifacts/koiquant/attempt-prepare-cuda-NEW"
```

`-device` labels the planned native tier. It initializes no device. The command has no native execution flag. `-check` reads and hashes existing files without running a compiler, producer, archive loader, GPU query, or stock lock operation.

Preparation requires the original manifest hash `3faad1885b81146b604eea6e2028767ba204a6f62b71a782add4a268d7b5aae1`. It calls unchanged `koiweights.ValidateCapture`, selects `genuine.Interleaved`, and calls unchanged `koiweights.ReadHalf`. The selected payload must have 786432 half elements and SHA256 `cb27b916510b9bf98576a247db446d842ef7a199c74b54da32bae031258879f0`. The tool pins the original manifest before and after use. Reading hashes of the retained archive is part of legacy validation. The tool never deserializes the archive or runs the neural network.

Each fresh attempt contains `manifest.json`, `oracle.cpp`, `fixture-list.txt`, and ten full-size input files. The manifest records the invocation, UTC start and finish clocks, current package source hashes, extracted text and whole source hashes, generated source hashes, planned device, and the explicit `prepared_native_held` status. Failed preparation after output creation preserves a `prepare_failed` receipt. Existing attempts and symlink outputs are rejected.

`-check` requires the exact ordered fixture set and regenerates deterministic fixture inputs in memory for comparison. Removing a source pin, grounding pin, fixture, or stage declaration fails validation. A changed producer source requires a new preparation attempt.

## Extracted oracle source

Extraction requires pinned whole source files before selecting these verbatim blocks.

| Block | Source range |
| --- | --- |
| `ScaledTensor` | `tensor_utils.h` lines 47 through 50 |
| Untouched `quantize_tensor` | `tensor_utils.cpp` lines 291 through 298 |
| Untouched QKV quantize, reciprocal, and tile | `TxModules.cpp` lines 455 through 461 |
| Raw half reader and storage writer | Existing `verification/koiweights/native/oracle.cpp` lines 13 through 27 |

The generated C++ adapter uses ATen operators on raw input storage. It has no archive loader, model graph, Koi descriptor, or caller. It first runs the untouched helper and cache block on a fresh copy. An instrumented companion runs on a separate copy and preserves the source operator order and separate materializations. Final int8 bytes and reciprocal float32 bytes must equal the untouched outputs. Any mismatch invalidates the instrumented observations.

The adapter saves the host input and the device-transferred input and compares their raw half bytes. CPU and CUDA invocations use distinct prepared directories. Each fixture runs inside its own exception boundary. Partial outputs and errors remain observations. The adapter continues to later fixtures and returns failure if any fixture failed.

`stages.tsv` records the observed dtype, device, shape, strides, and byte count for each materialized output. The expected storage contracts below remain hypotheses until native capture. `ValidateStages` checks explicit saved `koiweights.Data` entries with a closed width map. It is a payload validator, not a capture importer or native qualification gate.

| Saved stages | Expected storage |
| --- | --- |
| `input`, `transfer`, `abs`, `hproduct`, `hround`, `hclip` | Half `[3,8,64,512]`, strides `[262144,32768,512,1]` |
| `amax`, `hscale` | Half `[3,8,64]`, strides `[512,64,1]` |
| `char` | Int8 `[3,8,64,512]`, strides `[262144,32768,512,1]` |
| `fscale`, `freciprocal`, `untouched-freciprocal` | Float32 `[3,8,64]`, strides `[512,64,1]` |
| `i8tile`, `untouched-i8tile` | Int8 `[3,8,4,32,16,16]`, strides `[262144,32768,8192,256,16,1]` |

All data entries require little endian storage, exact sizes, hashes, and payload files. Unknown dtypes and missing stages fail. Transfer parity and untouched parity use exact hashes. Float32 output comparisons preserve bit patterns without tolerance or a Go reciprocal approximation.

Fixtures include the genuine saved buffer, row and axis variation, positive and negative round ties, values near 127 and 128, neighboring half ranges, largest finite half, subnormal elements, all 65536 half patterns, signed zero rows, tiny ranges, infinities, and several NaN patterns. These are observational inputs. The genuine input is explicitly unclassified. This package makes no domain-admission claim from its values. The fixtures do not exhaust all finite row combinations or the 31743 positive finite nonzero ranges.

## Held native work

The generated C++ has not been compiled or executed as part of this source-only unit. There is intentionally no working native capture command here. The coordinator must first integrate the reviewed runner maintenance path rather than copy private orchestration from `koiweights`.

Before any CPU or CUDA native work, that runner must require the pinned CUDA13 build closure, coordinator-qualified ABI4 basic runtime evidence, and explicit runtime admission. It must open the original existing `/home/jake/Projects/nanopore_basecaller/results/stock/host.lock` without creation or truncation, acquire it exclusively without blocking, and hold it throughout native work. An admission flag alone does not qualify the runtime. Missing or invalid qualification and a held lock must fail before a compiler, producer, or GPU query launches.

The next native owner must add and verify these pieces.

1. Compile the generated source with pinned compiler flags, compiler executable, depfile closure, linked libraries, and observed mapped libraries. Preserve raw command argv, UTC clocks, exits, stdout, stderr, and hashes before interpreting results.
2. Capture actual Torch configuration and CUDA environment. For CUDA, retain the actual GPU UUID, driver, ordinal, and configuration under admission. Label CPU and CUDA captures separately against identical half input pins.
3. Import observed metadata and payloads into a closed capture receipt. Independently require every source, payload, build, ABI4, and runtime pin. Reject deleted pins and unknown fields. Preserve partial fixture exceptions without fabricating missing stages.
4. Validate every stage, raw transfer bytes, and untouched final outputs. Compare CPU and CUDA captures stage by stage. Exhaust positive finite range classification before selecting a production domain.
5. After arithmetic evidence exists, choose the smallest production constructor. No current CPU or CUDA arithmetic equality or arbitrary Go reciprocal parity is inferred.
6. Add producer-free consumer replay. Pin current consumer sources and executable separately from saved producer sources. Require each named test to run and pass, with no skip or empty-regex acceptance. `RequireTestPass` rejects those false passes, and `ValidateCommand` checks raw receipt hashes and clocks first. A pure host replay must not initialize a GPU. A future GPU consumer also needs matching GPU UUID, driver, and configuration evidence.

Native compilation, CPU arithmetic capture, CUDA arithmetic capture, domain classification, Koi kernel use, biological accuracy, and performance remain separate evidence gates. Source-only preparation and passing Go tests close none of those gates.

The Model the Domain principle placed dtype widths, stage shapes, and fixture identities in closed tables. The throughput checkpoint for this unit is source extraction, genuine input reuse, deterministic preparation, and rejection checks. It is not a sequencing throughput measurement.
