**PASS for the frozen SUP QKV FP16 preparation unit.** No blocking correctness finding in the assigned source, fixture producer, or saved native comparison. This verdict covers host half-bit preparation only.

The reviewed repository is `/home/jake/Projects/nanopore_basecaller_go`, branch `port/go-caller`. Its HEAD advanced from `612fb0c4c75aeb1ee6cade5442ed862274366e4a` to `69c35c4c618f497c8aefd515ecdea7d5bc7c28dc` during separate parent work. The reviewed files remained untracked at final observation. This verdict therefore binds the eight file hashes in [source-before.stdout](/tmp/nanopore-go-koiweights-review/source-before.stdout), which equal [source-after.stdout](/tmp/nanopore-go-koiweights-review/source-after.stdout), and the saved [frozen-source.tar](/tmp/nanopore-go-koiweights-review/frozen-source.tar). It does not qualify either HEAD without those scoped bytes.

The final native capture SHA256 is `3faad1885b81146b604eea6e2028767ba204a6f62b71a782add4a268d7b5aae1` at [attempt-qkv-f16-002/manifest.json](/home/jake/Projects/nanopore_basecaller_go/artifacts/koiweights/attempt-qkv-f16-002/manifest.json). The retained replay SHA256 is `404e855289e82594a1d6c86e0da8c18ac46c0f59e390e927640a7ee14486c126` at [replay-qkv-f16-002/manifest.json](/home/jake/Projects/nanopore_basecaller_go/artifacts/koiweights/replay-qkv-f16-002/manifest.json). Both match the assigned artifact identities.

Source inspection confirms the accepted design in `/tmp/nanopore-go-gpu-design/design.md`.

- `prepareSUPQKVF16Weights` rejects every wrong length before allocating output arrays. It returns a typed `qkvWeightShapeError` and nil result slices.
- For Q and K, output D row `d` reads source row `d/2+(d%2)*32`. V copies its original D row. The first result preserves the unrounded bits.
- The tile index follows `[q,h,d/16,c/8,d%16,c%8]` with shape `[3,8,4,64,16,8]` and strides `[262144,32768,8192,128,8,1]`.
- The second result uses `(bits+8)&0xfff0` in uint16 arithmetic. Exhaustive native comparison covers signed zero, NaNs, overflow, and modular wrap. The `0x7bff` overflow and `0x0008` tie distinguish saturation and round-to-even replacements.
- Each call creates two independent buffers. There is no native handle, cache, global result state, model loading, or inference in the preparation function. Input preservation, result independence, concurrent immutable-input calls, and race behavior passed.

The independent artifact audit freshly hashed 5,010 distinct paths. It checked all 4,553 qualified extracted archive members, all 4,916 capture producer-input pins, generated files, compiler and Go tool pins, fixture payloads, and raw command stdout and stderr hashes. See [artifact-audit.json](/tmp/nanopore-go-koiweights-review/artifact-audit.json) and [hashed-paths.json](/tmp/nanopore-go-koiweights-review/hashed-paths.json).

The complete `TxModules.cpp` hash and clean native revision `8b8fc5d36a9c0baab262a743cb175b5878e38ca6` match the capture. The inclusive blocks at lines 99-106, 445-452, and 462-464 match the generated producer verbatim. I read the complete shim. It adds CPU tensor storage, raw half I/O, one genuine archive load, one float32-to-half cast, and saved runtime maps. It executes no model graph. The genuine archive, float32 snapshot, and half-input hashes match the documented pins.

The saved compiler receipt identifies GCC 16.2.1. Its exact argv is retained in the capture manifest. It compiles the single generated adapter with C++17, the declared ABI flag, CPU13 include paths, a depfile, and only `-ltorch_cpu -lc10`. The dependency and mapped-runtime checks pass. Saved `/proc/self/maps` contains the qualified `libtorch_cpu.so`, `libc10.so`, and CPU dependencies. No CUDA, Koi, cuDNN, or NVRTC library appears. The retained replay has exactly one command, the Go consumer.

All fresh commands below ran from `/home/jake/Projects/nanopore_basecaller_go`. Each command has a `.command`, `.stdout`, `.stderr`, and `.sha256` receipt under `/tmp/nanopore-go-koiweights-review`. [results.tsv](/tmp/nanopore-go-koiweights-review/results.tsv) records UTC start and finish clocks and exits.

```sh
go test -json -count=1 ./internal/model ./verification/koiweights/... -run '^(TestSUPQKVF16|TestDataBoundary|TestOutputRejectsOtherModuleBeforeWrite|TestNonblockingLock)'

env NANOPORE_KOI_QKV_MANIFEST=/home/jake/Projects/nanopore_basecaller_go/artifacts/koiweights/attempt-qkv-f16-002/manifest.json strace -f -qq -e trace=execve -o /tmp/nanopore-go-koiweights-review/native-replay.execve.log go test -json -count=1 -race ./internal/model -run '^TestSUPQKVF16'

env CGO_ENABLED=0 NANOPORE_KOI_QKV_MANIFEST=/home/jake/Projects/nanopore_basecaller_go/artifacts/koiweights/attempt-qkv-f16-002/manifest.json go test -json -count=1 ./internal/model -run '^TestSUPQKVF16'

bash /tmp/nanopore-go-koiweights-review/rerun-mutants.sh

env CGO_ENABLED=0 go run /tmp/nanopore-go-koiweights-review/probe.go

env UV_CACHE_DIR=/tmp/nanopore-go-koiweights-review/uv-cache uv run --offline --no-project --no-managed-python --no-python-downloads python /tmp/nanopore-go-koiweights-review/audit.py
```

All six commands exited 0. The ordinary scoped run explicitly skipped the optional native fixture test without its environment variable. Both required native runs executed and passed `TestSUPQKVF16Native` and all three fixture subtests without a skip. They compare both complete 786,432-element outputs for each fixture, the pre-round tile, and all 65,536 source rounding outputs. The [execve trace](/tmp/nanopore-go-koiweights-review/native-replay.execve.log) contains the Go toolchain and test process, including ordinary GCC link probes for the race test. It contains no native oracle producer execution and no Torch producer compilation.

The mutant script reruns the six frozen source-overlay controls. Each individual `go test -json -count=1 -overlay ... ./internal/model -run '^TestSUPQKVF16'` exited 1 and emitted a named behavioral test failure. The controls omit Q/K interleaving, interleave V, change tile order, round the preserved buffer early, saturate, or round to even. The script requires behavioral failure, so compiler failure alone cannot pass this check. Individual exact commands and raw failures are in `negative-*.command` and `negative-*.stdout`.

The review-only probe independently accepts the actual capture, then rejects 19 malformed cases through the actual `ValidateCapture` implementation. They cover input and output metadata, foreign shape, byte counts, fixed options, archive and weight pins, source and library pins, failed or missing producer receipts, and copied source, input, and output byte corruption. [probe-results.json](/tmp/nanopore-go-koiweights-review/probe-results.json) records the exact error for each case.

The probe uses a clearly derived manifest copy under the review directory for mutation tests. Producer source paths and generated-file paths are relocated to byte-identical copies. The derived producer argv points to the copied producer solely to satisfy path identity checks. That copied file is never executed. This is a verifier fixture, not another native capture.

In that temporary module, changing only the Go consumer copy changes its hash from `41fb7fe5e644b962b43619ddc33e3da3f83076ec53dc9fbb2791f375ba5267a3` to `d6470cc183186b004da76ee5e1c71cee02af2623243a1a9eedd25e118618038d`. Source-identical producer validation still passes. The old consumer receipt retains the old hash and cannot describe the changed copy. `consumer` records current consumer sources before every new comparison and verifies them afterward. Native payload reuse and fresh consumer comparison are separate as required. This probe does not claim a new comparison for the altered copy.

The same probe calls `AdmissionControl` with the actual original stock lock. Its fresh [blocked manifest](/tmp/nanopore-go-koiweights-review/module-copy/artifacts/koiweights/admission-control/manifest.json) has `admission_blocked`, zero `Commands`, and zero `Generated`. Source inspection confirms read-only open without create or truncate, nonblocking exclusive flock, and lock ownership through native build and capture. The fresh API control does not observe the separate CLI exit-3 mapping, which is source-inspected.

I also checked the seven retained final broad-check receipts and their raw log hashes. All seven exits are 0 in [attempt-checks-002/results.tsv](/home/jake/Projects/nanopore_basecaller_go/artifacts/koiweights/attempt-checks-002/results.tsv). Their scoped source hashes match this review. Earlier broad failures remain in `attempt-checks-001`. I did not rerun core parity or the broad suite because the fresh review target is this independent unit and the exact frozen broad receipts are already available.

No implementation or acceptance check was changed. This review writes reports and verifier fixtures only under its assigned directory. It launches no native producer, model inference, GPU work, download, or child agent. The maintained Replay CLI was not freshly invoked because it requires output beneath the production module's `artifacts/koiweights`; the assigned writable scope excludes that path. Fresh direct native-consumer replay uses the same required consumer test and validates the actual capture.

One evidence-only finalizer attempt failed because it asserted that the entire original experiment workspace remained clean. The original workspace was clean earlier, but a separate `source-qualification-resume` decision row appeared at `2026-10-23T14:03:01Z`. [original-decisions-delta.stdout](/tmp/nanopore-go-koiweights-review/original-decisions-delta.stdout) retains that unrelated delta. The review did not write it. The failed projection is preserved as `final-evidence-check.*`; the second projection records the observed concurrent change and passes. No tool was broken. A globally clean original workspace is therefore not claimed at final time.

The bounded evidence leaves GPU execution, default SUP int8 behavior, int8 scales, FP8, GPU half conversion, activation arithmetic, RoPE, attention, tensor ABI changes, current caller integration, score tensors, decoding, FASTQ, biological accuracy, and performance unqualified. Only one genuine QKV archive was used. Saved source-block CPU parity does not qualify the full native translation unit or original module ABI. Receipt authenticity against deliberate coordinated rewriting of manifests and payloads is not established.

Prove It Works shaped the review by requiring actual byte comparisons, raw receipt checks, fresh behavioral failures, and a producer-free replay rather than relying on the implementation summary or passing builds. The final machine-readable assessment is [review-summary.json](/tmp/nanopore-go-koiweights-review/review-summary.json).
