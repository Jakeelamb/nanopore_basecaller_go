# Go GPU caller feature cards

Generated from `cards.json`. Read [the dispatch rules](README.md) before assigning work.

The commands marked to implement are deliverables, not existing tools. A source-ready card still needs native acceptance.

| Card | Work | State | Dependencies |
| --- | --- | --- | --- |
| [P01](#p01) | Freeze supported runtime, dispatch and CLI profile | planned | None |
| [P02](#p02) | Extend the single typed CUDA ABI and Koi descriptors | planned | P01 |
| [P03](#p03) | Plan source workspaces and checked view lifetimes | planned | P01 |
| [P04](#p04) | Qualify official CUDA12 identity separately from source CUDA13 | planned | P01 |
| [P05](#p05) | Freeze capture, replay and admission contracts | planned | None |
| [H01](#h01) | Capture all HAC GPU stages once per declared case | planned | P01, P05 |
| [H02](#h02) | Prepare and execute the first two HAC convolutions | planned | P02, P03, H01 |
| [H03](#h03) | Execute final HAC convolution and source fallbacks | planned | H02 |
| [H04](#h04) | Prepare seven HAC FLSTM quantized caches | planned | H01, P02 |
| [H05](#h05) | Schedule the seven fused HAC recurrent layers | planned | H03, H04 |
| [H06](#h06) | Implement the half FLSTM fallback timestep loop | planned | P02, P03, H01 |
| [H07](#h07) | Prepare and run HAC output projection | planned | H04, P02, P03 |
| [H08](#h08) | Compose fixed HAC inference in Go | planned | H02, H03, H05, H06, H07 |
| [H09](#h09) | Add default HAC variable-chunk inference | planned | H08, I01 |
| [S00](#s00) | Capture the complete SUP GPU oracle once per case | planned | P01, P05, S01 |
| [S01](#s01) | Finish the held SUP int8 arithmetic capture | source-ready | None |
| [S02](#s02) | Implement admitted Go int8 row arithmetic | planned | S01, P01 |
| [S03](#s03) | Construct and tile device-precision RoPE | planned | P02, S00 |
| [S04](#s04) | Prepare projection, norms and non-FP8 caches | planned | S02, P02, S00 |
| [S05](#s05) | Qualify conditional FP8 conversion and caches | planned | P01, P02, S00 |
| [S06](#s06) | Tile activations and refresh initial int8 storage | planned | S02, P02, S00, I01 |
| [S07](#s07) | Execute tagged QKV projection and rotary | planned | P02, S00, S03, S04 |
| [S08](#s08) | Execute fixed and variable-window attention | planned | P02, S00, I01 |
| [S09](#s09) | Execute projection and FC2 linear operations | planned | P02, S00, S04, S05 |
| [S10](#s10) | Execute residual RMSNorm and precision refresh | planned | P02, S00, S04 |
| [S11](#s11) | Execute selected SwiGLU arithmetic | planned | P02, S00, S04, S05 |
| [S12](#s12) | Execute five SUP convolutions and VCS LUTs | planned | P02, S00, I01 |
| [S13](#s13) | Execute upsampling and half-scaled CRF projection | planned | P02, S00 |
| [S14](#s14) | Compose all 18 SUP encoders in Go | planned | S02, S03, S04, S05, S06, S07, S08, S09, S10, S11, S12, S13 |
| [D01](#d01) | Resolve and capture the opaque GPU decoder contract | planned | P01, P02, P05, H01, S00 |
| [D02](#d02) | Implement Go-owned guide and posterior scans | planned | D01, P02 |
| [D03](#d03) | Implement Go-owned GPU beam decisions | planned | D01, P02 |
| [D04](#d04) | Implement GPU final quality, moves and unpacking | planned | D01, P02 |
| [I01](#i01) | Freeze fixed and packed batch geometry | planned | P01, P05 |
| [I02](#i02) | Join GPU chunk results to source read assembly | planned | I01, D02, D03, D04 |
| [I03](#i03) | Schedule bounded GPU batches and safe drain | planned | I01, P02 |
| [I04](#i04) | Account for every requested parent and emitted child | planned | None |
| [I05](#i05) | Reuse default postcall with GPU metadata and replay | planned | I02, I04 |
| [I06](#i06) | Write unaligned BAM and exact FASTQ contracts | planned | I04, I05 |
| [I07](#i07) | Integrate the supported GPU caller and CLI | planned | P01, P02, H09, S14, D02, D03, D04, I02, I03, I04, I05, I06 |
| [A00](#a00) | Freeze final conformance, accuracy and timing protocols | planned | None |
| [A01](#a01) | Qualify complete HG002 development and validation inputs | planned | A00, I07, P04 |
| [A02](#a02) | Verify biological equivalence and ALT retention | planned | A00, A01 |
| [A03](#a03) | Measure matched complete-process GPU performance | planned | A00, A01, A02 |
| [A04](#a04) | Audit release, reproducibility and maintenance closure | planned | P01, P02, P03, P04, P05, H01, H02, H03, H04, H05, H06, H07, H08, H09, S00, S01, S02, S03, S04, S05, S06, S07, S08, S09, S10, S11, S12, S13, S14, D01, D02, D03, D04, I01, I02, I03, I04, I05, I06, I07, A00, A01, A02, A03 |

<a id="p01"></a>

## P01. Freeze supported runtime, dispatch and CLI profile

State is **planned**. Owner role is `pstack_worker; pstack_deep for arithmetic or lifetime work`. Dependencies are none. Resources are source, native-build, gpu-capture.

### Own these files

- `provenance/gpu-target.json`
- `docs/gpu-target.md`
- `verification/gpuprofile/cmd/main.go`
- `verification/gpuprofile/profile.go`
- `verification/gpuprofile/profile_test.go`
- `verification/gpuprofile/native/probe.cpp`

### Implement this boundary

Immutable BackendIdentity and DispatchPlan bind the two model configs, runtime family, device UUID, capability queries, default dev options, fixed/VCS and selected precision before caches exist.

- Record actual Koi F16/E4M3 availability and fused HAC predicates. Freeze supported options, explicit batch policy and all four trim/split combinations. If the target selects generic SUP, add its cards before claiming completion.

### Read and reuse

- external/dorado-source/dorado/nn/KoiUtils.cpp:7-43; nn/TxModules.cpp:403-441,805-834; api/runner_creation.cpp:24-60; cli/cli_lib/basecaller.cpp:1192

- docs/cuda-kernel-trace.md; accepted ABI4 runtime identity; config and 50/139 weight manifests.

### Pass these gates

- Source predicates and independently observed dispatch agree. Default VCS and selected FP8 cannot be disabled to pass. Reject unsupported models, families or flags before inference.
- Pin exact source and inputs, command clocks/exits/raw logs, consumer code and required test names. No required test may skip. Audit exact live bytes and source-declared initialized regions; keep uninitialized storage exclusions prospective.

### Reject these failures

- Change capability result, CUDA_VISIBLE_DEVICES presence, runtime family, config hash or a default precision option.

### Run or supply these commands

- to-implement. `go run ./verification/gpuprofile/cmd -plan provenance/gpu-target.json -runtime-admitted -out NEW_ATTEMPT`

Independent review. A fresh pstack_verifier audits the exact delta, raw receipts and one effective negative control. Source-only evidence cannot close native acceptance. Coordinate shared-file changes with the ABI or caller integration owner.

<a id="p02"></a>

## P02. Extend the single typed CUDA ABI and Koi descriptors

State is **planned**. Owner role is `pstack_worker; pstack_deep for arithmetic or lifetime work`. Dependencies are P01. Resources are source, native-build, gpu-capture.

### Own these files

- `internal/tensor/abi.h`
- `internal/tensor/loader.c`
- `internal/tensor/loader.h`
- `internal/tensor/native_cgo.go`
- `internal/tensor/native_nocgo.go`
- `internal/tensor/cuda_storage.go`
- `internal/tensor/cuda_koi.go`
- `verification/tensor/backend.cpp`
- `verification/tensor/cuda_storage.inc`
- `verification/tensor/build/main.go`
- `internal/tensor/cuda_storage_test.go`
- `internal/tensor/cuda_koi_test.go`
- `verification/tensor/cuda_backend.h`
- `verification/tensor/source-pins.json`
- `verification/tensor/cuda/main.go`
- `verification/tensor/cuda/main_test.go`

### Implement this boundary

One ABI integrator owns shared declarations, loader symbols, build wiring and registration. DeviceBuffer/View supports F16,I8,U8,I32,F32 and selected FP8 plus tagged Koi dimensions and typed optional scale owners.

- Keep the reviewed synchronous drain contract first. Add typed allocate/upload/download/copy/fill and required elementary operations. Validate rank, tags, offset/strides, overlap, bounds, device, dtype and scales; no retained Go pointers. Wire worker fragments serially.

### Read and reuse

- internal/tensor/abi.h:30-59; external/dorado-source/dorado/nn/TxModules.cpp:30-79; pinned koi.h:3-36

- ABI4 Device/Stream/HalfBuffer, accepted GEMM and ownership receipts; verification/tensor/runner. Do not repeat accepted basic cases unless changed inputs invalidate them.

### Pass these gates

- Actual raw storage roundtrips and source-shaped operations pass on the qualified family. Required CPU contracts still pass against CPU-only and CUDA-enabled builds; old incompatible ABI is rejected.
- Pin exact source and inputs, command clocks/exits/raw logs, consumer code and required test names. No required test may skip. Audit exact live bytes and source-declared initialized regions; keep uninitialized storage exclusions prospective.

### Reject these failures

- Foreign engine/device, released parent, dtype/tag/scale mismatch, offset overflow, forbidden alias, launch error and close during work.

### Run or supply these commands

- to-implement. `go test -json -count=1 ./internal/tensor -run '^TestCUDAStorageAndKoiDescriptors$'`
- to-implement. `go run ./verification/tensor/cuda -unit storage-koi -capture CAPTURE -backend BACKEND -runtime-admitted -out NEW_ATTEMPT`

Independent review. A fresh pstack_verifier audits the exact delta, raw receipts and one effective negative control. Source-only evidence cannot close native acceptance. Coordinate shared-file changes with the ABI or caller integration owner.

<a id="p03"></a>

## P03. Plan source workspaces and checked view lifetimes

State is **planned**. Owner role is `pstack_worker; pstack_deep for arithmetic or lifetime work`. Dependencies are P01. Resources are source, native-build.

### Own these files

- `internal/model/hac_gpu_memory.go`
- `internal/model/hac_gpu_memory_test.go`
- `verification/hacgpu/memory.go`

### Implement this boundary

Pure checked arena plan separates reservation from use, maximum from used extents, orientation and native binding. SUP invocation-owned scratch uses the same checked storage contract.

- Implement 256-byte alignment, alternating ends and overlap checks. CUTLASS storage is T+5; CUBLAS is T+3. Declare initialized sentinels and valid ranges from source before capture.

### Read and reuse

- external/dorado-source/dorado/nn/WorkingMemory.cpp:30-113; include/nn/WorkingMemory.h:55-81; ConvStack.cpp:191-211; FLSTMStack.cpp:97-108

- Source-pinned shapes and existing half views; new small planner oracle only if no adequate retained trace exists.

### Pass these gates

- Reservation/use traces and live views match an extracted source planner; overflow, overlap and undersized plans reject before device allocation.
- Pin exact source and inputs, command clocks/exits/raw logs, consumer code and required test names. No required test may skip. Audit exact live bytes and source-declared initialized regions; keep uninitialized storage exclusions prospective.

### Reject these failures

- T+3 substituted for CUTLASS, wrong reverse view, under-reservation, arithmetic overflow or reused in-flight scratch.

### Run or supply these commands

- to-implement. `go test -json -count=1 ./internal/model -run '^TestHACGPUMemory$'`

Independent review. A fresh pstack_verifier audits the exact delta, raw receipts and one effective negative control. Source-only evidence cannot close native acceptance. Coordinate shared-file changes with the ABI or caller integration owner.

<a id="p04"></a>

## P04. Qualify official CUDA12 identity separately from source CUDA13

State is **planned**. Owner role is `pstack_worker; pstack_deep for arithmetic or lifetime work`. Dependencies are P01. Resources are source, native-build, gpu-capture.

### Own these files

- `provenance/gpu-family-compatibility.json`
- `verification/gpufamily/check.go`
- `docs/gpu-family-compatibility.md`
- `verification/gpufamily/cmd/main.go`
- `verification/gpufamily/check_test.go`

### Implement this boundary

Separate family-pinned qualification and explicit output verdicts for source CUDA13 and official CUDA12. No mixed-process Torch/Koi families.

- Inventory complete retained SDK/header/library assets first. Obtain only missing pinned assets through an admitted task if required. Qualify the chosen source-family operations, then compare Go outputs with retained official controls independently.

### Read and reuse

- external/dorado-source/cmake/Torch.cmake; cmake/Koi.cmake; docs/independent-review-cuda-abi4.md; original research/source-qualification-results.md

- Original source qualification, existing CUDA13 archive/member pins and official controls. The older trace's partial CUDA12 archive is a historical inventory, not a fresh assertion.

### Pass these gates

- Exact supported output endpoints have separate source13 and official12 verdicts. A missing SDK or an official mismatch remains open and blocks an official-byte-identity claim.
- Pin exact source and inputs, command clocks/exits/raw logs, consumer code and required test names. No required test may skip. Audit exact live bytes and source-declared initialized regions; keep uninitialized storage exclusions prospective.

### Reject these failures

- Mix Koi12 with Torch13, change loader closure, relabel source13 as official12, or reuse a different device's acceptance.

### Run or supply these commands

- to-implement. `go run ./verification/gpufamily/cmd -plan provenance/gpu-family-compatibility.json -runtime-admitted -out NEW_ATTEMPT`

Independent review. A fresh pstack_verifier audits the exact delta, raw receipts and one effective negative control. Source-only evidence cannot close native acceptance. Coordinate shared-file changes with the ABI or caller integration owner.

<a id="p05"></a>

## P05. Freeze capture, replay and admission contracts

State is **planned**. Owner role is `pstack_worker; pstack_deep for arithmetic or lifetime work`. Dependencies are none. Resources are source.

### Own these files

- `docs/gpu-oracle-contract.md`
- `verification/gpuoracle/contract.go`
- `verification/gpuoracle/contract_test.go`
- `verification/gpucaller/cmd/main.go`
- `verification/gpucaller/registry.go`
- `verification/gpucaller/registry_test.go`

### Implement this boundary

A small shared contract reuses the maintained runner. Prepared, frozen producer, imported consumer and accepted runtime identities stay distinct; no new generic execution framework.

- Specify closed schemas, prospective live/initialized/guard ranges, partial-failure retention and named-test accounting. Before a broader producer, prove a genuine-file import and corrupt-copy rejection. Resource intake precedes original nonblocking host.lock; import is producer-free.
- Own the thin finite caller-verification CLI and handler registration from the first source checkpoint. Other card owners submit serialized wiring patches; do not build a general workflow framework. H01/S00/D01 likewise own their model/decoder checker entrypoints and shared registries.

### Read and reuse

- verification/tensor/runner/{admission,command,archive,gotests}.go; verification/koiquant/{capture,receipt,import,replay}.go

- Accepted ABI4 runner and reviewed koiquant receipt design. Keep all raw failures; repair observers from retained bytes.

### Pass these gates

- Missing admission, invalid qualification and held temporary-lock controls launch zero native children. A real fixture handoff and effective corruption rejection pass before expanded capture.
- Pin exact source and inputs, command clocks/exits/raw logs, consumer code and required test names. No required test may skip. Audit exact live bytes and source-declared initialized regions; keep uninitialized storage exclusions prospective.

### Reject these failures

- Duplicate/null/missing fields, zero-match or skipped tests, changed payload, stale consumer pins, self-produced oracle or mismatch-derived comparison mask.

### Run or supply these commands

- to-implement. `go test -json -count=1 ./verification/gpuoracle`

Independent review. A fresh pstack_verifier audits the exact delta, raw receipts and one effective negative control. Source-only evidence cannot close native acceptance. Coordinate shared-file changes with the ABI or caller integration owner.

<a id="h01"></a>

## H01. Capture all HAC GPU stages once per declared case

State is **planned**. Owner role is `pstack_worker; pstack_deep for arithmetic or lifetime work`. Dependencies are P01, P05. Resources are source, native-build, gpu-capture.

### Own these files

- `verification/hacgpu/cmd/main.go`
- `verification/hacgpu/capture.go`
- `verification/hacgpu/import.go`
- `verification/hacgpu/native/oracle.cpp`

### Implement this boundary

Verification-only untouched HAC graph produces complete caches, arena storage, every convolution/recurrent layer, projection and scores. One producer supplies all downstream HAC cards.

- Prepare source/import first. Compile and capture one fixed padded graph per predeclared fused/fallback condition, with source dispatch and full raw parent storage. Add only missing VCS cases after I01 geometry. Native graph wrapper remains oracle-only.

### Read and reuse

- external/dorado-source/dorado/basecall/model/CRFModel.cpp:87-145; nn/ConvStack.cpp:214-373; nn/FLSTMStack.cpp:125-295; nn/FactorisedLinearLayer.cpp:52-106

- Existing normalized HG002 half input and 50 weight pins; accepted ABI4 identity. CPU HAC scores are context, not CUDA truth.

### Pass these gates

- Genuine normalized development input imports correctly and a corrupt stage/guard rejects. Stage metadata and compared valid ranges are source-derived. Consumers reuse every successful stage without another graph forward.
- Pin exact source and inputs, command clocks/exits/raw logs, consumer code and required test names. No required test may skip. Audit exact live bytes and source-declared initialized regions; keep uninitialized storage exclusions prospective.

### Reject these failures

- Changed source/input/runtime/dispatch, corrupt initialized padding, missing stage, incorrect capacity or candidate graph used as its own oracle.

### Run or supply these commands

- to-implement. `go run ./verification/hacgpu/cmd -mode capture -unit fixed -plan PLAN -runtime-admitted -out NEW_ATTEMPT`

Independent review. A fresh pstack_verifier audits the exact delta, raw receipts and one effective negative control. Source-only evidence cannot close native acceptance. Coordinate shared-file changes with the ABI or caller integration owner.

<a id="h02"></a>

## H02. Prepare and execute the first two HAC convolutions

State is **planned**. Owner role is `pstack_worker; pstack_deep for arithmetic or lifetime work`. Dependencies are P02, P03, H01. Resources are source, native-build, gpu-capture.

### Own these files

- `internal/model/hac_gpu_smallconv.go`
- `internal/model/hac_gpu_smallconv_test.go`
- `verification/tensor/hac_smallconv.inc`

### Implement this boundary

Go owns two width-5 stride-1 SiLU calls, weight packing, padding and retained output views; one named host_convolution_f16 primitive per call.

- Cast and permute weights to [W*Cin,Cout]; preserve the cropped view's batch stride and nine rows of last-convolution padding.
- Keep host-only import/replay separate from native verify-consumer. The latter requires current coordinator admission, matching runtime/build pins and the original host.lock before every native invocation.

### Read and reuse

- external/dorado-source/dorado/nn/ConvStack.cpp:143-148,226-263; koi.h:438-451

- H01 stage/weight captures and accepted raw half transfer; do not substitute CPU or cuDNN convolution.

### Pass these gates

- Both complete valid outputs, half cache bits and initialized padding match H01; input/output lifetime survives parent views as declared.
- Pin exact source and inputs, command clocks/exits/raw logs, consumer code and required test names. No required test may skip. Audit exact live bytes and source-declared initialized regions; keep uninitialized storage exclusions prospective.

### Reject these failures

- Contiguous-but-wrong-stride view, wrong SiLU, swapped convolution weights or omitted padding.

### Run or supply these commands

- to-implement. `go run ./verification/hacgpu/cmd -mode verify-consumer -unit small-convs -capture CAPTURE -backend BACKEND -runtime-admitted -out NEW_ATTEMPT`

Independent review. A fresh pstack_verifier audits the exact delta, raw receipts and one effective negative control. Source-only evidence cannot close native acceptance. Coordinate shared-file changes with the ABI or caller integration owner.

<a id="h03"></a>

## H03. Execute final HAC convolution and source fallbacks

State is **planned**. Owner role is `pstack_worker; pstack_deep for arithmetic or lifetime work`. Dependencies are H02. Resources are source, native-build, gpu-capture.

### Own these files

- `internal/model/hac_gpu_lastconv.go`
- `internal/model/hac_gpu_lastconv_test.go`
- `verification/tensor/hac_lastconv.inc`

### Implement this boundary

Go selects width-19 stride-6 tanh convolution, source non-success host_linear fallback or window/half-GEMM path, and converts to the selected layout.

- Preserve cutlass_conv/koi_conv/koi_conv_f32 dispatch and padding writes. Share host_linear and bias/activation primitive declarations through P02.
- Keep host-only import/replay separate from native verify-consumer. The latter requires current coordinator admission, matching runtime/build pins and the original host.lock before every native invocation.

### Read and reuse

- external/dorado-source/dorado/nn/ConvStack.cpp:264-373; torch_utils/cuda_utils.cpp:377-398; koi.h:241-258,370-391,406-436

- H01 retained branch stages and ABI4 GEMM for its accepted cases; only new source-shaped GEMM domains need added cases.

### Pass these gates

- Source-selected fast case and predeclared supported fallbacks match their H01 stages. A fallback success never counts as fast-path support.
- Pin exact source and inputs, command clocks/exits/raw logs, consumer code and required test names. No required test may skip. Audit exact live bytes and source-declared initialized regions; keep uninitialized storage exclusions prospective.

### Reject these failures

- F32 accumulation mutant, changed fallback status, incorrect I8 conversion or overwritten guards.

### Run or supply these commands

- to-implement. `go run ./verification/hacgpu/cmd -mode verify-consumer -unit last-conv -capture CAPTURE -backend BACKEND -runtime-admitted -out NEW_ATTEMPT`

Independent review. A fresh pstack_verifier audits the exact delta, raw receipts and one effective negative control. Source-only evidence cannot close native acceptance. Coordinate shared-file changes with the ABI or caller integration owner.

<a id="h04"></a>

## H04. Prepare seven HAC FLSTM quantized caches

State is **planned**. Owner role is `pstack_worker; pstack_deep for arithmetic or lifetime work`. Dependencies are H01, P02. Resources are source, gpu-capture.

### Own these files

- `internal/model/hac_gpu_flstm_weights.go`
- `internal/model/hac_gpu_flstm_weights_test.go`
- `verification/hacgpu/flstm_weights.go`

### Implement this boundary

Go binds all seven layers and owns half-cast-before-quantization, row reductions, I8/half-scale packing, permutations and input-bias-only cache.

- Implement exact down-IH/down-HH and combined up caches. Establish HAC-specific row geometry and scale dtype; SUP QKV evidence alone is insufficient.

### Read and reuse

- external/dorado-source/dorado/nn/FLSTMStack.cpp:165-217; torch_utils/tensor_utils.cpp:291-297

- 50 HAC weight pins and H01 arithmetic/cache captures; reuse narrow quantization observations only when dtype/shape/domain pins match.

### Pass these gates

- Every live cache byte and scale bit for all 42 recurrent weights matches H01. Zero/nonfinite behavior has an explicit admitted domain.
- Pin exact source and inputs, command clocks/exits/raw logs, consumer code and required test names. No required test may skip. Audit exact live bytes and source-declared initialized regions; keep uninitialized storage exclusions prospective.

### Reject these failures

- Wrong reduction axis, float32 assumption, epsilon insertion, packing/clip mutant or added recurrent bias.

### Run or supply these commands

- to-implement. `go run ./verification/hacgpu/cmd -mode replay -unit flstm-weights -capture CAPTURE -out NEW_ATTEMPT`

Independent review. A fresh pstack_verifier audits the exact delta, raw receipts and one effective negative control. Source-only evidence cannot close native acceptance. Coordinate shared-file changes with the ABI or caller integration owner.

<a id="h05"></a>

## H05. Schedule the seven fused HAC recurrent layers

State is **planned**. Owner role is `pstack_worker; pstack_deep for arithmetic or lifetime work`. Dependencies are H03, H04. Resources are source, native-build, gpu-capture.

### Own these files

- `internal/model/hac_gpu_flstm.go`
- `internal/model/hac_gpu_flstm_test.go`
- `verification/tensor/hac_flstm.inc`

### Implement this boundary

One named host_factorised_lstm operation per layer. Go owns first_reverse XOR layer direction, parity, sentinels, scratch spans and loop order.

- Require N%64=0,C=1024,K=128; zero shared scratch once before the loop. Check Koi status, launch and drain even though source ignores the integer return.
- Keep host-only import/replay separate from native verify-consumer. The latter requires current coordinator admission, matching runtime/build pins and the original host.lock before every native invocation.

### Read and reuse

- external/dorado-source/dorado/nn/FLSTMStack.cpp:125-160,220-235; koi.h:351-368

- H01 complete layer captures; H04 caches; P03 declared initialization masks.

### Pass these gates

- Each layer's live inout/scratch matches H01. Recording-backend call traces and genuine GPU outputs agree; VCS reverse T+1 remains explicit.
- Pin exact source and inputs, command clocks/exits/raw logs, consumer code and required test names. No required test may skip. Audit exact live bytes and source-declared initialized regions; keep uninitialized storage exclusions prospective.

### Reject these failures

- Wrong parity/direction, per-layer scratch re-zeroing, sync-size error, recurrent bias addition or premature cache release.

### Run or supply these commands

- to-implement. `go run ./verification/hacgpu/cmd -mode verify-consumer -unit flstm-fused -capture CAPTURE -backend BACKEND -runtime-admitted -out NEW_ATTEMPT`

Independent review. A fresh pstack_verifier audits the exact delta, raw receipts and one effective negative control. Source-only evidence cannot close native acceptance. Coordinate shared-file changes with the ABI or caller integration owner.

<a id="h06"></a>

## H06. Implement the half FLSTM fallback timestep loop

State is **planned**. Owner role is `pstack_worker; pstack_deep for arithmetic or lifetime work`. Dependencies are P02, P03, H01. Resources are source, native-build, gpu-capture.

### Own these files

- `internal/model/hac_gpu_flstm_fallback.go`
- `internal/model/hac_gpu_flstm_fallback_test.go`
- `verification/tensor/hac_lstm_step.inc`

### Implement this boundary

Go owns each timestep and three source half GEMMs followed by host_lstm_step_f16; this is a CUDA graph, not the CPU graph cast to half.

- Prepare half transposed caches, concatenated up cache and input bias; zero cell per layer. host_lstm_step_f16 returns void, so check launch/drain. Freeze supported matmul mode.
- Keep host-only import/replay separate from native verify-consumer. The latter requires current coordinator admission, matching runtime/build pins and the original host.lock before every native invocation.

### Read and reuse

- external/dorado-source/dorado/nn/FLSTMStack.cpp:239-295; torch_utils/cuda_utils.cpp:106-117,377-398; koi.h:323-331

- ABI4 half GEMM and H01 fallback stages. Forced fallback verifies algorithm only, not another device's default support.

### Pass these gates

- Forward/reverse offset traces and every declared state/output match the separately captured fallback. Source-rejected VCS fallback remains rejected.
- Pin exact source and inputs, command clocks/exits/raw logs, consumer code and required test names. No required test may skip. Audit exact live bytes and source-declared initialized regions; keep uninitialized storage exclusions prospective.

### Reject these failures

- F32-compute mutant, wrong hard activation, timestep offset, nonzero initial cell or recurrent bias addition.

### Run or supply these commands

- to-implement. `go run ./verification/hacgpu/cmd -mode verify-consumer -unit flstm-half-fallback -capture CAPTURE -backend BACKEND -runtime-admitted -out NEW_ATTEMPT`

Independent review. A fresh pstack_verifier audits the exact delta, raw receipts and one effective negative control. Source-only evidence cannot close native acceptance. Coordinate shared-file changes with the ABI or caller integration owner.

<a id="h07"></a>

## H07. Prepare and run HAC output projection

State is **planned**. Owner role is `pstack_worker; pstack_deep for arithmetic or lifetime work`. Dependencies are H04, P02, P03. Resources are source, native-build, gpu-capture.

### Own these files

- `internal/model/hac_gpu_output.go`
- `internal/model/hac_gpu_output_test.go`
- `verification/tensor/hac_output.inc`

### Implement this boundary

Go owns source fused factorized projection and its supported two-LinearCRF fallback, correct live row, scale and contiguous half NTC result.

- Pack quantized down and half up caches, zero temporary, parity=1, scale=5. Preserve counterintuitive registered dn/up naming; pass clamp to decoder rather than adding CPU-style clamp.
- Keep host-only import/replay separate from native verify-consumer. The latter requires current coordinator admission, matching runtime/build pins and the original host.lock before every native invocation.

### Read and reuse

- external/dorado-source/dorado/basecall/model/CRFModel.cpp:45-66; nn/FactorisedLinearLayer.cpp:30-106; nn/CRFModules.cpp:37-121

- H01 projection/weight captures and H04 admitted quantization geometry.

### Pass these gates

- All selected source projection intermediates and score half bits match H01, including fixed row2 and VCS row3 when applicable.
- Pin exact source and inputs, command clocks/exits/raw logs, consumer code and required test names. No required test may skip. Audit exact live bytes and source-declared initialized regions; keep uninitialized storage exclusions prospective.

### Reject these failures

- Swap same-shaped weights, scale twice, wrong parity/live row, extra clamp or output-stride error.

### Run or supply these commands

- to-implement. `go run ./verification/hacgpu/cmd -mode verify-consumer -unit output -capture CAPTURE -backend BACKEND -runtime-admitted -out NEW_ATTEMPT`

Independent review. A fresh pstack_verifier audits the exact delta, raw receipts and one effective negative control. Source-only evidence cannot close native acceptance. Coordinate shared-file changes with the ABI or caller integration owner.

<a id="h08"></a>

## H08. Compose fixed HAC inference in Go

State is **planned**. Owner role is `pstack_worker; pstack_deep for arithmetic or lifetime work`. Dependencies are H02, H03, H05, H06, H07. Resources are source, gpu-capture.

### Own these files

- `internal/model/hac_gpu.go`
- `internal/model/hac_gpu_test.go`
- `verification/hacgpu/fixed.go`

### Implement this boundary

HACGPU.Forward accepts owned half NCT plus fixed batch metadata and returns retained half NTC scores with clamp/Q metadata. Go owns every graph call.

- Compose accepted operations for one declared logical chunk in valid N=64 padding. Freeze inactive-slot policy; preserve CPU model selection until I07 integrates GPU.
- Keep host-only import/replay separate from native verify-consumer. The latter requires current coordinator admission, matching runtime/build pins and the original host.lock before every native invocation.

### Read and reuse

- external/dorado-source/dorado/basecall/model/CRFModel.cpp:87-145; basecall/CudaCaller.cpp:211-214,246-289,729-748

- H01 one-time all-stage capture and all accepted kernel/cache receipts. No model oracle rerun for consumer fixes.

### Pass these gates

- Input, caches, all stages and final scores match H01. Cancellation and native error drain safely and cannot return a partial score batch as success.
- Pin exact source and inputs, command clocks/exits/raw logs, consumer code and required test names. No required test may skip. Audit exact live bytes and source-declared initialized regions; keep uninitialized storage exclusions prospective.

### Reject these failures

- Swap layers, bind same-shaped foreign weights, contaminate inactive slots, close workspace early or add a production ForwardHAC native symbol.

### Run or supply these commands

- to-implement. `go run ./verification/hacgpu/cmd -mode verify-consumer -unit fixed-graph -capture CAPTURE -backend BACKEND -runtime-admitted -out NEW_ATTEMPT`

Independent review. A fresh pstack_verifier audits the exact delta, raw receipts and one effective negative control. Source-only evidence cannot close native acceptance. Coordinate shared-file changes with the ABI or caller integration owner.

<a id="h09"></a>

## H09. Add default HAC variable-chunk inference

State is **planned**. Owner role is `pstack_worker; pstack_deep for arithmetic or lifetime work`. Dependencies are H08, I01. Resources are source, native-build, gpu-capture.

### Own these files

- `internal/model/hac_gpu_vcs.go`
- `internal/model/hac_gpu_vcs_test.go`
- `verification/hacgpu/vcs.go`
- `verification/tensor/hac_vcs.inc`

### Implement this boundary

Consume scheduler-owned chunk membership; Go owns VCS model layout, directional encoding, preprocess operation scheduling and live projection view.

- Use subbatch32, even T_lstm, reverse T+1 and VCS sentinels. Expose host_lstm_preprocess only as a named primitive. Capture one additional complete VCS source case when geometry is ready.
- Keep host-only import/replay separate from native verify-consumer. The latter requires current coordinator admission, matching runtime/build pins and the original host.lock before every native invocation.

### Read and reuse

- external/dorado-source/dorado/nn/AuxiliaryData.cpp:24-118; nn/ConvStack.cpp:251-318; nn/FLSTMStack.cpp:125-143,220-235; api/runner_creation.cpp:24-45

- All fixed caches/operations and I01 packed metadata. Add only missing VCS observations through H01 owner.

### Pass these gates

- Mixed lengths, boundary lengths and consecutive distinct batches match retained VCS tables/stages/scores; no stale encoding and no unsupported half-fallback VCS.
- Pin exact source and inputs, command clocks/exits/raw logs, consumer code and required test names. No required test may skip. Audit exact live bytes and source-declared initialized regions; keep uninitialized storage exclusions prospective.

### Reject these failures

- Reuse previous membership, incorrect mutable interval scaling, max/used confusion, wrong row3 or VCS allowed on half fallback.

### Run or supply these commands

- to-implement. `go run ./verification/hacgpu/cmd -mode verify-consumer -unit vcs-graph -capture CAPTURE -backend BACKEND -runtime-admitted -out NEW_ATTEMPT`

Independent review. A fresh pstack_verifier audits the exact delta, raw receipts and one effective negative control. Source-only evidence cannot close native acceptance. Coordinate shared-file changes with the ABI or caller integration owner.

<a id="s00"></a>

## S00. Capture the complete SUP GPU oracle once per case

State is **planned**. Owner role is `pstack_worker; pstack_deep for arithmetic or lifetime work`. Dependencies are P01, P05, S01. Resources are source, native-build, gpu-capture.

### Own these files

- `verification/supgpu/cmd/main.go`
- `verification/supgpu/schema.go`
- `verification/supgpu/registry.go`
- `verification/supgpu/capture.go`
- `verification/supgpu/native/graph_oracle.cpp`

### Implement this boundary

Verification-only untouched SUP graph captures every cache and boundary across all 18 encoders for declared fixed/VCS/selected-precision cases. Primitive workers consume saved inputs and expected outputs independently.

- Freeze complete closed stage schemas and prospective initialized/live regions. One source producer owns each case; save half/I8/FP8/F32 scales, counters and descriptors. Prove genuine import plus corrupt-copy rejection before expanding the case matrix.

### Read and reuse

- external/dorado-source/dorado/basecall/model/TxModel.cpp:20-47; nn/TxModules.cpp:431-611,837-916; docs/cuda-kernel-trace.md

- Existing normalized half chunks, 139 SUP weight pins and original source distribution. CPU 287-stage results are regression context only.

### Pass these gates

- All declared stages and 139 weight bindings import with runtime/dispatch identity. An observer fix reimports original raw output; it never reruns 18-layer inference.
- Pin exact source and inputs, command clocks/exits/raw logs, consumer code and required test names. No required test may skip. Audit exact live bytes and source-declared initialized regions; keep uninitialized storage exclusions prospective.

### Reject these failures

- Missing layer, mismatched cache, changed precision, altered dtype/tag/shape/stride or candidate implementation used as source oracle.

### Run or supply these commands

- to-implement. `go run ./verification/supgpu/cmd -unit source-graph -capture PREPARED -runtime-admitted -out NEW_ATTEMPT`

Independent review. A fresh pstack_verifier audits the exact delta, raw receipts and one effective negative control. Source-only evidence cannot close native acceptance. Coordinate shared-file changes with the ABI or caller integration owner.

<a id="s01"></a>

## S01. Finish the held SUP int8 arithmetic capture

State is **source-ready**. Owner role is `pstack_worker; pstack_deep for arithmetic or lifetime work`. Dependencies are none. Resources are source, native-build, gpu-capture.

### Own these files

- `verification/koiquant/**`
- `docs/koi-qkv-int8.md`

### Implement this boundary

Complete the existing frozen source-only unit. One CPU compile/capture, genuine import/rejection, then CUDA with the same executable; preserve all 31 fixtures and every intermediate.

- After direct harness2 coordination and all holds/admission, execute the frozen held-commands plan. Classify all 31,743 finite positive half ranges. Keep source exceptions and CPU/CUDA differences as observations.

### Read and reuse

- docs/koi-qkv-int8.md; verification/koiquant/{capture,receipt,import,replay,compare}.go; external/dorado-source/dorado/torch_utils/tensor_utils.cpp:291-297; nn/TxModules.cpp:454-460

- artifacts/koiweights/attempt-qkv-f16-002/manifest.json; both 31-input prepared manifests; artifacts/koiquant/independent-capture-source-review-001/report.md. Never re-export weights or run a model.

### Pass these gates

- Actual compilation, complete raw CPU/CUDA captures, genuine metadata handoff, corruption rejection and exact difference/classification reports exist. This card does not implement a Go quantizer.
- Pin exact source and inputs, command clocks/exits/raw logs, consumer code and required test names. No required test may skip. Audit exact live bytes and source-declared initialized regions; keep uninitialized storage exclusions prospective.

### Reject these failures

- Absent admission, invalid receipts, held lock, changed genuine input, missing raw field, transfer mismatch, untouched-output mismatch or silent skipped tests.

### Run or supply these commands

- existing. `Read artifacts/koiquant/attempt-capture-source-handoff-001/held-commands.sh.txt; execute only after coordinated native admission`

Independent review. A fresh pstack_verifier audits the exact delta, raw receipts and one effective negative control. Source-only evidence cannot close native acceptance. Coordinate shared-file changes with the ABI or caller integration owner.

Current evidence.

- artifacts/koiquant/attempt-capture-source-handoff-001/report.md SHA256 d7c5c52ddaafe7d8795e4eeae1ca0f0bc96e04b1e8f5577b75fb3c04025add27
- artifacts/koiquant/independent-capture-source-review-001/report.md SHA256 d2b3cb61ee140c37669abd60eade63b5b8a99ff06e603b40a132992544e32dbb

<a id="s02"></a>

## S02. Implement admitted Go int8 row arithmetic

State is **planned**. Owner role is `pstack_worker; pstack_deep for arithmetic or lifetime work`. Dependencies are S01, P01. Resources are source, gpu-capture.

### Own these files

- `internal/model/supgpu/quant.go`
- `internal/model/supgpu/quant_test.go`
- `verification/supgpu/quant.go`

### Implement this boundary

Typed row geometry accepts half input and returns independent I8 values and raw reciprocal-scale bits; Go owns row/layout decisions.

- Match actual half reduction/division/product/round/clip/cast and F32 reciprocal order. If device reciprocal cannot be reproduced exactly, qualify one narrow primitive via P02; no whole-cache native replacement. Add FC1/activation row domains before use.

### Read and reuse

- external/dorado-source/dorado/torch_utils/tensor_utils.cpp:291-297; nn/TxModules.cpp:454-460,494-495,885-905

- S01 raw observations and existing internal/signal/half.go; add only missing shape/domain cases without repeating QKV producer.

### Pass these gates

- Every S01 admitted bit/intermediate matches and caller domains are explicit. Unknown/unsupported numerical domains reject or have separately reviewed evidence; no arbitrary tolerance.
- Pin exact source and inputs, command clocks/exits/raw logs, consumer code and required test names. No required test may skip. Audit exact live bytes and source-declared initialized regions; keep uninitialized storage exclusions prospective.

### Reject these failures

- Quantize already-rounded input, change reduction axis, widen too early, truncate ties, omit reciprocal or alias returned storage.

### Run or supply these commands

- to-implement. `go test -json -count=1 ./internal/model/supgpu -run '^TestSUP02Quantization'`

Independent review. A fresh pstack_verifier audits the exact delta, raw receipts and one effective negative control. Source-only evidence cannot close native acceptance. Coordinate shared-file changes with the ABI or caller integration owner.

<a id="s03"></a>

## S03. Construct and tile device-precision RoPE

State is **planned**. Owner role is `pstack_worker; pstack_deep for arithmetic or lifetime work`. Dependencies are P02, S00. Resources are source, gpu-capture.

### Own these files

- `internal/model/supgpu/rope.go`
- `internal/model/supgpu/rope_test.go`
- `verification/supgpu/rope.go`

### Implement this boundary

Go sequences source theta powers, option-dependent casts, reciprocal, positions, multiply, sin/cos and layout; native elementary operations stay individually named.

- Produce owned half [max_T/16,D/8,16,8], max_T4096,D64, sin before cos. Reject requested sequence lengths outside cache.

### Read and reuse

- external/dorado-source/dorado/nn/TxModules.cpp:162-195,466-480

- S00 captured cache and source model constants; accepted half conversion only where arithmetic order matches.

### Pass these gates

- Full raw cache bits match S00 device observations; CPU float32 rotary is not used as GPU truth.
- Pin exact source and inputs, command clocks/exits/raw logs, consumer code and required test names. No required test may skip. Audit exact live bytes and source-declared initialized regions; keep uninitialized storage exclusions prospective.

### Reject these failures

- Float32-intermediate mutant, sin/cos swap, wrong theta/position, tile-axis error or overlong sequence.

### Run or supply these commands

- to-implement. `go run ./verification/supgpu/cmd -unit rope -verify-consumer CAPTURE -backend BACKEND -runtime-admitted -out NEW_ATTEMPT`

Independent review. A fresh pstack_verifier audits the exact delta, raw receipts and one effective negative control. Source-only evidence cannot close native acceptance. Coordinate shared-file changes with the ABI or caller integration owner.

<a id="s04"></a>

## S04. Prepare projection, norms and non-FP8 caches

State is **planned**. Owner role is `pstack_worker; pstack_deep for arithmetic or lifetime work`. Dependencies are S02, P02, S00. Resources are source, gpu-capture.

### Own these files

- `internal/model/supgpu/weights.go`
- `internal/model/supgpu/weights_test.go`
- `verification/supgpu/weights.go`

### Implement this boundary

Go binds and packs projection, bias, norm, FC1 and FC2 caches from immutable half archive values. Quantized caches precede four-bit half rounding.

- Preserve FC1 gate interleave [2,128,16,512] to [128,2,16,512], I8 tile [256,32,16,16] and reciprocal scales; preserve FC2 half layout and source cast-before-transfer order.

### Read and reuse

- external/dorado-source/dorado/nn/TxModules.cpp:403-412,482-495,508-518; basecall/crf_utils.cpp:168-175

- Accepted QKV FP16 prep and all 65,536 rounding patterns; S00 cache bytes. No archive deserialization on replay.

### Pass these gates

- All compared caches match S00 with axis sentinels and complete half rounding controls. Bind all 18 sets by archive hash during graph integration.
- Pin exact source and inputs, command clocks/exits/raw logs, consumer code and required test names. No required test may skip. Audit exact live bytes and source-declared initialized regions; keep uninitialized storage exclusions prospective.

### Reject these failures

- Swap gates, omit bias/reciprocal, quantize rounded weights, permute wrong axes or mutate archived input.

### Run or supply these commands

- to-implement. `go test -json -count=1 ./internal/model/supgpu -run '^TestSUP04WeightCaches'`

Independent review. A fresh pstack_verifier audits the exact delta, raw receipts and one effective negative control. Source-only evidence cannot close native acceptance. Coordinate shared-file changes with the ABI or caller integration owner.

<a id="s05"></a>

## S05. Qualify conditional FP8 conversion and caches

State is **planned**. Owner role is `pstack_worker; pstack_deep for arithmetic or lifetime work`. Dependencies are P01, P02, S00. Resources are source, gpu-capture.

### Own these files

- `internal/model/supgpu/fp8_weights.go`
- `internal/model/supgpu/fp8_weights_test.go`
- `verification/supgpu/fp8_weights.go`

### Implement this boundary

Observed Koi E4M3 capability selects source FP8 cache/activation path. Go owns FC1 shuffle and tile; conversion is a qualified narrow operation.

- Use E4M3FN and FC1 order [0,1,4,5,8,9,12,13,2,3,6,7,10,11,14,15]; FC1 [256,32,16,16], FC2 [32,128,16,16], no int8 scales.

### Read and reuse

- external/dorado-source/dorado/nn/TxModules.cpp:439-440,496-507,515-518

- S00 selected caches; add one bounded conversion-domain source observation if S00 lacks it. CPU conversion does not qualify CUDA.

### Pass these gates

- When selected, all 65,536 half conversion payloads plus axis/genuine caches match source GPU. When unavailable, close only with pinned capability-false scope evidence, never a skipped FP8 pass. Reopen on profile change.
- Pin exact source and inputs, command clocks/exits/raw logs, consumer code and required test names. No required test may skip. Audit exact live bytes and source-declared initialized regions; keep uninitialized storage exclusions prospective.

### Reject these failures

- Wrong FP8 format, saturation mutant, missing shuffle, rounded source before cast or bogus scale attachment.

### Run or supply these commands

- to-implement. `go run ./verification/supgpu/cmd -unit fp8-weights -verify-consumer CAPTURE -backend BACKEND -runtime-admitted -out NEW_ATTEMPT`

Independent review. A fresh pstack_verifier audits the exact delta, raw receipts and one effective negative control. Source-only evidence cannot close native acceptance. Coordinate shared-file changes with the ABI or caller integration owner.

<a id="s06"></a>

## S06. Tile activations and refresh initial int8 storage

State is **planned**. Owner role is `pstack_worker; pstack_deep for arithmetic or lifetime work`. Dependencies are S02, P02, S00, I01. Resources are source, gpu-capture.

### Own these files

- `internal/model/supgpu/layout.go`
- `internal/model/supgpu/layout_test.go`
- `verification/supgpu/layout.go`

### Implement this boundary

Typed fixed/VCS activation contract separates valid lengths from padded rows and owns independent half/I8/scales representations.

- Fixed half tiles to [N,T/16,64,16,8], I8 to [N,T/16,32,16,16]. VCS starts [64,M,8], encoder T64 and pads N*T to multiple256; preserve distinct permutations.

### Read and reuse

- external/dorado-source/dorado/nn/TxModules.cpp:840-905,912-916

- S00 source activations and I01 geometry; QKV arithmetic alone does not establish activation rows.

### Pass these gates

- All live storage/scales and source padding match S00 for fixed and mixed VCS tails; padded slots never become real work.
- Pin exact source and inputs, command clocks/exits/raw logs, consumer code and required test names. No required test may skip. Audit exact live bytes and source-declared initialized regions; keep uninitialized storage exclusions prospective.

### Reject these failures

- Wrong-axis quantization, fixed reshape on VCS, omitted reciprocal, undersized arena or swapped untile axes.

### Run or supply these commands

- to-implement. `go run ./verification/supgpu/cmd -unit layout -verify-consumer CAPTURE -backend BACKEND -runtime-admitted -out NEW_ATTEMPT`

Independent review. A fresh pstack_verifier audits the exact delta, raw receipts and one effective negative control. Source-only evidence cannot close native acceptance. Coordinate shared-file changes with the ABI or caller integration owner.

<a id="s07"></a>

## S07. Execute tagged QKV projection and rotary

State is **planned**. Owner role is `pstack_worker; pstack_deep for arithmetic or lifetime work`. Dependencies are P02, S00, S03, S04. Resources are source, native-build, gpu-capture.

### Own these files

- `internal/model/supgpu/qkv.go`
- `internal/model/supgpu/qkv_test.go`
- `verification/tensor/sup_qkv.inc`
- `verification/supgpu/qkv.go`

### Implement this boundary

One typed koi_qkv_rotary primitive takes frozen tiled descriptors/caches, scale owners, counter and optional VCS RoPE LUT.

- Return half [N,T/16,3,8,8,16,8] with exact dimension tags. Fixed LUT is null; VCS uses scheduler-owned LUT. Check Koi return, input preservation and drain.

### Read and reuse

- external/dorado-source/dorado/nn/TxModules.cpp:533-565; koi.h:164-170

- S00 per-layer input/QKV stages and accepted half QKV preparation; no transformer producer rerun.

### Pass these gates

- Complete live output, descriptor metadata and counter guards match retained S00 stage; one genuine imported operand and corrupt metadata control pass.
- Pin exact source and inputs, command clocks/exits/raw logs, consumer code and required test names. No required test may skip. Audit exact live bytes and source-declared initialized regions; keep uninitialized storage exclusions prospective.

### Reject these failures

- Wrong dimension/scale tag, stale scales, wrong interleave, foreign device, released cache or uninitialized counter.

### Run or supply these commands

- to-implement. `go run ./verification/supgpu/cmd -unit qkv -verify-consumer CAPTURE -backend BACKEND -runtime-admitted -out NEW_ATTEMPT`

Independent review. A fresh pstack_verifier audits the exact delta, raw receipts and one effective negative control. Source-only evidence cannot close native acceptance. Coordinate shared-file changes with the ABI or caller integration owner.

<a id="s08"></a>

## S08. Execute fixed and variable-window attention

State is **planned**. Owner role is `pstack_worker; pstack_deep for arithmetic or lifetime work`. Dependencies are P02, S00, I01. Resources are source, native-build, gpu-capture.

### Own these files

- `internal/model/supgpu/attention.go`
- `internal/model/supgpu/attention_test.go`
- `verification/tensor/sup_attention.inc`
- `verification/supgpu/attention.go`

### Implement this boundary

Named koi_masked_attention or koi_vcs_attn returns half [N,T/16,8,8,16,8]; Go selects the source operation and exact 127/128 windows.

- Consume retained QKV operands directly, independent of candidate QKV implementation. Enforce chunk boundaries, head layout and valid VCS extents.

### Read and reuse

- external/dorado-source/dorado/nn/TxModules.cpp:568-575; koi.h:76-80,174-178

- S00 QKV and attention stage captures; CPU split SDPA is comparison context, not oracle.

### Pass these gates

- Boundary and adjacent mixed-length chunks match every retained live output. Padded/cross-chunk values cannot affect valid attention.
- Pin exact source and inputs, command clocks/exits/raw logs, consumer code and required test names. No required test may skip. Audit exact live bytes and source-declared initialized regions; keep uninitialized storage exclusions prospective.

### Reject these failures

- Swap upper/lower window, cross-chunk access, changed start/length, wrong head flattening or poisoned padding.

### Run or supply these commands

- to-implement. `go run ./verification/supgpu/cmd -unit attention -verify-consumer CAPTURE -backend BACKEND -runtime-admitted -out NEW_ATTEMPT`

Independent review. A fresh pstack_verifier audits the exact delta, raw receipts and one effective negative control. Source-only evidence cannot close native acceptance. Coordinate shared-file changes with the ABI or caller integration owner.

<a id="s09"></a>

## S09. Execute projection and FC2 linear operations

State is **planned**. Owner role is `pstack_worker; pstack_deep for arithmetic or lifetime work`. Dependencies are P02, S00, S04, S05. Resources are source, native-build, gpu-capture.

### Own these files

- `internal/model/supgpu/linear.go`
- `internal/model/supgpu/linear_test.go`
- `verification/tensor/sup_linear.inc`
- `verification/supgpu/linear.go`

### Implement this boundary

One typed koi_linear operation implements both projection and FC2 using source tagged operands; projection has bias and FC2 has null bias.

- Support selected half/FP8 FC2 layout, exact tile tags and counters. Primitive acceptance consumes source operands, so it does not depend cyclically on candidate attention or SwiGLU.

### Read and reuse

- external/dorado-source/dorado/nn/TxModules.cpp:577-582,599-604; koi.h:145

- S00 source attention/SwiGLU inputs and linear outputs; no graph forward repeated.

### Pass these gates

- Projection and selected FC2 family match S00 independently. Default FP8 profile cannot pass using only half operands.
- Pin exact source and inputs, command clocks/exits/raw logs, consumer code and required test names. No required test may skip. Audit exact live bytes and source-declared initialized regions; keep uninitialized storage exclusions prospective.

### Reject these failures

- Omit projection bias, add FC2 bias, half metadata for FP8 bytes, wrong tile size, forbidden alias or dirty counter.

### Run or supply these commands

- to-implement. `go run ./verification/supgpu/cmd -unit linear -verify-consumer CAPTURE -backend BACKEND -runtime-admitted -out NEW_ATTEMPT`

Independent review. A fresh pstack_verifier audits the exact delta, raw receipts and one effective negative control. Source-only evidence cannot close native acceptance. Coordinate shared-file changes with the ABI or caller integration owner.

<a id="s10"></a>

## S10. Execute residual RMSNorm and precision refresh

State is **planned**. Owner role is `pstack_worker; pstack_deep for arithmetic or lifetime work`. Dependencies are P02, S00, S04. Resources are source, native-build, gpu-capture.

### Own these files

- `internal/model/supgpu/rms.go`
- `internal/model/supgpu/rms_test.go`
- `verification/tensor/sup_rms.inc`
- `verification/supgpu/rms.go`

### Implement this boundary

Named koi_rmsnorm_residual updates half residual in place and produces precisely the selected optional representation and scales.

- Preserve float32 DeepNorm alpha bits. Norm1 emits FP8 if selected, otherwise I8; norm2 emits I8 and never requests FP8. Define all intentional aliases and retain owners.

### Read and reuse

- external/dorado-source/dorado/nn/TxModules.cpp:584-588,607-611; koi.h:155-162

- S00 norm inputs/results, not candidate linear output; no execution dependency cycle with S09/S11.

### Pass these gates

- Both norm roles match complete changed half/quantized/scales storage and guards from S00, using captured source operands independently.
- Pin exact source and inputs, command clocks/exits/raw logs, consumer code and required test names. No required test may skip. Audit exact live bytes and source-declared initialized regions; keep uninitialized storage exclusions prospective.

### Reject these failures

- Wrong alpha placement, stale I8 scales, FP8 on norm2, omitted in-place update, unrounded norm weight or early release.

### Run or supply these commands

- to-implement. `go run ./verification/supgpu/cmd -unit rms -verify-consumer CAPTURE -backend BACKEND -runtime-admitted -out NEW_ATTEMPT`

Independent review. A fresh pstack_verifier audits the exact delta, raw receipts and one effective negative control. Source-only evidence cannot close native acceptance. Coordinate shared-file changes with the ABI or caller integration owner.

<a id="s11"></a>

## S11. Execute selected SwiGLU arithmetic

State is **planned**. Owner role is `pstack_worker; pstack_deep for arithmetic or lifetime work`. Dependencies are P02, S00, S04, S05. Resources are source, native-build, gpu-capture.

### Own these files

- `internal/model/supgpu/swiglu.go`
- `internal/model/supgpu/swiglu_test.go`
- `verification/tensor/sup_swiglu.inc`
- `verification/supgpu/swiglu.go`

### Implement this boundary

Named koi_mm_swiglu consumes selected FP8 or I8 activation/cache. Go fixes gate order and source flags; f32_accum=false and uswiglu=false by default.

- FP8 output [N*T/16,128,16,16] and non-FP8 half [N*T/16,256,16,8] represent 2048 values per position. Use source norm1 operands for independent primitive acceptance.

### Read and reuse

- external/dorado-source/dorado/nn/TxModules.cpp:590-597; koi.h:147-153

- S00 norm1/SwiGLU stage captures; primitive can progress beside norm/linear implementation.

### Pass these gates

- Selected representation, output bits, counter and input guards match S00. Unselected branch is labeled conditional rather than a silent substitute.
- Pin exact source and inputs, command clocks/exits/raw logs, consumer code and required test names. No required test may skip. Audit exact live bytes and source-declared initialized regions; keep uninitialized storage exclusions prospective.

### Reject these failures

- Reverse gate/value, omit FP8 shuffle, force other precision, change developer flags or reinterpret output dtype.

### Run or supply these commands

- to-implement. `go run ./verification/supgpu/cmd -unit swiglu -verify-consumer CAPTURE -backend BACKEND -runtime-admitted -out NEW_ATTEMPT`

Independent review. A fresh pstack_verifier audits the exact delta, raw receipts and one effective negative control. Source-only evidence cannot close native acceptance. Coordinate shared-file changes with the ABI or caller integration owner.

<a id="s12"></a>

## S12. Execute five SUP convolutions and VCS LUTs

State is **planned**. Owner role is `pstack_worker; pstack_deep for arithmetic or lifetime work`. Dependencies are P02, S00, I01. Resources are source, native-build, gpu-capture.

### Own these files

- `internal/model/supgpu/conv.go`
- `internal/model/supgpu/conv_test.go`
- `verification/tensor/sup_conv.inc`
- `verification/supgpu/conv.go`

### Implement this boundary

Go owns five conv calls, weight tilings, per-invocation granularity and LUT-fill scheduling. Scheduler owns immutable batch membership/table.

- Channels 1,64,64,128,128,512; widths5,5,9,9,5; strides1,1,3,2,2. Fixed uses actual CUDA ATen conv+SiLU; VCS uses koi_vcs_tx_fill_conv_load_store_lut and koi_vcs_tx_cnn. Preserve first padding, special tiles and half accumulation defaults.

### Read and reuse

- external/dorado-source/dorado/nn/ConvStack.cpp:160-185,376-457; basecall/CudaModelRunner.cpp:17-40

- S00 conv stages, source half weights and I01 geometry; half GEMM evidence alone cannot qualify CUDA conv1d.

### Pass these gates

- Every layer matches S00 for fixed and mixed VCS, including first-layer zero padding, last-layer size and consecutive-batch LUT reset.
- Pin exact source and inputs, command clocks/exits/raw logs, consumer code and required test names. No required test may skip. Audit exact live bytes and source-declared initialized regions; keep uninitialized storage exclusions prospective.

### Reject these failures

- Corrupt/reuse LUT, omit stride update or half-window padding, wrong final tile, change accumulation mode or cross-chunk contamination.

### Run or supply these commands

- to-implement. `go run ./verification/supgpu/cmd -unit conv -verify-consumer CAPTURE -backend BACKEND -runtime-admitted -out NEW_ATTEMPT`

Independent review. A fresh pstack_verifier audits the exact delta, raw receipts and one effective negative control. Source-only evidence cannot close native acceptance. Coordinate shared-file changes with the ABI or caller integration owner.

<a id="s13"></a>

## S13. Execute upsampling and half-scaled CRF projection

State is **planned**. Owner role is `pstack_worker; pstack_deep for arithmetic or lifetime work`. Dependencies are P02, S00. Resources are source, native-build, gpu-capture.

### Own these files

- `internal/model/supgpu/head.go`
- `internal/model/supgpu/head_test.go`
- `verification/supgpu/head.go`
- `verification/tensor/sup_head.inc`

### Implement this boundary

Go schedules half linear+bias, reshape [N,T,1024] to [N,2*T,512], and bias-free 4096-channel CRF projection after one in-place half weight scaling.

- Scale already-half device CRF weight by five once. Define cache initialization state and concurrency so repeated forwards cannot rescale.

### Read and reuse

- external/dorado-source/dorado/nn/LinearUpsample.cpp:17-22; nn/TxModules.cpp:952-957

- S00 final encoder/head/weight captures and 139 weight pins. CPU float32 scaling is not this endpoint.

### Pass these gates

- Scaled weight bits, both head stages and final scores match S00. Two invocations expose repeated-scaling and mutable-cache bugs.
- Pin exact source and inputs, command clocks/exits/raw logs, consumer code and required test names. No required test may skip. Audit exact live bytes and source-declared initialized regions; keep uninitialized storage exclusions prospective.

### Reject these failures

- Float32 scale-then-cast, scale twice, T/C reshape error, omitted upsample bias, added CRF bias or HAC channel count.

### Run or supply these commands

- to-implement. `go run ./verification/supgpu/cmd -unit head -verify-consumer CAPTURE -backend BACKEND -runtime-admitted -out NEW_ATTEMPT`

Independent review. A fresh pstack_verifier audits the exact delta, raw receipts and one effective negative control. Source-only evidence cannot close native acceptance. Coordinate shared-file changes with the ABI or caller integration owner.

<a id="s14"></a>

## S14. Compose all 18 SUP encoders in Go

State is **planned**. Owner role is `pstack_worker; pstack_deep for arithmetic or lifetime work`. Dependencies are S02, S03, S04, S05, S06, S07, S08, S09, S10, S11, S12, S13. Resources are source, gpu-capture.

### Own these files

- `internal/model/supgpu/graph.go`
- `internal/model/supgpu/graph_test.go`
- `verification/supgpu/graph.go`

### Implement this boundary

Go owns five convs, 18 seven-call encoder loops, precision/cache/layout transitions, upsampling and CRF head. Native whole-graph wrappers remain absent from production.

- Bind all139 weights; use four fresh zero I32 counters per layer and invocation. Norm2 feeds next layer; preserve half residual alias and selected FP8. Fixed and default VCS are distinct required cases.

### Read and reuse

- external/dorado-source/dorado/basecall/model/TxModel.cpp:20-47; nn/TxModules.cpp:558-611,837-916

- One S00 source graph per declared case; all individual primitive/caches receipts. Host import never executes GPU; verify-consumer is explicitly native-admitted.

### Pass these gates

- Every source-declared stage and final half score tensor matches S00. Repeated/in-flight invocations do not share mutable counters/scratch; close/cancel drains.
- Pin exact source and inputs, command clocks/exits/raw logs, consumer code and required test names. No required test may skip. Audit exact live bytes and source-declared initialized regions; keep uninitialized storage exclusions prospective.

### Reject these failures

- Layer/cache swap, disabling default I8/FP8, forcing fixed on default VCS, native model symbol, stale counters or cross-stream alias.

### Run or supply these commands

- to-implement. `go run ./verification/supgpu/cmd -unit graph -verify-consumer CAPTURE -backend BACKEND -runtime-admitted -out NEW_ATTEMPT`

Independent review. A fresh pstack_verifier audits the exact delta, raw receipts and one effective negative control. Source-only evidence cannot close native acceptance. Coordinate shared-file changes with the ABI or caller integration owner.

<a id="d01"></a>

## D01. Resolve and capture the opaque GPU decoder contract

State is **planned**. Owner role is `pstack_worker; pstack_deep for arithmetic or lifetime work`. Dependencies are P01, P02, P05, H01, S00. Resources are source, native-build, gpu-capture.

### Own these files

- `verification/gpudecode/cmd/main.go`
- `verification/gpudecode/capture.go`
- `verification/gpudecode/schema.go`
- `verification/gpudecode/native/oracle.cpp`
- `docs/gpu-decoder-contract.md`

### Implement this boundary

Four Koi entrypoints are staged reference oracles only. Freeze typed score/chunk buffers and resolve scratch/result semantics, beam ties, posterior and quality behavior before Go replacement decisions.

- Use retained model half scores and synthetic all-stay/tie/near-tie/blank/tail controls for HAC256 and SUP1024 states. Snapshot each of the four calls. Obtain additional implementation source or targeted tracing when captures underdetermine semantics; no guessed float layout for byte scratch.

### Read and reuse

- external/dorado-source/dorado/basecall/decode/CUDADecoder.cpp:20-197; include/basecall/decode/CUDADecoder.h; pinned koi.h:469-544

- H01/S00 complete scores, existing CPU beam fixtures as context and docs/cuda-kernel-trace.md. No model inference replay to debug decoding.

### Pass these gates

- Independent review traces every required semantic rule to source or discriminating observation. A genuine score handoff and corruption rejection pass. Unresolved ties/Q/layout are an explicit research blocker, with no promised completion date.
- Pin exact source and inputs, command clocks/exits/raw logs, consumer code and required test names. No required test may skip. Audit exact live bytes and source-declared initialized regions; keep uninitialized storage exclusions prospective.

### Reject these failures

- Changed score stride/dtype, chunk table, clamp/beam/Q parameters, truncated scratch, inferred mismatch mask or opaque whole-decoder production wrapper.

### Run or supply these commands

- to-implement. `go run ./verification/gpudecode/cmd capture -plan PLAN -scores RETAINED_SCORES -runtime-admitted -out NEW_ATTEMPT`

Independent review. A fresh pstack_verifier audits the exact delta, raw receipts and one effective negative control. Source-only evidence cannot close native acceptance. Coordinate shared-file changes with the ABI or caller integration owner.

<a id="d02"></a>

## D02. Implement Go-owned guide and posterior scans

State is **planned**. Owner role is `pstack_worker; pstack_deep for arithmetic or lifetime work`. Dependencies are D01, P02. Resources are source, native-build, gpu-capture.

### Own these files

- `internal/gpudecode/scans.go`
- `internal/gpudecode/scans_test.go`
- `verification/gpudecode/scans.go`
- `verification/tensor/gpu_crf_scans.inc`

### Implement this boundary

Typed logical chunk extents, guides and posterior owners use Go-owned scan order/state transitions and individually reviewed arithmetic kernels.

- Preserve source dtype, accumulation order, clamp, blank, direction and fixed/packed indexing. Expose only semantics established by D01; native primitive boundary needs review.

### Read and reuse

- external/dorado-source/dorado/basecall/decode/CUDADecoder.cpp:95-126; internal/crf/crf.go for CPU context only; D01 resolved contract

- D01 scores/stage captures, accepted CPU scan controls; no repeated model or decoder oracle for consumer fixes.

### Pass these gates

- Every declared valid guide/posterior intermediate matches D01. If an intermediate remains opaque, close only after a reviewed discriminating equivalence contract, not an invented layout.
- Pin exact source and inputs, command clocks/exits/raw logs, consumer code and required test names. No required test may skip. Audit exact live bytes and source-declared initialized regions; keep uninitialized storage exclusions prospective.

### Reject these failures

- Reverse-index mutant, wrong blank/clamp, cross-chunk access, changed stride, stale scratch or overflowing capacity.

### Run or supply these commands

- to-implement. `go run ./verification/gpudecode/cmd verify-consumer -stage scans -manifest CAPTURE -backend BACKEND -runtime-admitted -out NEW_ATTEMPT`

Independent review. A fresh pstack_verifier audits the exact delta, raw receipts and one effective negative control. Source-only evidence cannot close native acceptance. Coordinate shared-file changes with the ABI or caller integration owner.

<a id="d03"></a>

## D03. Implement Go-owned GPU beam decisions

State is **planned**. Owner role is `pstack_worker; pstack_deep for arithmetic or lifetime work`. Dependencies are D01, P02. Resources are source, native-build, gpu-capture.

### Own these files

- `internal/gpudecode/beam.go`
- `internal/gpudecode/beam_test.go`
- `verification/gpudecode/beam.go`
- `verification/tensor/gpu_beam_arithmetic.inc`

### Implement this boundary

Go owns pruning, merging, ties, clamp and terminal decisions from typed scores/guides; GPU arithmetic may use narrowly reviewed primitives. Calling opaque host_beam_search_step is not completion.

- Use D01 source guides directly for independent implementation and verification. Encode discovered GPU rules without assuming the public CPU beam is equivalent.

### Read and reuse

- external/dorado-source/dorado/basecall/decode/CUDADecoder.cpp:95-126; internal/decode for comparison context; D01 semantic evidence

- D01 retained synthetic and genuine score/guide fixtures. D02 candidate need not finish before source-operand beam work.

### Pass these gates

- Exact path/move decisions pass tied transitions, merge collisions, beam-width/cut boundaries, zero-base and genuine HAC/SUP fixtures. Underdetermined behavior blocks closure.
- Pin exact source and inputs, command clocks/exits/raw logs, consumer code and required test names. No required test may skip. Audit exact live bytes and source-declared initialized regions; keep uninitialized storage exclusions prospective.

### Reject these failures

- Tie-order, merge-identity, beam-cut/width or score-clamp mutant; whole native beam stage hidden behind Go method.

### Run or supply these commands

- to-implement. `go run ./verification/gpudecode/cmd verify-consumer -stage beam -manifest CAPTURE -backend BACKEND -runtime-admitted -out NEW_ATTEMPT`

Independent review. A fresh pstack_verifier audits the exact delta, raw receipts and one effective negative control. Source-only evidence cannot close native acceptance. Coordinate shared-file changes with the ABI or caller integration owner.

<a id="d04"></a>

## D04. Implement GPU final quality, moves and unpacking

State is **planned**. Owner role is `pstack_worker; pstack_deep for arithmetic or lifetime work`. Dependencies are D01, P02. Resources are source, gpu-capture.

### Own these files

- `internal/gpudecode/quality.go`
- `internal/gpudecode/unpack.go`
- `internal/gpudecode/decoder.go`
- `verification/gpudecode/final.go`

### Implement this boundary

Go finalizes source-established posterior/Q behavior and returns independently owned sequence, quality and moves with explicit valid length.

- Copy exactly valid T moves and sum(moves) sequence/Q bytes. Preserve source Q scale/shift/clamp and move_pad; check capacity before slices and allocations. Compose the four replacement stages through typed inputs.

### Read and reuse

- external/dorado-source/dorado/basecall/decode/CUDADecoder.cpp:136-197; D01 resolved quality and result-field contract

- D01 stage outputs and existing Go decode result ownership; do not use CPU Q math as unobserved GPU truth.

### Pass these gates

- Exact sequence/Q/moves pass both models, zero emissions, packed/padded tails and rounding boundaries against D01; no opaque whole-decoder production call.
- Pin exact source and inputs, command clocks/exits/raw logs, consumer code and required test names. No required test may skip. Audit exact live bytes and source-declared initialized regions; keep uninitialized storage exclusions prospective.

### Reject these failures

- Wrong Q shift/scale, rounding mutant, off-by-one tail, nonbinary move, stale padded bytes or move sum over capacity.

### Run or supply these commands

- to-implement. `go run ./verification/gpudecode/cmd verify-consumer -stage final -manifest CAPTURE -backend BACKEND -runtime-admitted -out NEW_ATTEMPT`

Independent review. A fresh pstack_verifier audits the exact delta, raw receipts and one effective negative control. Source-only evidence cannot close native acceptance. Coordinate shared-file changes with the ABI or caller integration owner.

<a id="i01"></a>

## I01. Freeze fixed and packed batch geometry

State is **planned**. Owner role is `pstack_worker; pstack_deep for arithmetic or lifetime work`. Dependencies are P01, P05. Resources are source, native-build, gpu-capture.

### Own these files

- `internal/gpubatch/batch.go`
- `internal/gpubatch/aux.go`
- `internal/gpubatch/batch_test.go`
- `verification/gpubatch/native/oracle.cpp`
- `verification/gpubatch/oracle.go`
- `verification/gpubatch/oracle_test.go`

### Implement this boundary

One owner defines real chunk identity, raw valid lengths, packed membership, batch padding, chunk table, intervals, LUT owners and granularity transitions consumed by both model lanes.

- HAC physical granularity64, SUP32; SUP raw granularity768 and first-conv half-window2 zero padding. Preserve chunk table [start_block,length_blocks], source stride_out scaling and encoder T64/M multiple256. Model kernels mutate only local invocation granularity.

### Read and reuse

- external/dorado-source/dorado/basecall/CudaModelRunner.cpp:17-59; nn/AuxiliaryData.cpp:24-61,129-150; nn/TxModules.cpp:840-864

- Accepted chunk planners and normalized signals. Scheduler owns shared formulas; SUP convolution owns layer-specific LUT fill and arithmetic.

### Pass these gates

- Pure geometry matches an extracted source oracle for fixed, mixed lengths, minimum/tail and two consecutive batches; actual uploaded tables match declared bytes after P02 binding.
- Pin exact source and inputs, command clocks/exits/raw logs, consumer code and required test names. No required test may skip. Audit exact live bytes and source-declared initialized regions; keep uninitialized storage exclusions prospective.

### Reject these failures

- Treat padding as work, use wrong units or stride, reuse previous table, bad SUP boundary zeros or integer overflow.

### Run or supply these commands

- to-implement. `go test -json -count=1 ./internal/gpubatch ./verification/gpubatch`

Independent review. A fresh pstack_verifier audits the exact delta, raw receipts and one effective negative control. Source-only evidence cannot close native acceptance. Coordinate shared-file changes with the ABI or caller integration owner.

<a id="i02"></a>

## I02. Join GPU chunk results to source read assembly

State is **planned**. Owner role is `pstack_worker; pstack_deep for arithmetic or lifetime work`. Dependencies are I01, D02, D03, D04. Resources are source.

### Own these files

- `internal/caller/gpu_assembly.go`
- `internal/caller/gpu_assembly_test.go`
- `verification/gpucaller/assembly.go`

### Implement this boundary

DecodedWork carries read/chunk identity, raw offset/length, valid score length, padding granularity and owned sequence/Q/moves. Collector restores source order before existing Stitch.

- Integrate fixed, HAC-balanced-variable and SUP-variable tails using existing planners/stitch. Keep sample, score and base coordinates distinct; SUP conv zeros are not signal.

### Read and reuse

- external/dorado-source/dorado/read_pipeline/base/{chunk,stitch}.cpp; read_pipeline/nodes/BasecallerNode.cpp:420-491; internal/chunk; internal/stitch

- Accepted 726 stitch/source cases and planner fixtures; D01 decoded source captures. Do not rewrite accepted stitch logic.

### Pass these gates

- Captured GPU decoded chunks reproduce source assembly with exact moves, sequence/Q and coverage. Completion permutation does not change deterministic output.
- Pin exact source and inputs, command clocks/exits/raw logs, consumer code and required test names. No required test may skip. Audit exact live bytes and source-declared initialized regions; keep uninitialized storage exclusions prospective.

### Reject these failures

- Missing/duplicate/cross-read chunk, shuffled completion without reorder, half-overlap error, partial-stride error, stale padding or premature stitch.

### Run or supply these commands

- to-implement. `go test -json -count=1 ./internal/caller -run '^TestGPUAssembly'`

Independent review. A fresh pstack_verifier audits the exact delta, raw receipts and one effective negative control. Source-only evidence cannot close native acceptance. Coordinate shared-file changes with the ABI or caller integration owner.

<a id="i03"></a>

## I03. Schedule bounded GPU batches and safe drain

State is **planned**. Owner role is `pstack_worker; pstack_deep for arithmetic or lifetime work`. Dependencies are I01, P02. Resources are source, gpu-capture.

### Own these files

- `internal/scheduler/gpu.go`
- `internal/scheduler/gpu_test.go`
- `verification/gpucaller/schedule.go`

### Implement this boundary

Bounded queues separate real work from padded capacity. Invocation owners retain all views, caches, scratch and stream resources through completion and drain.

- Freeze explicit batch rounding plus automatic memory/batch-selection policy if supported by P01; otherwise reject automatic requests. Preserve deterministic membership and no silent CPU/path fallback. Keep synchronous primitives initially; add async events only if a predeclared endpoint requires them.

### Read and reuse

- external/dorado-source/dorado/basecall/CudaCaller.cpp:211-214,246-289,477-505,681-748; CudaModelRunner.cpp; internal/tensor/cuda.go

- Accepted CUDA lifetime/drain behavior and source batch policy. Do not infer performance from concurrency.

### Pass these gates

- Model-free event tests plus genuine stream cases show bounded memory, no deadlock/UAF and complete error propagation. Shutdown/cancellation never reports partial batch success.
- Pin exact source and inputs, command clocks/exits/raw logs, consumer code and required test names. No required test may skip. Audit exact live bytes and source-declared initialized regions; keep uninitialized storage exclusions prospective.

### Reject these failures

- OOM before enqueue, error after enqueue, queued/running cancel, partial flush, empty input, repeated close, close during work or reused scratch.

### Run or supply these commands

- to-implement. `go run ./verification/gpucaller/cmd schedule-check -plan PLAN -runtime-admitted -out NEW_ATTEMPT`

Independent review. A fresh pstack_verifier audits the exact delta, raw receipts and one effective negative control. Source-only evidence cannot close native acceptance. Coordinate shared-file changes with the ABI or caller integration owner.

<a id="i04"></a>

## I04. Account for every requested parent and emitted child

State is **planned**. Owner role is `pstack_worker; pstack_deep for arithmetic or lifetime work`. Dependencies are none. Resources are source, cpu-heavy.

### Own these files

- `internal/runledger/ledger.go`
- `internal/runledger/ledger_test.go`
- `verification/gpucaller/inventory.go`

### Implement this boundary

Typed run ledger records absent/rejected/loaded/failed/zero-survivor/child-bearing parent outcomes, raw/admitted samples, chunks, lineage, filter reasons, writes and finalization.

- Reconcile physical file UUIDs, requested selections and output QNAME/pi lineage independently of progress text. Reader integration patches go through I07; do not rewrite existing selection semantics silently.

### Read and reuse

- internal/pod5/reader.go: IDs,Metadata,Read,Iterate; internal/caller/process.go; original source-qualification parent and requested-ID ledgers

- Retained HG002 development/validation inventories and POD5 fixtures; no HG005 inspection for selection.

### Pass these gates

- Every requested parent has exactly one final disposition and inventories reconcile actual input IDs, samples and output files. Zero surviving children still represents processed work.
- Pin exact source and inputs, command clocks/exits/raw logs, consumer code and required test names. No required test may skip. Audit exact live bytes and source-declared initialized regions; keep uninitialized storage exclusions prospective.

### Reject these failures

- Missing selected ID, duplicate file UUID, corrupt row, unsupported metadata, cancellation, renamed child, all-filtered output or writer failure.

### Run or supply these commands

- to-implement. `go test -json -count=1 ./internal/runledger`

Independent review. A fresh pstack_verifier audits the exact delta, raw receipts and one effective negative control. Source-only evidence cannot close native acceptance. Coordinate shared-file changes with the ABI or caller integration owner.

<a id="i05"></a>

## I05. Reuse default postcall with GPU metadata and replay

State is **planned**. Owner role is `pstack_worker; pstack_deep for arithmetic or lifetime work`. Dependencies are I02, I04. Resources are source, gpu-capture.

### Own these files

- `verification/gpucaller/postcall.go`
- `internal/caller/gpu_metadata.go`
- `internal/caller/gpu_capture.go`

### Implement this boundary

Feed GPU-assembled records, original normalized signal and metadata into existing postcall policy. Versioned GPU captures identify runtime, layout, decoder and policy without CPU reinterpretation.

- Keep MUX always active, split children/clocks, quality prefix reset, intervals, mean-Q/filter and atomic publication. Add DS only under source predicate. Test all four trim/split modes. Shared process.go edits go through I07.

### Read and reuse

- internal/caller/{process,mux,processed_capture,fastq}.go; internal/postcall/postcall.go; external/dorado-source read tag generation

- Accepted default CPU controls and publication fix; accepted source split/trim/filter fixtures. Reuse GPU core capture; no new inference for postcall fixes.

### Pass these gates

- Producer-free GPU capture replay matches actual source default headers/records in each mode. Genuine positive split/MUX coverage is obtained only if adequate retained controls are absent.
- Pin exact source and inputs, command clocks/exits/raw logs, consumer code and required test names. No required test may skip. Audit exact live bytes and source-declared initialized regions; keep uninitialized storage exclusions prospective.

### Reject these failures

- Wrong child clock, DS tag, leading trim, primer orientation, mean-Q offset, policy, CPU-as-GPU capture, payload hash or final-manifest write/sync failure.

### Run or supply these commands

- to-implement. `go run ./verification/gpucaller/cmd postcall-check -capture CAPTURE -modes all -out NEW_ATTEMPT`

Independent review. A fresh pstack_verifier audits the exact delta, raw receipts and one effective negative control. Source-only evidence cannot close native acceptance. Coordinate shared-file changes with the ABI or caller integration owner.

<a id="i06"></a>

## I06. Write unaligned BAM and exact FASTQ contracts

State is **planned**. Owner role is `pstack_worker; pstack_deep for arithmetic or lifetime work`. Dependencies are I04, I05. Resources are source, native-build, cpu-heavy.

### Own these files

- `internal/bam/writer.go`
- `internal/bam/writer_test.go`
- `verification/bam/cmd/main.go`
- `verification/bam/tags_oracle.cpp`

### Implement this boundary

Typed final-record contract preserves unaligned mandatory fields, read groups, source tags, model/device/scaling, samples/moves, parent identity and split offsets. FASTQ tag policy remains distinct.

- Implement Go-owned serialization policy and predeclare deterministic order/header/PG/compression topology. Use a narrowly pinned BGZF primitive only if explicitly recorded. Propagate short-write, close and broken-pipe errors.

### Read and reuse

- external/dorado-source/dorado/read_pipeline/base/messages.cpp; nodes/ReadToBamTypeNode.cpp; hts_utils/hts_file.cpp; HtsFileWriter/StreamHtsFileWriter; cmd/readcompare/README.md

- Existing canonical comparator and retained source BAMs/tag fixtures. Comparator self-equality does not establish a Go writer.

### Pass these gates

- samtools quickcheck, exact FASTQ, mandatory/tag/canonical record, decoded header and raw BAM verdicts all exist separately. If raw identity is required, differing bytes block that claim even with canonical equality.
- Pin exact source and inputs, command clocks/exits/raw logs, consumer code and required test names. No required test may skip. Audit exact live bytes and source-declared initialized regions; keep uninitialized storage exclusions prospective.

### Reject these failures

- Record/tag reorder canonical-positive/raw-negative, changed tag, duplicate/missing record, header-only mutation, truncated BGZF/EOF, ENOSPC and short write.

### Run or supply these commands

- to-implement. `go run ./verification/bam/cmd verify -plan PLAN -attempt ATTEMPT -out NEW_ATTEMPT`

Independent review. A fresh pstack_verifier audits the exact delta, raw receipts and one effective negative control. Source-only evidence cannot close native acceptance. Coordinate shared-file changes with the ABI or caller integration owner.

<a id="i07"></a>

## I07. Integrate the supported GPU caller and CLI

State is **planned**. Owner role is `pstack_worker; pstack_deep for arithmetic or lifetime work`. Dependencies are P01, P02, H09, S14, D02, D03, D04, I02, I03, I04, I05, I06. Resources are source, gpu-capture.

### Own these files

- `cmd/basecaller/main.go`
- `internal/caller/caller.go`
- `internal/caller/process.go`
- `internal/caller/processed_capture.go`
- `internal/caller/fastq.go`
- `internal/model/model.go`
- `internal/pod5/reader.go`
- `verification/gpucaller/integration.go`

### Implement this boundary

One caller integrator wires reviewed GPU model/decoder/scheduler/ledger/output contracts into production. Existing CPU calls and retained replay remain supported.

- Add device/output/batch/VCS options only from P01 supported matrix. Resolve default options exactly. No whole-model/caller subprocess, no silent CPU fallback, and replay must initialize no model/POD5/GPU.

### Read and reuse

- cmd/basecaller/main.go currently rejects non-CPU; internal/caller/caller.go callStage/infer/configuration; docs/caller.md

- All accepted model/decoder stage captures, CPU/default controls and inventory contract; one new bounded end-to-end producer only after genuine replay handshake.

### Pass these gates

- Real CLI smoke matrix covers HAC/SUP, fixed/default VCS, both formats, selections/full-file fixture and four trim/split modes. Complete child/parent/sample accounting and exact declared outputs pass.
- Pin exact source and inputs, command clocks/exits/raw logs, consumer code and required test names. No required test may skip. Audit exact live bytes and source-declared initialized regions; keep uninitialized storage exclusions prospective.

### Reject these failures

- Excluded model/chemistry/output/runtime/flag before inference, CPU capture presented as GPU, wrong default, truncated pipe, missing parent, duplicate child or final write failure.

### Run or supply these commands

- to-implement. `go run ./verification/gpucaller/cmd qualify -scope smoke -plan PLAN -runtime-admitted -out NEW_ATTEMPT`

Independent review. A fresh pstack_verifier audits the exact delta, raw receipts and one effective negative control. Source-only evidence cannot close native acceptance. Coordinate shared-file changes with the ABI or caller integration owner.

<a id="a00"></a>

## A00. Freeze final conformance, accuracy and timing protocols

State is **planned**. Owner role is `pstack_worker; pstack_deep for arithmetic or lifetime work`. Dependencies are none. Resources are source.

### Own these files

- `provenance/gpu-acceptance-plan.json`
- `docs/gpu-acceptance.md`
- `verification/gpuprotocol/cmd/main.go`
- `verification/gpuprotocol/check.go`
- `verification/gpuprotocol/check_test.go`

### Implement this boundary

Prospective endpoint matrix defines engineering, official/source bytes, canonical records, headers, artifact equality, biological/ALT, yield and measurement validity before final runs.

- Pin HG002 memberships, requested parents, flowcells, references/truth/masks, order and holdout rules. Predeclare ALT-loss failure and uncertainty. Performance acceptance requires valid equal-work matched complete-process measurement; no required speedup is invented. Any numerical speed budget must be frozen before clocks.

### Read and reuse

- docs/port-plan.md exactness contract; original research/{controlled-baseline-results,prospective-experiment,source-qualification-results}.md

- Existing stock/source parent ledgers, eight numerical score objects, GIAB domains and plans. Preserve original six literal row-order failures as historical FAIL.

### Pass these gates

- Independent protocol review passes genuine retained control ingestion plus corrupted input rejection before expensive cohorts. No endpoint switches after results; no new experiment is armed by this plan.
- Pin exact source and inputs, command clocks/exits/raw logs, consumer code and required test names. No required test may skip. Audit exact live bytes and source-declared initialized regions; keep uninitialized storage exclusions prospective.

### Reject these failures

- HG005 selection, reference-only surrogate for ALT, changed memberships/denominators, relaxed literal gate, posthoc tolerance or historical stock clocks labeled fresh matched data.

### Run or supply these commands

- to-implement. `go run ./verification/gpuprotocol/cmd -plan provenance/gpu-acceptance-plan.json`

Independent review. A fresh pstack_verifier audits the exact delta, raw receipts and one effective negative control. Source-only evidence cannot close native acceptance. Coordinate shared-file changes with the ABI or caller integration owner.

<a id="a01"></a>

## A01. Qualify complete HG002 development and validation inputs

State is **planned**. Owner role is `pstack_worker; pstack_deep for arithmetic or lifetime work`. Dependencies are A00, I07, P04. Resources are source, gpu-capture, full-cohort.

### Own these files

- `verification/gpucaller/cohort.go`
- `provenance/gpu-cohort-plan.json`
- `docs/gpu-cohort-results.md`

### Implement this boundary

Full supported target processes all frozen requested parents for both models, with exact work, lineage, final outputs and separate source13/official12 comparisons.

- Admit development first, then validation under frozen rules. Use existing controls read-only and one Go producer per declared condition. Broader mode combinations use bounded controls unless the prospective plan explicitly requires full cohorts.

### Read and reuse

- Original data/splits, config/assets.json, provenance/assets.lock.json and source-qualification full ledger; I07 supported matrix

- Retained 2,608 HG002 development and 2,027 validation parent memberships and official/source outputs per tier; verify current pins at consumption. No stock producer rerun just to reformat reports.

### Pass these gates

- Every requested parent/sample is accounted for and all required final artifact comparisons pass at exact head. Separate canonical, header, raw-byte and replay verdicts; preserve differences rather than calling overall pass.
- Pin exact source and inputs, command clocks/exits/raw logs, consumer code and required test names. No required test may skip. Audit exact live bytes and source-declared initialized regions; keep uninitialized storage exclusions prospective.

### Reject these failures

- Drop parent/read, count padding as work, duplicate split child, suppress producer failure, missing required mode, changed input or final artifact.

### Run or supply these commands

- to-implement. `go run ./verification/gpucaller/cmd qualify -scope full -plan provenance/gpu-cohort-plan.json -runtime-admitted -out NEW_ATTEMPT`

Independent review. A fresh pstack_verifier audits the exact delta, raw receipts and one effective negative control. Source-only evidence cannot close native acceptance. Coordinate shared-file changes with the ABI or caller integration owner.

<a id="a02"></a>

## A02. Verify biological equivalence and ALT retention

State is **planned**. Owner role is `pstack_worker; pstack_deep for arithmetic or lifetime work`. Dependencies are A00, A01. Resources are source, cpu-heavy, full-cohort.

### Own these files

- `verification/gpucaller/biological.go`
- `provenance/gpu-biological-plan.json`
- `docs/gpu-biological-results.md`

### Implement this boundary

A candidate-specific reviewed adapter coordinates existing native aligners and Rust scoring, preserving fixed truth/reference domains and separate ALT support/loss endpoints.

- Reuse scores only when candidate output and every scorer pin/endpoint exactly match. Otherwise score retained candidate output once; do not rerun inference for scorer repairs. Reference-identical neighborhoods exclude variants and cannot replace ALT retention.

### Read and reuse

- Original scripts/accuracy.py actual interface; research/controlled-baseline-results.md; frozen GIAB truth/reference/mask manifests

- Retained official/source alignments, truth and full reference; preserve original six alternate-alignment row-order FAILs. No HG005 candidate selection.

### Pass these gates

- Predeclared numerical accuracy, callable/yield denominators, truth substitutions/indels, ALT support/loss, exposure and uncertainty pass with full accounting. Literal artifacts retain their own strict verdict.
- Pin exact source and inputs, command clocks/exits/raw logs, consumer code and required test names. No required test may skip. Audit exact live bytes and source-declared initialized regions; keep uninitialized storage exclusions prospective.

### Reject these failures

- Reference-base substitution at truth ALT, delete ALT-supporting reads, relabel donor, change mask, remove split lineage or replace a score object.

### Run or supply these commands

- to-implement. `go run ./verification/gpucaller/cmd biological -plan provenance/gpu-biological-plan.json -attempt ATTEMPT -out NEW_ANALYSIS`

Independent review. A fresh pstack_verifier audits the exact delta, raw receipts and one effective negative control. Source-only evidence cannot close native acceptance. Coordinate shared-file changes with the ABI or caller integration owner.

<a id="a03"></a>

## A03. Measure matched complete-process GPU performance

State is **planned**. Owner role is `pstack_worker; pstack_deep for arithmetic or lifetime work`. Dependencies are A00, A01, A02. Resources are source, timed, gpu-capture.

### Own these files

- `verification/gpucaller/benchmark.go`
- `provenance/gpu-benchmark-plan.json`
- `docs/gpu-benchmark-results.md`

### Implement this boundary

Measure complete process startup/load/input/model/decode/output/drain/close for policy-matched official and Go runs with identical admitted work, output and biological gates.

- Freeze warmup, batching/threads, formats/storage, runtime, power/profile/cooldown, repeat order and accepted-run rules. Existing policy uses five repetitions per arm; read current authority before scheduling. Reuse outputs for scoring, never historical clocks as fresh samples.

### Read and reuse

- Original research/{controlled-baseline-results,controlled-benchmark-plan,controlled-commands}.md and maintained stock runner; pstack benchmark-checklist

- Historical stock receipts for method/context; fresh matched clocks need separately admitted reservations. Apply benchmark checklist before reporting any number.

### Pass these gates

- Every accepted clock corresponds to complete equivalent work; telemetry and limiting-resource evidence support interpretation. Report ratio/uncertainty and regressions honestly. Valid measurement does not imply speedup.
- Pin exact source and inputs, command clocks/exits/raw logs, consumer code and required test names. No required test may skip. Audit exact live bytes and source-declared initialized regions; keep uninitialized storage exclusions prospective.

### Reject these failures

- Foreign GPU job, profile drift, missing work/sample, output truncation, thermal rejection, changed batch plan, producer error or telemetry gap hidden as accepted.

### Run or supply these commands

- to-implement. `go run ./verification/gpucaller/cmd benchmark -plan provenance/gpu-benchmark-plan.json -runtime-admitted -out NEW_ATTEMPT`

Independent review. A fresh pstack_verifier audits the exact delta, raw receipts and one effective negative control. Source-only evidence cannot close native acceptance. Coordinate shared-file changes with the ABI or caller integration owner.

<a id="a04"></a>

## A04. Audit release, reproducibility and maintenance closure

State is **planned**. Owner role is `pstack_worker; pstack_deep for arithmetic or lifetime work`. Dependencies are P01, P02, P03, P04, P05, H01, H02, H03, H04, H05, H06, H07, H08, H09, S00, S01, S02, S03, S04, S05, S06, S07, S08, S09, S10, S11, S12, S13, S14, D01, D02, D03, D04, I01, I02, I03, I04, I05, I06, I07, A00, A01, A02, A03. Resources are source.

### Own these files

- `verification/release/cmd/main.go`
- `provenance/gpu-release.json`
- `docs/gpu-release.md`
- `.github/workflows/gpu-contracts.yml`
- `README.md`
- `docs/port-plan.md`

### Implement this boundary

Completion consumes exact-head independent receipts for every mandatory card, full supported CLI/artifact matrix and reproducible package/build recipe, with native dependency inventory and upgrade rules.

- Audit missing/stale evidence, unsupported-request rejection, actual final files, clean-machine build/package limits and license/source obligations. Keep CPU CI scope explicit; add source/contract CI without implying GPU coverage. Upstream changes invalidate affected boundaries.

### Read and reuse

- AGENTS.md; docs/architecture.md; .github/workflows/cpu-parity.yml; every exact release source/runtime/weight/fixture pin

- All retained accepted evidence, failed attempts and current board. Empty queue, green CPU CI or card count cannot close this gate; no merge authority.

### Pass these gates

- All mandatory gates have pinned independent verdicts at the release tree; no research blocker or opaque whole-decoder/model bridge remains. Engineering, source/official identity, canonical/header/raw artifacts, biology/ALT and measured performance are reported separately.
- Pin exact source and inputs, command clocks/exits/raw logs, consumer code and required test names. No required test may skip. Audit exact live bytes and source-declared initialized regions; keep uninitialized storage exclusions prospective.

### Reject these failures

- Change one source/runtime/weight/payload/clock/exit, inject native-test skip or stale review, omit a card, falsely accept source-ready or mutate original controls.

### Run or supply these commands

- to-implement. `go run ./verification/release/cmd verify -manifest provenance/gpu-release.json -out NEW_AUDIT`

Independent review. A fresh pstack_verifier audits the exact delta, raw receipts and one effective negative control. Source-only evidence cannot close native acceptance. Coordinate shared-file changes with the ABI or caller integration owner.
