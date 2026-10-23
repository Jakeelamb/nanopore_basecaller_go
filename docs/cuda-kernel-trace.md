# CUDA kernel boundary for the Go caller

Source investigation on 2026-10-23 UTC. Go baseline is
`2ea835007b332b09eb4cc4ae239aa87d9ec30ed8`. Dorado source is
`8b8fc5d36a9c0baab262a743cb175b5878e38ca6`, committed
2026-08-26T11:54:52Z. All Dorado pointers below are relative to
`external/dorado-source/` at that revision. Koi pointers are relative to
`external/backend/koi/include/koi.h`, version 0.7.5. Existing CPU graph and
FASTQ evidence does not qualify any CUDA operation. No GPU execution, build,
download, model modification, or original-workspace write was performed here.

## Boundary decision

Go must own model recognition, weight binding and preparation, graph order,
recurrent direction, batching, workspace planning, stream sequencing, and read
assembly. Native code may expose individual tensor operations and explicitly
named fused Koi operations. A fused recurrent layer is an existing GPU primitive,
not evidence that a native whole-model wrapper is needed.

`CRFModelImpl::run_koi` (`dorado/basecall/model/CRFModel.cpp:87-131`) and
`TxModelImpl::forward` (`dorado/basecall/model/TxModel.cpp:20-47`) are the
orchestration to port. Exposing either as a production bridge would retain the
model graph in C++. `CudaCaller::call_chunks` and a native whole-decoder wrapper
likewise remain diagnostic oracles rather than completion of Go caller ownership.
The current boundary is ABI 3 (`internal/tensor/abi.h:5-36`) with CPU float32
computation and separate int64 index tensors. It cannot represent CUDA half,
quantized tensors, streams, or in-place workspace views.

## HAC uses different arithmetic on CUDA

`CRFModel.cpp:36-39` selects FLSTM. CPU `FLSTMLayerImpl::forward`
(`dorado/nn/FLSTMStack.cpp:19-64`) uses two factorized input matmuls, two
recurrent matmuls per timestep, separate input and recurrent biases, hard
sigmoid gates, hard candidate tanh, and ordinary tanh on the cell state.
`FLSTMStack.cpp:83-92` owns alternating flips and final orientation.

CUDA dispatch is input-device and half dependent (`CRFModel.cpp:135-145`).
`dorado/nn/ConvStack.cpp:66-104` chooses the FLSTM layout. The fused path needs
`C=1024,K=128`, compute major at least 8, and at least nine SMs
(`dorado/nn/KoiUtils.cpp:7-11,19-24,33-38`). These are eligibility predicates,
not proof that the installed binary supports the host architecture.

| CUDA operation | Go work and native boundary | Exact source |
| --- | --- | --- |
| Convolution | Go selects convolution, padding, activation, output layout and fallback. Native `host_convolution_f16`, `host_convolution`, or `host_linear` performs one convolution or windowed linear operation. | `ConvStack.cpp:215-320`; Koi declarations `:241-258,417-467` |
| Quantized FLSTM | Go schedules one `host_factorised_lstm` per layer with direction `first_reverse ^ (layer & 1)` and parity `1-layer%2`. This kernel includes the layer's recurrence. | `FLSTMStack.cpp:125-235`; Koi `:351-368` |
| FP16 fallback FLSTM | Go schedules three GEMMs and one `host_lstm_step_f16` for every timestep. It owns zero cell state and the input, recurrent and output offsets. | `FLSTMStack.cpp:239-295`; Koi `:323-331` |
| Factorized output | Go prepares quantized down weights, FP16 up weights, scales, temporary buffers and final NTC view. Native `host_factorised_linear` is one fused projection with activation scale. | `dorado/nn/FactorisedLinearLayer.cpp:52-106`; Koi `:260-273` |

The fused FLSTM receives stream, N, T, direction, parity, inout, sync,
optional encoding, down input weights/scales, down recurrent weights/scales,
down buffer/scales, combined up weights/scales, and up bias. The API returns an
integer Koi status. Shapes are constrained by its header comment `N%64=0,
K=128,C=1024`. Its prepared weights are not ordinary CPU weight arrays.
`FLSTMStack.cpp:165-217` defines the exact quantization, reshape and permutation
sequence. Both CUDA paths pass `up_bias_ih` alone (`:213-217,274`), while CPU
uses both biases. Do not silently add the recurrent bias to the CUDA path.
Whether these models make that difference numerically inert requires tensor
evidence.

The quantized scratch allocation is U8 storage containing
`2*(T+1)*N*(2*K)` down-buffer bytes, `4*(T+1)*N` scale bytes and `4*(N/2)`
sync bytes. It is zeroed once before the layer loop (`FLSTMStack.cpp:149-158`).
For fixed chunks the inout sentinel rows 0 and T+1 are zero and row T+2 is one.
VCS zeros rows T+2 and T+3 (`:137-144`). The fallback scratch is half
`N*(2*K+4*C)` plus zeroed half cell state `[N,C]` per layer. Its down-weight
transposes and concatenated up-weight transpose differ from the fused packing.

Use the allocation code over comments when they disagree. `WorkingMemory.cpp:83-98`
allocates CUTLASS buffers `[T+5,N,C]`, not the `[T+3,N,C]` stated in
`ConvStack.cpp:60-61`. CUBLAS_TNC really allocates `[T+3,N,C]`.
`WorkingMemory.cpp:61-76` defines the orientation-dependent live NTC view.
Its two-pass arena reserves aligned byte maxima, then alternates ends and rejects
overlap (`:30-58,108-113`; alignment 256 in `include/nn/WorkingMemory.h:55-58`).

For fallback GEMM, `dorado/torch_utils/cuda_utils.cpp:377-397` requires half
inputs/output, unit final stride, explicit leading strides, and `cublasGemmEx`
with `CUDA_R_16F` compute and tensor-op algorithm. A float32 accumulation GEMM
is not an assumed substitute.

## SUP is a tiled and potentially quantized graph

`dorado/nn/TxModules.cpp:805-834` selects tiled Koi for the recognized SUP
shape, CUDA device and successful `koi_tc_is_available(KOI_F16)`. The SUP shape
is C=512, eight heads, D=64, attention window 127/128 and feedforward 2048.
`koi_use_i8` defaults true. FP8 SwiGLU defaults enabled when the Koi capability
query succeeds (`:439-440`). Capture actual capability and dispatch results.
Do not infer FP8 support from the GPU model name.

Each encoder schedules seven native calls (`TxModules.cpp:558-610`).

1. `koi_qkv_rotary(stream,in,qkv_weights,sincos,out_qkv,t_lut_block64,ctr)`.
2. `koi_masked_attention(stream,win_upper,win_lower,qkv,out)` for fixed chunks,
   or `koi_vcs_attn` for variable chunks.
3. `koi_linear(stream,a,b,bias,out,ctr)` for output projection.
4. `koi_rmsnorm_residual(stream,in,residual,alpha,weights,out_f16,out_f8,out_i8)`.
5. `koi_mm_swiglu(stream,a,b,out,ctr,use_f32_accum,use_uswiglu)`.
6. `koi_linear` for the second feedforward projection, with null bias.
7. `koi_rmsnorm_residual` for the second residual and norm.

All return integer Koi status. Declarations are Koi `:145-178`.
`KoiTensor` (`:3-36`) carries pointer, dtype, up to eight tagged dimensions with
int64 sizes and element strides, plus optional scale pointer/tag/type.
`TxModules.cpp:30-79` accepts half, float32, FP8 E4M3 and int8. Scales attached
to int8 are float32. The port needs these descriptors, not just a shape and
untyped device pointer.

Go must reproduce these transformations before claiming CUDA graph parity.

- Fixed NTC half input tiles to `[N,T/16,C/8,16,8]`. Int8 input tiles to
  `[N,T/16,C/16,16,16]`. Quantization computes `128/amax(abs(x))`, rounds,
  clamps to [-127,127], and converts to int8. SUP takes reciprocal scales.
  Sources are `TxModules.cpp:871-905` and
  `dorado/torch_utils/tensor_utils.cpp:291-297`.
- Q/K weight halves interleave while V copies directly. RoPE sin/cos pairs
  are half and tiled `[max_T/16,D/8,16,8]`. Projection and feedforward weights
  have separate tilings, including an FP8-specific FC1 shuffle. The exact
  one-time cache construction is `TxModules.cpp:443-518`.
- Weight rounding defaults to removing four low half mantissa bits, including
  the residual norm weights. This adds `1<<(remove_bits-1)` through an int16
  view and masks the low bits. It occurs after the FP8/int8 caches are made.
  Sources are `TxModules.cpp:99-105,403-412,517`. Preserve order and dtype.
- QKV output is half `[N,T/16,3,H,D/8,16,8]`. Attention output is half
  `[N,T/16,H,D/8,16,8]`. Residual norms update the half input in place and
  may also refresh FP8 or int8 representations. Four zeroed int32 counters
  belong to each encoder invocation. Intermediate shapes and descriptor tags
  are explicit at `TxModules.cpp:431-432,520-555`.

The generic CUDA path is separate. `TxModules.cpp:297-343` transposes Q/K
weights and bias once, then calls `host_rotary_embed_transpose_f16` with NT3HD
input and 3NHTD output. Generic attention uses split masked Torch SDPA
(`:350-382`). Generic residual norms try `host_fused_residual_rmsnorm_f16`
before the tensor fallback (`:756-802`). Volta has a separate kernel family
compiled only below CUDA 13 (`:819-823,919-941`). None is an interchangeable
CPU float32 graph.

SUP fixed-chunk convolution still uses ordinary conv1d plus activation
(`dorado/basecall/model/TxModel.cpp:20-45`; `ConvStack.cpp:169-185`). VCS uses
five layer-indexed `koi_vcs_tx_cnn` calls, explicit load/store LUTs, convolution
weight tilings and padded output layouts (`ConvStack.cpp:376-457`). Encoder VCS
uses 64-step granularity and pads N so M is a multiple of 256, fills its RoPE
LUT and changes attention dispatch (`TxModules.cpp:840-864,570-574`). Go also
owns the final linear upsample and CRF projection. Their operations are linear
plus reshape and one-time weight scaling plus linear
(`dorado/nn/LinearUpsample.cpp:17-22`; `TxModules.cpp:952-957`).

## GPU CRF decode has four calls at this revision

`dorado/basecall/decode/CUDADecoder.cpp:95-126` calls
`host_back_guide_step`, `host_beam_search_step`, `host_compute_posts_step`,
then `host_run_decode`. There is no fifth call in this function. A proposed
fifth operation needs a different revision or a separately identified source.

The first three share this signature (Koi `:469-524`).

```c
int step(void *stream, void *chunks, void *chunk_results, int num_chunks,
         void *post, float post_clamp, int post_stride, void *aux_buffer,
         void *path, void *moves, void *weights, void *sequence, void *q_string,
         float qscale, float qshift, int beam_width, float beam_cut,
         float fixed_stay_score);
```

`host_run_decode` adds final `int move_pad` (Koi `:526-544`). Dorado passes
null weights, half contiguous NTC scores, C as score stride, and model quality
scale/bias plus beam and blank parameters. Score clamp moves into decoder reads
on HAC (`CRFModel.cpp:125-127`). `CUDADecoder.h:19` specifies half input.

For fixed N,T,C and beam B, the buffers are (`CUDADecoder.cpp:30-93`).

| Buffer | Allocation and initialization |
| --- | --- |
| `chunks` | int32 `[N,4]`, rows `[i*T,T,i*T,0]` |
| `chunk_results` | uninitialized int32 `[N,8]` |
| `aux` | uninitialized int8 `N*(T+1)*(C+4*B)` bytes |
| `path` | zero int32 `N*(T+1)` |
| moves, sequence, quality | zero int8 planes `[3,N*T]`, reshaped to `[3,N,T]` after kernels |

VCS allocation and chunk-table mutations differ (`CUDADecoder.cpp:38-54,68-84`).
Do not reuse fixed formulas for packed chunks. Host conversion copies exactly T
move bytes and takes `sum(moves)` sequence and quality bytes
(`:136-194`). Kernel pointers are opaque. Public sources opened here do not
define aux internal float/int packing, result-field semantics, tie-breaking,
posterior calculation or quality rounding. Treating byte scratch as a documented
float layout would invent a contract. The static archive exports these symbols,
but a symbol is not their implementation source.

For the final Go decoder target, these four native entrypoints are staged
oracles. Go must own decoder decisions. Capture guides, paths, moves and quality
after each entrypoint before replacing it. A production opaque decoder bridge
cannot establish that ownership. Exact tie and quality behavior remains blocked
on GPU differential captures or additional implementation source.

## Scheduling and buffer lifetimes

CPU uses float32 and fixed batches (`dorado/basecall/ModelRunner.cpp:9-37`).
CUDA batch granularity is 64 for recurrent models and 32 for transformers
(`dorado/basecall/CudaCaller.cpp:211-214`). Requested sizes are rounded to that
granularity and zero means automatic memory/benchmark selection (`:477-505`).
CLI variable chunks default enabled (`dorado/cli/cli_lib/basecaller.cpp:1192`),
subject to the exact model/device predicates
(`dorado/api/runner_creation.cpp:24-60`). CPU forces fixed chunks
(`basecaller.cpp:513-517`). A fixed-chunk CUDA slice must explicitly disable
VCS and freeze N,T, beam parameters and all dev options. One logical chunk may
therefore require a padded 64-slot HAC or 32-slot SUP batch.

`CudaModelRunner.cpp:26-58` packs variable chunks, tracks their lengths, inserts
SUP padding and creates auxiliary data. `CudaCaller.cpp:246-289` copies input,
queues work, waits and copies decoded output. A pooled stream is owned by the
caller (`:51-53,132`) and guarded in its worker (`:681`). Model and decode run
in order and synchronize before task completion (`:729-748`). Native calls get
the current stream. Go must retain device owners, views, weight caches,
descriptors, counters, and scratch until the stream completes. Per-operation
freeing after enqueue is unsafe. Thread-local Torch device/stream selection
must be established inside each native call rather than assumed across Go
goroutine migration.

## Smallest proposed Go API and acceptance sequence

These are proposed interfaces, not implemented functionality.

Use opaque `DeviceContext` and `Stream` owners, `DeviceBuffer` with byte length,
dtype and device, and `TensorView` with owner, byte offset, tagged shape and
element strides. Add explicit `ScaleView`, typed per-operation parameter
structs, and a completion token retaining all referenced owners. The C adapter
validates bounds, dtype, layout, device, liveness and overlap before enqueue.
It exposes allocation, upload/download, cast, view/copy, GEMM, named fused Koi
operations and synchronization. Go performs graph loops and weight preparation
by primitive operations. Do not add `ForwardHAC`, `ForwardSUP` or `CallRead`.

1. Build an isolated family-pinned backend and run one real half device GEMM
   from Go. Capture output, driver/device, stream, all loaded library paths and
   hashes. Reject wrong dtype, shape, device, stride, byte bounds and released
   handles. Exercise a view after parent release, enqueue followed by early
   release, and stream completion. Reject unsafe lifetimes or retain owners.
2. Admit one fixed HAC logical chunk in a valid padded batch. Capture input,
   every convolution, prepared FLSTM weights/scales, every recurrent layer,
   factorized output and half scores from a native oracle with the same family,
   model and branch. Compare Go scheduling at every boundary. Run fallback and
   fused paths as distinct cases, never count fallback as fused acceptance.
3. Admit one fixed SUP chunk. Capture selected int8/FP8 capabilities, all
   caches, post-rounding weights, RoPE output, attention output, every residual
   norm and feedforward result, upsample and CRF output. Preserve native half
   bytes and separately report numerical deviations. Then qualify VCS packing,
   LUTs and the default branch as another unit.
4. Feed captured half scores to each GPU decode stage and retain complete
   scratch before/after with allocation metadata. Freeze tied-score and
   near-tie controls, blank transitions, chunk padding, posterior/quality edge
   cases and exact moves. Replace stages with Go decisions only against this
   oracle. CPU decode equality is a different endpoint.

## Available assets and remaining blockers

Stock-family Koi 12.8 exists at `external/backend/koi/`. Its archive pin is
`bfb3bea6f1bb7f7e6e602d4899db14f422a46519ce69e2d1b645050eba307aad`
(`cmake/Koi.cmake:23`). Stock-family Torch12.8 remains a partial archive at
`external/backend/archives/torch-2.9.0.0-ont-CUDA-12.8-linux-x64-cxx11-abi.zip.partial`.
A complete stock-family header/library set is therefore still missing locally.

A separate complete Koi13.0 archive and extracted header/static library are
present under original-workspace
`software/source-build-qualification-20261023/`. Read-only SHA-256 checks here
matched the archive source pin `0600a138761b17467bd3a1f8933c6a6a02e17415a9da377f22bc14357c3101ca`
(`cmake/Koi.cmake:24`). Exact extracted paths are
`dependencies/libkoi-0.7.5-Linux-x86_64-cuda-13.0/libkoi-0.7.5-Linux-x86_64-cuda-13.0/include/koi.h`
and the sibling `lib/libkoi.a`. Their hashes matched the retained
`evidence/dependency-files.sha256` entries.

- Header SHA-256 is `8b4115c2e92ebf1acf1db2aa76bdc9b3a22768cb595b92363f88e44ff92d3b88`.
- Static library SHA-256 is `17cf3083c5db538f1e275b0a41ac19560b5986c143634e8a25540b9735f5805f`.

The Koi12.8 and Koi13.0 headers compared byte-identical. The complete Torch13
development bundle is at
`dependencies/torch-2.9.0.0-ont-CUDA-13.0-linux-x64-cxx11-abi/libtorch` in that
original qualification root. Its prior read-only verification is supplied by
the coordinator, not repeated by this investigation. These assets allow a
separate proposed CUDA13 qualification. They do not qualify stock CUDA12.8
parity. Never load Torch13 with the stock12.8 runtime family. Host toolkit13.4
compatibility remains a build/runtime check, not an inferred pass.

`nm -g --defined-only external/backend/koi/lib/libkoi.a` confirmed the fused
FLSTM/output, SUP and four decode exports listed above. `nm -u` found CUDA
runtime dependencies including registration, launch, memcpy/memset, device
attribute, synchronization and driver-entrypoint calls. A small adapter likely
needs CUDA runtime headers/library, Torch headers plus `torch`, `torch_cpu`,
`torch_cuda`, `c10`, `c10_cuda`, and cuBLAS for fallback GEMM. This is a link
plan, not a verified link closure. Match C++ ABI, exact runtime family and
loader resolution. Full-model linking may also pull in flashattention, but the
named Koi-only operations do not justify importing the full Dorado caller.

Koi implementation source is not in the public snapshot. Its optional source
override names a vendor-internal repository (`cmake/Koi.cmake:40-60`). Hardware
kernel availability, numerical equality, decode internal layouts, CUDA13 source
oracle equivalence, GPU memory use and performance remain unqualified. Preserve
the stock controls and both existing CPU endpoints unchanged while obtaining
that evidence.
