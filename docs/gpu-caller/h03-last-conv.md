# H03: execute the final HAC convolution and the source dispatch

The retained backend is **device-verified on the admitted host**. The corrected
source below still needs a fresh backend build and admitted native receipt.
H03 completes the HAC
convolution stack: the Go caller plans the final width-19 stride-6 tanh
convolution, zeroes the two input padding bands, packs its weight tensor into the
CUTLASS `[C_out, W*C_in]` form, runs the source's `host_convolution`-then-
`host_linear` dispatch through one native primitive per attempt, and reproduces
the H01 `conv2` stage byte for byte on the declared written rows.

It is not end-to-end basecalling. It does not run any recurrent layer, the
projection or the scores, and it is not biological evidence.

## Boundary

Go owns everything except one named primitive per attempt.

| Owned in Go | File |
| --- | --- |
| Working-memory placement, output-view arithmetic, allocation aliasing | `internal/model/hac_gpu_lastconv.go` |
| Weight packing to `[C_out, W*C_in]` and the RNE half cast | `internal/model/hac_gpu_lastconv.go` |
| Input padding writes (`ConvStack.cpp:265-266`) | `internal/model/hac_gpu_lastconv.go` |
| `cutlass_conv` / `koi_conv` / `koi_conv_f32` dispatch | `internal/model/hac_gpu_lastconv.go` |
| Activation and dtype validation, checked and unchecked calls | `internal/tensor/cuda_lastconv.go` |
| The two primitives | `verification/tensor/hac_lastconv.inc` |

ABI4 gains three additive symbols; a backend without them is rejected by
mandatory symbol resolution at `nt_open`, so an older backend cannot silently
serve this path.

| Symbol | Role |
| --- | --- |
| `nt_cuda_conv_koi_f16` | The caller-facing `host_convolution` primitive. |
| `nt_cuda_linear_f16` | The caller-facing `host_linear` fallback. |
| `nt_cuda_storage_reinterpret` | Typed equivalent of `backing_tensor.view(dtype)` (`WorkingMemory.cpp:48`); lets F16 conv views and the I8 CUTLASS output view share one reservation. |

The two primitives are exposed separately, and each reports its own `KoiResult`
through an out-parameter, so the receipt can name what actually ran. Collapsing
them into one symbol would hide the dispatch the source performs and would make
"the fallback produced the right answer" indistinguishable from "the fast path
is supported".

The native guard is a real contract, not a formality: it requires F16 input,
weights and bias with the exact `[C_out, W*C_in]` weight shape, `padding ==
window / 2`, `T_in % stride == 0`, the input allocation to be exactly
`T_in + 2*padding` rows with the declared batch stride, the output view's exact
strides `(out_n_stride, out_t_stride, 1)` with no overlap against the input
interval, an activation in `{SWISH, SWISH_CLAMP, TANH}`, a Koi output type that
agrees with the storage dtype, and F16 or F32 accumulation.

The output is deliberately **not** contiguous. The primitive writes through the
transposed CUTLASS view, whose strides are `(C_out, T_out*N*C_out, 1)`, not the
contiguous pair. A contiguity requirement would reject the source's own view.

## Declared case

Inherited from H01: batch 64, chunk 192, stride 6, `T_out = 32`, whole-model
reservation 7782400 bytes.

| Layer | Offset (bytes) | Extent (bytes) | Allocation shape | Runs |
| --- | --- | --- | --- | --- |
| input | 0 | 24576 | `[64, 192, 1]` F16 | — |
| conv0 | 7389184 | 393216 | `[64, 192, 16]` F16 | yes |
| conv1 | 0 | 430080 | `[64, 210, 16]` F16 | yes |
| conv2 | 5357568 | 2424832 | `[37, 64, 1024]` I8 | yes |

The final layer's input is the Go conv0→conv1 chain output, not the capture. The
source then zeroes the two nine-row padding bands
(`ConvStack.cpp:265-266`) one batch at a time through the contiguous
`[padding, C]` view; 36864 bytes were re-read and every one is zero.

The output view starts 65536 elements (65536 bytes) into the I8 allocation and
has shape `[64, 32, 1024]` with strides `[1024, 65536, 1]`.

## Allocation aliasing

`WorkingMemory::next` reuses the reservation without clearing it, and the last
allocation overlaps conv0's:

| Fact | Value |
| --- | --- |
| conv0 offset / extent | 7389184 / 393216 |
| Last-allocation row size | 65536 bytes (`N*C_out`, I8) |
| First shared row | 31 |
| Shared rows | 6 (conv0's whole extent) |
| Rows the output view writes | 31, 32 |
| Rows left holding conv0's output | 33–36 (262144 bytes) |

Both halves of that claim are observed, not assumed. Rows 33–36 are byte-equal to
conv0's rows 2–5 in the same read that produced the conv2 comparison, and rows
31–32 no longer match conv0. The declared-unwritten rows are
`unwritten_t = [[0,1],[33,4]]`, which the plan derives from the source and
`Import` validates against the capture **before execution** — the mask is never
derived from observed mismatches.

Row 0 of the allocation (65536 bytes) is still the declared `0x5a` fill pattern,
and every byte outside the four declared views — the span `[430080, 5423104)` —
is still fill. Those regions are proven untouched, not value-compared.

## What was compared

| Stage | Compared bytes | Go written SHA256 | Capture written SHA256 | Equal |
| --- | --- | --- | --- | --- |
| conv0 | 393216 | `7f9f503230ff8f85…` | `7f9f503230ff8f85…` | yes |
| conv1 | 393216 | `fa6f0f8c3b213cad…` | `fa6f0f8c3b213cad…` | yes |
| conv2 | 2097152 | `a279972e5bf66cd0…` | `a279972e5bf66cd0…` | yes |

conv2 compares 2097152 of its 2424832 bytes: the 32 written time rows. The
remaining 327680 bytes are the five declared-unwritten rows accounted for above
(one still fill, four still holding conv0's output).

The dispatch the source actually took is recorded rather than inferred:

| Field | Value |
| --- | --- |
| `dev_options` | `cutlass_conv=true koi_conv=true koi_conv_f32=false` |
| `fast_attempted` / `fast_result` | true / `KOI_SUCCESS` |
| `fallback_attempted` | false |
| `path` | `host_convolution` |

Weight packing is a separate differential against the H02 verification-only
native packing oracle. The first two layers pack to `[W*C_in, C_out]`; the last
packs to `[C_out, W*C_in]`, which is proven against the **transpose** of the same
native bytes. No new native packing symbol was added for H03.

| Layer | Packed layout | Weight pin | Go = reference | Transposed |
| --- | --- | --- | --- | --- |
| conv0 | `W*C_in,C_out` | `8c6b47fe…` | yes | no |
| conv1 | `W*C_in,C_out` | `8d5cfa76…` | yes | no |
| conv2 | `C_out,W*C_in` | `90105d8c…` | yes | yes |

## Rejection controls

`verification/hacgpu/h03-controls.sh` disables exactly one native guard per
attempt, rebuilds the ABI4 backend, and re-runs
`verification/hacgpu/h03/cmd/guards`. That probe calls
`Device.LastConvKoiUnchecked` / `Device.LastConvLinearUnchecked`, which bypass
every Go pre-check, so an observed outcome can only come from the native
boundary. It also runs three replay mutants against the unmutated baseline.

Every case uses the declared case's geometry. A scaled-down geometry was tried
first and the primitive refused it with `KOI_NOT_SUPPORTED` even for the valid
request, which would have made "the kernel ran" indistinguishable from "the
kernel refused the shape"; that attempt is preserved as
`artifacts/tensor/h03-controls-001`. In the baseline build below, `valid` and
`fallback-valid` both return `ok`, so a mutated `ok` means the primitive really
executed the request.

| Control | Guard removed | Baseline | After mutation |
| --- | --- | --- | --- |
| `padding` | `padding == window/2` | `padding must be window/2` | advances to `input rows are not T_in plus two paddings` |
| `input-rows` | `T_in + 2*padding` rows | `input rows are not T_in plus two paddings` | `ok` |
| `input-strides` | `(T_alloc*C_in, C_in, 1)` | `input strides differ from the view` | `ok` |
| `output-strides` | output view strides | `output strides differ from the view` | `ok` |
| `weight-shape` | `[C_out, W*C_in]` | `weights shape differs from C_out,W*C_in` | `ok` |
| `activation` | unknown-activation throw | `unknown Koi activation for the last convolution` | `koi KOI_NOT_SUPPORTED` |
| `output-type` | output dtype agreement | `output type differs from its storage` | `ok` |
| `accumulation` | F16/F32 accumulation | `accumulation must be F16 or F32` | `koi KOI_NOT_SUPPORTED` |
| `overlap` | input/output interval | `output overlaps an input interval` | `ok` |
| `forced-fallback` | — (`koi_conv=false`) | — | conv2 differs; `path: host_linear` |
| `f32-accum` | — (`koi_conv_f32=true`) | — | conv2 differs; `path: host_convolution` |
| `i8-corrupt` | — (one output byte replaced) | — | conv2 differs |

The retained baseline build prints `valid: ok`, `fallback-valid: ok`, and rejects the
twelve negative probe cases with the boundary's own messages.

The current probe adds `shifted-overlap` and `fallback-shifted-overlap` for the
corrected source. The F16 input occupies `[0,430080)` bytes. The I8 output
occupies `[262144,2359296)` bytes in the same allocation. The old native guard
compared their element endpoints and accepted 167936 overlapping bytes.
Every typed native overlap guard now uses `storage_overlap`, which converts
endpoints to bytes through `storage_byte_interval.h` and rejects conversion
overflow before comparing the intervals.
Reinterpretation uses the same checked byte interval and bounds its relative
offset before multiplication or addition. It rejects unaligned offsets and
endpoints that cannot fit the signed tensor-storage offset.

The standalone failure probe includes the actual shared helper. It checks this
shifted case, the one-byte overlap at 430079, exact adjacency at 430080, F32
intervals, empty intervals, invalid or overflowing endpoints, and huge
reinterpret offsets. The bounded
host checks are retained under `artifacts/native-overlap-fix-20261024-001` and
`artifacts/native-overlap-fix-20261024-003`, with
source hashes, exact command arguments, clocks, exit statuses, and raw logs.
The original source-expression reproduction exits 1. The corrected failure
probe compiles and exits 0. These host checks establish helper behavior and
preserve the existing owner-retention checks. They do not establish native
CUDA execution or parity of the corrected backend.

After a fresh ABI4 CUDA13 backend build and explicit runtime admission, the
remaining native guard command is:

```sh
go run ./verification/hacgpu/h03/cmd/guards -backend "$NEW_BACKEND" -ordinal 0 -runtime-admitted
```

Both shifted cases must report native overlap rejection. The fresh backend also
needs the maintained H03 differential replay and rejection controls before the
corrected source can inherit device qualification. Existing receipts identify
the old backend and remain evidence only for that artifact.

Two of the mutants are honest about what they show. Removing the activation
guard or the accumulation guard lets a request the primitive will not serve reach
the primitive, which returns `KOI_NOT_SUPPORTED`; those guards turn an opaque
non-success result into a typed rejection naming the disagreement. Six of the
remaining controls let the kernel run to completion on a request the source would
never make, and the padding control advances to the stricter rows guard instead
of reaching the kernel with a window that does not match the declared padding.

Forced fallback is a **negative** control, not a second accepted path:
`host_linear` is a different kernel and does not reproduce the capture. A
fallback success is not fast-path support, and the receipt records the path
separately from the comparison result.

`artifacts/tensor/h03-controls-002` is preserved as a partial attempt: nine
controls held, and the `weight-shape` verdict was mis-derived because the
script matched the probe line by substring and `weight-shape:` is a suffix of
`fallback-weight-shape:`. The matcher now anchors on the line prefix, and the
control was re-run in isolation in `h03-controls-003`, where it held. The raw
probe output for it in `-002` (`weight-shape: ok`) is the same evidence.

## Evidence

| Item | Value |
| --- | --- |
| Backend build | `artifacts/tensor/h03-build-003`, ABI 4, CUDA family 13.0 |
| Backend SHA256 | `a804a9499407b54e28ae3047946338c0f4f74187d08bcb5f3349ee41b2ef1e07` |
| Build manifest SHA256 | `cecc21eeed345ae65d274e8cea6c3ab0a4c08f4ad95558877fa10064e2af9b02` |
| Oracle capture | H01 `artifacts/tensor/hacgpu-capture-008/stages` |
| Capture manifest SHA256 | `61dd1db61a7d61234cbc6a8afb852a6cfbde1c5a53ad56a501e2027c1f897149` |
| Differential receipt | `artifacts/tensor/h03-lastconv-009/receipt.json`, status `lastconv_passed` |
| Verified backend build inputs | 4830, digest `3964493145725231…` |
| Runtime pins | 7 objects (`libtorch_cpu`, `libc10`, `libtorch_cuda`, `libc10_cuda`, `libcudart`, `libcublas`, `libcuda`) — `libkoi` is linked statically, so it is a build pin, not a mapped object |
| Rejection controls | `artifacts/tensor/h03-controls-002/transcript.txt` (9 of 10) and `-003` (`weight-shape`, held) |
| Pure unit tests | `go test ./verification/hacgpu/h03`, 8 tests |

Preserved attempts: `h03-build-001` and `-002` (backend rebuilt after source and
pin changes), `h03-lastconv-001` (input padding band measured with a one-byte row
stride), `-002` (alias rows taken from the input offset instead of the recorded
alias), `-003` (row 0 assumed aliased), `-004` (first green run, before the alias
flags were derived from the observation), `-005` (green, superseded by the
receipt-field cleanup), `-006`/`-007` (the forced-fallback and f32-accum replay
mutants, kept with their dispatch and stage digests), `-008` (green, but the
receipt still carried null placeholder fields for evidence the replay does not
collect), `-009` (the recorded receipt), `h03-controls-001` (a probe geometry the
primitive refuses, so no mutant could be shown to run), and
`h03-controls-002` (nine controls held; the tenth verdict was mis-derived by a
substring match, see above).

The replay refuses to run unless the backend manifest is ABI 4, CUDA family
13.0, status `built`, and every declared build input still matches on disk; and
unless the capture's declared graph shape, stage list, offsets, extents, layouts
and unwritten rows agree with the Go plan. A drifted plan is a rejection, not a
warning.

## Non-claims

- No recurrent layer, projection or score is executed here.
- Byte-identical convolution stages are not a basecall, an accuracy measurement
  or biological evidence.
- The four unwritten rows are proven untouched or proven to still hold an earlier
  allocation's output; they are never value-compared with the capture.
- The `cutlass_conv=false` window/half-GEMM path is **not ported**. It is
  declared out of scope and rejected at construction and at `Run`, not silently
  substituted.
- A successful `host_linear` fallback would not count as fast-path support, and
  in the declared case it does not reproduce the capture at all.
- `nt_cuda_weights_permute_cast_f16` is not on the caller path; it exists only so
  the pure-Go packer can be checked bit for bit.
- This is a same-family implementation and review; no independent
  cross-family review has been performed.
