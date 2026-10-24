# H02: prepare and execute the first two HAC convolutions

Status is **device-verified on the admitted host**. H02 is the first port unit
that executes real caller arithmetic: the Go caller plans the HAC working-memory
placement, packs both convolution weight tensors, runs the two width-5 stride-1
SiLU convolutions through one native primitive each, and reproduces the H01
convolution stages byte for byte on the declared written rows.

It is not end-to-end basecalling. It does not run the final width-19 stride-6
tanh convolution, any recurrent layer, the projection or the scores, and it is
not biological evidence.

## Boundary

Go owns everything except one named primitive per call.

| Owned in Go | File |
| --- | --- |
| Working-memory placement, crop geometry, batch stride | `internal/model/hac_gpu_smallconv.go` |
| Weight/bias packing and the RNE half cast | `internal/model/hac_gpu_smallconv.go` |
| Activation mapping (`get_koi_activation`, `ConvStack.cpp:21-31`) | `internal/model/hac_gpu_smallconv.go` |
| Typed request validation and the checked call | `internal/tensor/cuda_smallconv.go` |
| The primitive itself | `verification/tensor/hac_smallconv.inc` |

ABI4 gains exactly two symbols, both additive; a backend without them is
rejected by mandatory symbol resolution at `nt_open`, so an older backend cannot
silently serve this path.

| Symbol | Role |
| --- | --- |
| `nt_cuda_conv_small_f16` | The single caller-facing `host_convolution_f16` primitive. |
| `nt_cuda_weights_permute_cast_f16` | Verification-only packing oracle. **The Go caller never calls it.** |

`libkoi.a` is linked into the backend and pinned as a build input, so the
primitive resolves against the same `koi.h` the pinned source was built against.

The native guard is a real contract, not a formality: it requires F16 for all
four storages, one device, contiguous input/weights/bias, `padding == window/2`,
`T_in % stride == 0`, the exact shapes, `out.stride(0) == out_n_stride` with
`stride(1) == C_out` and `stride(2) == 1`, no overlap between the output and any
input, and an activation in `{IDENTITY, SWISH, SWISH_CLAMP, TANH}`.

The output is deliberately **not** contiguous. The cropped view starts nine rows
into its allocation and keeps the parent's batch stride, which is what preserves
the nine rows of last-convolution padding between batches
(`ConvStack.cpp:241-242`). A contiguity requirement on the output would silently
drop them.

## Declared case

Inherited from H01: batch 64, chunk 192, stride 6, `T_out = 32`, whole-model
reservation 7782400 bytes. `WorkingMemory::next` (`WorkingMemory.cpp:36-58`)
alternates front and back allocation, which reproduces these offsets exactly:

| Layer | Offset (bytes) | Extent (bytes) | Allocation shape | Runs |
| --- | --- | --- | --- | --- |
| input | 0 | 24576 | `[64, 192, 1]` F16 | — |
| conv0 | 7389184 | 393216 | `[64, 192, 16]` F16 | yes |
| conv1 | 0 | 430080 | `[64, 210, 16]` F16 | yes |
| conv2 | 5357568 | 2424832 | `[37, 64, 1024]` I8 | planned, not executed |

conv1's allocation aliases the input buffer the graph has already consumed.
`WorkingMemory::next` reuses memory without clearing it, so H02 pre-fills that
allocation with the declared `0x5a` pattern **after** conv0 runs and **before**
conv1 runs, and records this in `declared_preparation`. Nothing reads the
allocation before conv1 writes it. Without that step the padding rows are
unobservable rather than proven unwritten; `artifacts/tensor/h02-smallconv-003`
is preserved as the failed attempt that discovered this.

## What was compared

| Stage | Compared bytes | Go written SHA256 | Capture written SHA256 | Equal |
| --- | --- | --- | --- | --- |
| conv0 | 393216 | `7f9f503230ff8f85…` | `7f9f503230ff8f85…` | yes |
| conv1 | 393216 | `fa6f0f8c3b213cad…` | `fa6f0f8c3b213cad…` | yes |

conv1 compares 393216 of its 430080 bytes because the capture declares
`unwritten_t = [[0,9],[201,9]]`: eighteen time rows that the kernel never
writes. That mask comes from the pinned source and is validated by `Import`
before execution — it is never derived from observed mismatches. The eighteen
rows were instead proven to still hold the declared fill pattern (576 bytes, all
`0x5a`), which is the evidence that the crop offset and the parent batch stride
were honoured.

Weight packing is a separate differential. The Go packer's bytes and the
verification-only native oracle's bytes agree exactly for both layers:

| Layer | Weight pin | Go packed = reference | Bias cast = reference |
| --- | --- | --- | --- |
| conv0 | `8c6b47fe…` | yes | yes |
| conv1 | `8d5cfa76…` | yes | yes |

## Lifetime

Closing the input view, the conv0 allocation view and conv1's crop view must not
disturb anything still live. H02 asserts three things:

1. conv1's parent allocation is byte-identical before and after its crop view is
   released.
2. The whole backing is re-read afterwards; every byte outside the executed
   views still holds `0x5a`, including the whole conv2 allocation except the
   interval conv0 legitimately occupies.
3. The unrelated backing is untouched, so neither kernel wrote outside its own
   view.

## Rejection controls

`verification/hacgpu/h02-controls.sh` removes exactly one native guard per
attempt, rebuilds the ABI4 backend, and re-runs
`verification/hacgpu/h02/cmd/guards`. That probe calls
`Device.SmallConvF16Unchecked`, which bypasses every Go pre-check, so an observed
outcome can only come from the native boundary. A Go pre-check would prove
nothing about the native guard; that is why the unchecked entry point is
exported.

| Control | Mutation | Observed after mutation |
| --- | --- | --- |
| `omitted-input-padding` | padding guard | advances to the stride guard: `output batch stride differs from the view` |
| `wrong-output-stride` | output stride guard | `ok` — the kernel runs with a stride that is not the view's |
| `unknown-activation` | activation throw | `ok` — the kernel runs as SWISH instead |
| `non-multiple-stride` | `T_in % stride` guard | advances to the shape guard: `input shape differs from N,T,C_in` |
| `non-f16-storage` | F16 dtype guard | `ok` — the kernel runs on F32 weights |

The baseline build, with no mutation, prints `valid: ok` and rejects all five
cases with the boundary's own messages. Each mutation therefore changes exactly
one outcome, which is what makes the guard load-bearing rather than incidental.

`artifacts/tensor/h02-controls-001` is preserved as a failed attempt: four
controls held, and the fifth produced the predicted behaviour but the script's
expected string said `N,T_in,C_in` where the boundary says `N,T,C_in`. The
expectation was corrected and the control re-run in isolation
(`h02-controls-002`) before the complete run in `h02-controls-003`.

## Evidence

| Item | Value |
| --- | --- |
| Backend build | `artifacts/tensor/h02-build-002`, ABI 4, CUDA family 13.0 |
| Backend SHA256 | `8bfd17449cd78f7d4a9fb442913920af68c72b3abb406f8d81448837c4ea0cd3` |
| Build manifest SHA256 | `fa2e5f617b75304a0ae00a5f7cec68e625bd03f37789a53b16a090fb877c0db4` |
| Oracle capture | H01 `artifacts/tensor/hacgpu-capture-008/stages` |
| Capture manifest SHA256 | `61dd1db61a7d61234cbc6a8afb852a6cfbde1c5a53ad56a501e2027c1f897149` |
| Differential receipt | `artifacts/tensor/h02-smallconv-006/receipt.json`, status `smallconv_passed` |
| Verified backend build inputs | 4827, digest `fa8b37367896e0de…` |
| Runtime pins | 7 objects (`libtorch_cpu`, `libc10`, `libtorch_cuda`, `libc10_cuda`, `libcudart`, `libcublas`, `libcuda`) |
| Rejection controls | `artifacts/tensor/h02-controls-003/transcript.txt`, 5 of 5 held |
| Storage unit | `artifacts/tensor/h02-storage-002`, `storage_koi_device_passed` |

Preserved failed attempts: `h02-build-001` (backend rebuilt after a pin change),
`h02-storage-001` (new mandatory symbols missing from a close-failure stub),
`h02-smallconv-001` (backend pin), `-002` (wrong capture path), `-003` (padding
rows observed without the declared pre-fill), `-004` (receipt field names).
`-005` and `-006` are two independent runs of the same pinned inputs; every
substantive field of the two receipts is identical, which is the determinism
check. `-006` is the recorded receipt.

The replay refuses to run unless the backend manifest is ABI 4, CUDA family
13.0, status `built`, and every declared build input still matches on disk; and
unless the capture's declared graph shape, stage list, offsets, extents and
layouts agree with the Go plan. A drifted plan is a rejection, not a warning.

## Non-claims

- No final convolution, recurrent layer, projection or score is executed here.
- Byte-identical convolution stages are not a basecall, an accuracy measurement
  or biological evidence.
- The nine padding rows are proven unwritten, not value-compared. The capture
  declares them unwritten from source, and H02 never compares their contents.
- `nt_cuda_weights_permute_cast_f16` is not on the caller path; it exists only so
  the pure-Go packer can be checked bit for bit.
- The receipt's `input_sha256` map pins the backend build inputs as well as the
  capture payloads, so "pinned" here means every declared input of both the
  library and the oracle.
