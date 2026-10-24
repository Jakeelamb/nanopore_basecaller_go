# P02: typed CUDA storage ABI and Koi descriptors

Status is **device-captured on the admitted host**, one unit of the GPU caller.
It extends the existing single ABI with typed device storage and Koi
descriptors. It does not execute a model, and it is not end-to-end basecalling
or biological evidence.

## ABI decision

The pinned backend, probe, receipts and consumers all declare `NT_ABI_VERSION 4`
(`internal/tensor/abi.h:5`). P02 adds nine symbols
(`nt_cuda_storage_{new,view,meta,upload,download,copy,fill,scale,release}`) and
two descriptors (`nt_storage_meta`, dtype/tag enums) **without bumping the
version**:

- The addition is purely additive; every pre-existing symbol keeps its
  signature and semantics, so retained ABI4 receipts still replay.
- `loader.c` resolves every symbol at `nt_open`. A backend that lacks any new
  symbol is rejected with `missing ABI symbol nt_cuda_storage_*`, so an older
  ABI4 backend can never be mistaken for a P02-capable one.
- `verification/tensor/check` and `verification/gpuprofile.qualify` continue to
  require `abi_version == 4`.

Bumping to 5 would have invalidated the accepted ABI4 runtime identity for no
safety gain, because symbol resolution already fails closed.

## Pinned source

| Source | Pin | Lines used |
| --- | --- | --- |
| `dorado/nn/TxModules.cpp` | `9e73e81b38c6b6e36a17566a8dfc16e2da2e54fab44dfdbcb85485d77998b701` | `30-79` `KoiTensorExt`; `534-556` literal dim tags |
| `koi.h` (libkoi 0.7.5) | P01 pin | `3` `KoiTypeId`; `36-47` tensor dims |

Two rules are taken literally from `TxModules.cpp`:

- **Scale owners are I8-only with F32 data.** `KoiTensorExt` throws
  `KoiTensor scale tensor only compatible with I8 tensor` and
  `KoiTensor scale tensor must be F32` (`TxModules.cpp:36-47`). The Go API and
  the native guard both enforce I8 storage plus an F32 vector.
- **Tags are the literal values the source passes.** The named enum values are
  `KOI_DIM_MAT_Q_K_V=0`, `KOI_DIM_GATE_IFGO=1`, `KOI_DIM_MAT_W_U=2`, and the
  source also passes single characters `N T C M K H D W n t c m k d`
  (`TxModules.cpp:534-556`). P02 mirrors those values exactly; it does not
  invent a parallel tag numbering.

U8 has no `KoiTypeId`, so `Storage.KoiTensor()` rejects it.

## Surface

- `internal/tensor/abi.h`: `enum nt_dtype`, `enum nt_koi_tag`, `nt_storage_meta`
  (rank, dtype, device, ordinal, scale dtype/presence/tag, allocation,
  offset/storage elements, shape, strides, tags) and the nine entrypoints.
- `internal/tensor/loader.{h,c}`: symbol fields, `LOAD` entries and thin
  wrappers, so every call goes through the one loader.
- `verification/tensor/cuda_storage.inc`: the native implementation. It is
  included once from `backend.cpp` inside the single ABI translation unit, so
  the storage records share the engine/device/stream registries and the reviewed
  synchronous-drain contract (`CUDAWork`).
- `internal/tensor/cuda_storage.go`: `Storage`, `Device.NewStorage`,
  `View`, `Meta`, `Upload`, `Download`, `CopyTo`, `Fill`, `BindScale`, `Close`.
  Go validates rank, tags, dtypes, offsets, strides, span overflow, ownership
  and element counts before the call, and holds no pointers across it.
- `internal/tensor/cuda_koi.go`: `KoiTensor`, `KoiDim`, `KoiScale`, with the
  pinned `KoiTypeId` values and the I8/F32 scale rule.
- `internal/tensor/cuda_errors.go`: `DTypeError` now carries the rejection
  `Reason`, so a dtype rejection names the rule that fired instead of printing
  only handle and dtype ids. Go-side and native-side dtype rejections are
  therefore distinguishable in the evidence rather than by position alone.

Native guards (all in `cuda_storage.inc`):

| Guard | Rule |
| --- | --- |
| dtype | only F16, I8, U8, I32, F32, E4M3; unknown values reject |
| rank | 1..8, matching `NT_STORAGE_MAX_RANK` |
| tags | only the pinned named values and literal characters |
| shape | positive dimensions, overflow-checked element count |
| strides | non-negative, overflow-checked span |
| view bounds | `offset + span <= parent end`, and the view inherits the parent allocation |
| contiguity | upload, download, copy and fill require a contiguous view |
| overlap | a copy whose source and destination intervals intersect in one allocation rejects |
| device | every operand must belong to the same engine, device and ordinal as the stream |
| scale | I8 storage, F32 nonempty vector, known tag, not itself |
| scale owner | a storage that is the live scale owner of another storage cannot be released |
| release | a poisoned device retains storage until a successful close drain |
| foreign handle | an unknown or foreign-engine handle rejects as closed |

## Admitted capture

Host lock held for the whole run; no competing compute process was observed.

| Item | Value |
| --- | --- |
| Build | `artifacts/tensor/p02-build-004`, `abi_version 4`, `cuda_family 13.0` |
| Backend SHA-256 | `752f736e8383d554ad0c19887dce24f340636717e038285583dadbe9de29980c` |
| Capture | `artifacts/tensor/p02-capture-010`, status `storage_koi_device_passed` |
| Unit receipt | `storage-capture.json`, SHA-256 `85b7a05b2fef08a1d8418d80dd755f68213751133cd6410d6f316baabfb7a43f` |
| Device | `GPU-807d7ae7-6a99-c795-12c2-23fc801d38e7`, ordinal 0, driver 615.78.08 |
| Runtime | `driver=13040 runtime=13040`, CUDA 13.4, Torch 2.9.0.0+cxx11-abi |
| Consumer | head `6546dee`, 38 source pins, test binary SHA-256 `ae992c24a162...`, stable after execution |
| CPU contracts | 13 accepted CPU-side contracts pass against this CUDA-enabled build |
| Commands | 8 recorded with clocks and exits; 0 failing |
| Runtime pins | 34 mapped libraries, re-hashed outside the child |

The child re-hashes every mapped `libtorch*`, `libc10*`, `libcu*`, `libnv*`
object and the orchestrator re-hashes them again, rejecting any change and any
CUDA-12 or foreign Torch path. The runtime pin set is the *observed* one recorded
by the runtime probe, so it includes the host driver `libcuda.so.615.78.08`,
which is host-supplied and therefore cannot be a build pin.

### CPU contracts

The card requires the accepted CPU-side contracts to keep passing against the
CUDA-enabled build, so `storage-koi` runs them with the same observed runtime
pins as the device step and requires all thirteen roots:

`TestNativeTensorOperations`, `TestNativeOwnershipAndCopies`,
`TestNativeMalformedInputs`, `TestNativeLoadWeight`, `TestNativeConcurrentClose`,
`TestNativeMissingLibrary`, `TestNativeLoaderABIRejections`,
`TestNativeLoadedRuntimePins`, `TestNativeDecoderPrimitives`,
`TestNativeCloseFailureRetainsLibrary`, `TestNativeAttentionPrimitives`,
`TestNativeAttentionRejections`, `TestNativeABIBoundary`.

The list is one shared `cpuContractTests` value used by both the source-corpus
unit and the storage unit, so neither can silently drop a contract. Adding the
storage symbols also forced `TestNativeCloseFailureRetainsLibrary`'s stub backend
to declare all nine `nt_cuda_storage_*` entrypoints; without them `nt_open`
correctly fails closed on the missing symbol.

### Payload checks (exact bytes)

| Check | Bytes | Rule |
| --- | --- | --- |
| `f16-roundtrip` | 12 | allocate, upload, download, byte-equal |
| `i8-roundtrip` | 8 | allocate, upload, download, byte-equal |
| `f32-scale-roundtrip` | 16 | scale vector roundtrip |
| `e4m3-roundtrip` | 8 | FP8 payload roundtrip |
| `i32-roundtrip` | 12 | signed payload roundtrip |
| `u8-roundtrip` | 4 | unsigned payload roundtrip; no Koi type |
| `strided-view-guard` | 72 | a contiguous child view upload changes only its four elements of the parent |
| `strided-view-fill` | 72 | fill changes only the same four elements |
| `device-copy` | 72 | device-to-device copy reproduces the parent exactly |

Each check records `expected_sha256` and `observed_sha256`; the orchestrator
requires them to be equal and non-empty.

### Rejection controls

Every control records the native or Go-side rejection text. `native-*` controls
call the ABI directly so the native guard, not the Go pre-check, is the thing
under test.

| Control | Observed rejection |
| --- | --- |
| `u8-has-no-koi-type` | `koi_descriptor: half layout 0: u8 has no Koi type` |
| `overlapping-copy` | `storage_copy: half layout 16: storage copy source and destination intervals overlap` |
| `dtype-mismatch-copy` | `storage_copy: handle 8 dtype 1 requires 4: storage copy dtype differs` |
| `non-contiguous-upload` | `storage_upload: half layout 13: storage upload requires a contiguous view` |
| `non-contiguous-download` | `storage_download: half layout 13: storage download requires a contiguous view` |
| `scale-requires-i8` | `storage_scale: handle 4 dtype 1 requires 2: Koi scale owner requires an I8 tensor` |
| `scale-requires-f32` | `storage_scale: handle 4 dtype 1 requires 5: Koi scale owner must be F32` |
| `native-scale-requires-i8` | `storage_scale: handle 4 dtype 1 requires 2: Koi scale owner requires an I8 tensor` |
| `native-scale-requires-f32` | `storage_scale: handle 4 dtype 1 requires 5: Koi scale owner must be F32` |
| `scale-self` | `storage_scale: half layout 0: storage cannot own itself as its Koi scale` |
| `scale-unknown-tag` | `storage_scale: half layout 0: unknown Koi scale tag` |
| `scale-owner-release` | `storage_release: storage is still the Koi scale owner of a live view` |
| `fill-pattern-width` | `storage_fill: half layout 0: pattern length differs from element size` |
| `native-copy-dtype-mismatch` | `storage_copy: handle 17 dtype 4 requires 5: storage copy dtype differs` |
| `native-fill-pattern-width` | `storage_fill: half layout 11: storage fill pattern length differs from element size` |
| `view-exceeds-parent` | `storage_view: half layout 11: storage view exceeds parent interval` |
| `view-negative-stride` | `storage view: half layout 0: negative stride or span overflow` |
| `foreign-engine-handle` | `storage_meta: foreign engine handle (engine 1 ... requires engine 18 ...)` |
| `unknown-dtype` | `storage_new: handle 0 dtype 0 requires 1: unknown storage dtype` |
| `closed-download` | `tensor or engine is closed` |

Two controls exist specifically because the Go pre-checks would otherwise hide
the native guard:

- `view-exceeds-parent` uses a parent that is itself a view inside a larger
  allocation, so the escaping child is still inside the allocation and torch's
  own bounds check cannot reject it. Only `end <= parent->end` can.
- `native-copy-dtype-mismatch` copies between `I32` and `F32`, which share an
  element width, so the dtype guard is the only check that can reject the pair
  rather than the element-count guard.

## Required test

```sh
go test -count=1 ./internal/tensor -run '^TestCUDAStorageAndKoiDescriptors$'
```

The host-memory backend in that test exercises the real loader, cgo marshalling
and Go validation without a GPU. It covers roundtrip, strided view writes,
fill, copy, the Koi descriptor, the I8/F32 scale rule, the owner-release guard,
foreign-engine rejection, closed-handle behaviour and idempotent close. It is a
unit test, not native parity; the admitted capture above is the native evidence.

## Mutation controls

`verification/tensor/cuda/p02-controls.sh` removes one native guard at a time,
rebuilds the backend, reruns the storage capture and preserves the failed
attempt under `artifacts/tensor/p02-mutant-*`. The committed tree is the restored
one (`cuda_storage.inc` SHA-256 `39a3fe38...`). All seven mutations were observed
to break exactly the control that names the guard:

| Mutation | Guard removed | Observed failure |
| --- | --- | --- |
| `overlap` | interval-intersection check in `storage_overlap` | `overlapping-copy accepted` |
| `owner-release` | scale-owner scan in `nt_cuda_storage_release` | `scale-owner-release accepted` |
| `scale-dtype` | I8 requirement in `nt_cuda_storage_scale` | `native-scale-requires-i8 accepted` |
| `contiguity` | `is_contiguous` requirement | `non-contiguous-upload accepted` |
| `view-bounds` | `end <= parent end` | `view-exceeds-parent accepted` |
| `dtype-copy` | dtype equality in copy | `native-copy-dtype-mismatch accepted` |
| `fill-width` | pattern length check | `native-fill-pattern-width accepted` |

Each entry is the last line the failing consumer printed, taken from the
preserved `storage-koi-device.stdout` of that mutant attempt.

## Limits

- One unit on one device. No model, no graph, no kernel from the HAC or SUP
  callers was executed, so this card does not qualify those kernels.
- The host-memory fake backend is a transliteration of the native guards, not
  the native implementation; only the admitted capture is native evidence.
- Seven of the twenty rejection controls are Go pre-checks that fire before the
  native call; the mutation table names which ones actually exercise native
  code, and the two `native-*` controls exist so the native guard is not assumed.
- Elementary arithmetic over typed storage (GEMM, convolution, LSTM) belongs to
  the H, S and D cards. P02 provides storage, views and descriptors only.
- CUDA 12, foreign Torch, competing GPU processes and missing symbols are
  rejected by construction, not exercised as passing paths.
