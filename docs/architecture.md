# Go caller ownership

The target gives Go ownership of signal processing, model and decoder decisions,
scheduling and output. Native kernels remain behind typed tensor operations.
The implemented CPU HAC and SUP paths join POD5 input, scaling, fixed chunks, Go
model graph, CRF scans, beam decoding, stitching, and FASTQ output. Default
trimming and splitting, GPU execution, and complete input accounting remain open.

The reference is Dorado commit
`8b8fc5d36a9c0baab262a743cb175b5878e38ca6`. The original experiment checkout
and its dirty files remain outside this repository and must remain unchanged.

## Two designs

| Design | Go owns | Native boundary | Tradeoff |
| --- | --- | --- | --- |
| Read pipeline with chunk engine | Input metadata, calibration, planning, batching, stitching and output | Complete native model graph, forward pass and GPU decoding | Small interface and easier initial integration, but substantial caller logic remains C++. An intermediate port only. |
| Model runtime with kernel backend | All of the above plus config and weights, model graph, tensor layouts, workspace planning and decoder scheduling | Pinned tensor and GPU compute operations | Selected full-port target. More ABI and layout work, but caller decisions have one Go owner. |

An executable wrapper or a whole-read native API does not meet the target. Avoid
shipping both designs as interchangeable production paths. A coarse chunk-engine
adapter, if introduced during integration, must be identified as remaining native
scope and replaced before claiming the full target.

The native CUDA model outputs are contiguous binary16 tensors in `[N,T,C]`
order. The implemented CPU graph uses float32, with binary16 normalized inputs
expanded exactly. Native CUDA decoding calls guide, beam, posterior and decode
kernels. Preserve their
order, dtypes and layouts before considering numerical changes. See
[`CRFModel.cpp:108-131`](https://github.com/nanoporetech/dorado/blob/8b8fc5d36a9c0baab262a743cb175b5878e38ca6/dorado/basecall/model/CRFModel.cpp#L108-L131)
and
[`CUDADecoder.cpp:95-126`](https://github.com/nanoporetech/dorado/blob/8b8fc5d36a9c0baab262a743cb175b5878e38ca6/dorado/basecall/decode/CUDADecoder.cpp#L95-L126).

## Types and owners

| Module | Knowledge and state owned |
| --- | --- |
| `internal/signal` | Calibration, pA normalization, binary16 rounding, signal trimming and scaling metadata |
| `internal/chunk` | All three sample-coordinate planners and repeat-padding |
| `internal/stitch` | Decoded-chunk validation and final sequence, quality and moves |
| `internal/model` | Source-pinned config, weights and Go model graph |
| `internal/tensor` | Typed CPU float32 and int64 handles, primitives and native lifetime |
| `internal/crf` and `internal/decode` | CPU guide/posterior scheduling and beam decisions |
| `internal/caller` | Selected reads, cancellation, fixed chunk collection, capture/replay and FASTQ |
| `internal/pod5` | Copied input metadata, selected UUIDs and native signal decompression |
| Future GPU backend | Native allocations, streams, events and kernel ABI |

The implemented assembly interface is:

```go
type Config struct {
    RawSamples uint64 // after signal trimming
    Stride     uint64
    Variable   bool
}
type Chunk struct {
    Offset, RawSize, Granularity uint64
    Sequence, Qstring            string
    Moves                       []uint8
}
type Result struct {
    Sequence, Qstring string
    Moves             []uint8
}
func Stitch(Config, []Chunk) (Result, error)
```

Chunks arrive in input-offset order. `RawSize` is the nominal model chunk size
for fixed chunks and the unpadded interval length for variable chunks.
`Granularity` determines the padded decoded length. Input slices are borrowed
only for the duration of `Stitch`; outputs have independent storage. Quality is
an opaque byte string of the same length as the sequence. The upstream tests
include synthetic letters outside A/C/G/T, so assembly does not constrain the
sequence alphabet.

One collector owns each read's indexed results. Workers return immutable chunk
results and never append into a shared read. Queue limits must bound signal
bytes as well as job counts. Native asynchronous kernels must own native buffers
and must not retain Go pointers. Cancellation must drain outstanding work and
release allocations before the runtime closes. Failed chunks must prevent an
apparently successful partial read.

## Exact assembly behavior

The planner's last fixed start is rounded up to a stride boundary. Actual model
input preparation repeat-pads short chunks; it does not zero-pad them. Sources:
[`chunk.cpp:31-43`](https://github.com/nanoporetech/dorado/blob/8b8fc5d36a9c0baab262a743cb175b5878e38ca6/dorado/read_pipeline/base/chunk.cpp#L31-L43),
[`BasecallerNode.cpp:432-475`](https://github.com/nanoporetech/dorado/blob/8b8fc5d36a9c0baab262a743cb175b5878e38ca6/dorado/read_pipeline/nodes/BasecallerNode.cpp#L432-L475).

Assembly cuts an odd overlap asymmetrically: floor at the current chunk's rear,
ceil at the next chunk's front. Single chunks and variable final chunks first
discard repeat-padding. A final partial stride then loses exactly one move and,
if that move emits a base, one sequence and quality byte. The result satisfies
`sum(moves) == len(sequence) == len(quality)` and has
`floor(RawSamples / Stride)` moves. Source:
[`stitch.cpp:33-107`](https://github.com/nanoporetech/dorado/blob/8b8fc5d36a9c0baab262a743cb175b5878e38ca6/dorado/read_pipeline/base/stitch.cpp#L33-L107).

Go rejects invalid external data before assembly: nonbinary moves, inconsistent
decoded lengths, mismatched sequence/quality, crossed overlap cuts, missing or
gapped signal coverage, unaligned overlaps and coordinate/output overflow.
Coordinates are limited to the signed 32-bit domain used by the native routine's
casts. Granularity must be divisible by stride and move length must exactly match
the padded model length. These are the supported producer contracts; the native
routine uses assertions and does not safely define every malformed case.
No native-parity claim is made for rejected inputs. Empty chunks return an empty
result, matching the native early return.

## Native oracle and replay

`verification/stitch/oracle.go` checks hashes of the original `stitch.cpp`,
`stitch.h`, `math_utils.h` and `string_utils.h`, then compiles those untouched
files using the original NVTX include directory. Only `ReadCommon.h` is replaced
with a minimal test container. The adapter supplies raw sample count, stride and
sequence/quality/move storage. This verifies algorithm parity, not the original
ReadCommon ABI, ATen storage, native scheduler, model inference or full runtime.

```sh
go test ./internal/stitch
DORADO_SOURCE=/path/to/pinned/dorado-src go test -count=1 -v ./internal/stitch
```

The first command explicitly skips native tests without `DORADO_SOURCE`. The
second builds the CPU oracle and compares 726 cases: four upstream behavior
cases, empty and shorter-than-stride cases, and 720 deterministic randomized
fixed, balanced-variable and transformer-variable planner outputs. Decoder
outputs in those randomized fixtures are synthetic. The tests also reject
changed native source and malformed Go inputs. Set `STITCH_ARTIFACT_DIR` to an
ignored artifact directory to retain the executable, protocol inputs and outputs.

Content-addressed replay must key the source hashes, adapter/shim bytes, compiler
identity and flags, fixture inputs and expected output bytes. The coordinator's
parity command owns that manifest and cache policy. A stale pin invalidates the
cached claim. `BenchmarkStitch` consumes the result and can support CPU profiling;
it is not evidence of a speedup over Dorado.

## Remaining qualification

The official distribution contains CUDA 12 LibTorch runtime libraries. The
qualified CPU development backend uses the separately verified CUDA 13 ONT
LibTorch package and matching headers. POD5 0.3.39 and Koi 0.7.5 archives are
verified outside Git. Koi is not yet used by the Go caller. Matching stock CUDA
execution still requires its own backend and evidence. Dorado's build pins are
[`Koi.cmake:40-91`](https://github.com/nanoporetech/dorado/blob/8b8fc5d36a9c0baab262a743cb175b5878e38ca6/cmake/Koi.cmake#L40-L91),
[`Torch.cmake:49-95`](https://github.com/nanoporetech/dorado/blob/8b8fc5d36a9c0baab262a743cb175b5878e38ca6/cmake/Torch.cmake#L49-L95)
and
[`Pod5.cmake:4-35`](https://github.com/nanoporetech/dorado/blob/8b8fc5d36a9c0baab262a743cb175b5878e38ca6/cmake/Pod5.cmake#L4-L35).

The full caller still needs the boundaries listed in [the port plan](port-plan.md),
including GPU kernels, splitting, trimming, BAM and end-to-end
input accounting. CPU algorithm parity does not qualify native GPU behavior,
biological accuracy or performance. Retained HAC stage captures replay without
inference. Capture new expensive producer fixtures once and reuse them for
downstream work.
Use HG002 development and validation; never choose an implementation using HG005.
