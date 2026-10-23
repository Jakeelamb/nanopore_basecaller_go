# MUX tail trimming

`internal/readtrim.MuxChangeTrim(MuxRead) (MuxResult, error)` is a standalone Go
port of Dorado's mux-change tail transform at
`8b8fc5d36a9c0baab262a743cb175b5878e38ca6`. Caller and capture replay now apply it
after stitching. See the [caller contract](caller.md) and
[integration review](independent-review-mux-integration.md). Native
`BasecallerNode` applies this transform even with `--no-trim`; it is separate
from adapter and primer trimming.

`MuxRead` contains DNA sequence, ASCII qualities, binary moves, normalized
binary16 signal bits, model stride, the mux end-reason flag, MinKNOW events,
and the pre-trim sequence length. `MuxResult` returns the transformed read,
a trim flag, half-open sequence/move/signal prefix ends, and `NumSamples`.
`NumSamples` is current signal length. The caller retains sample clocks,
identity, leading scaling trim, and quality-start offset.

The function never mutates its input. Output strings and slices borrow the
input's storage. Trimmed slices have their capacity limited to the retained
prefix. No-op slices keep their original capacity. A caller that mutates either
input or output must copy first or guarantee exclusive access. Concurrent
read-only calls are supported. `PreTrimSequenceLength` becomes the retained
sequence length on trimming, and MinKNOW events become zero. No-op reads keep
both values.

The exported boundary rejects mismatched sequence/quality lengths, bases
outside A/C/G/T, qualities outside ASCII 33 through 126, nonbinary moves,
a move sum different from sequence length, nonfinite binary16 signal,
zero stride, and stride above the native signed 32-bit range. It bounds
sequence length to `MaxInt32-10`, move count to `MaxInt32`, and checks sample
multiplication overflow. Those bounds protect the source's signed accumulator,
string interval conversion, and Go slice coordinates. For partial strides,
move count can equal either floor or ceiling of signal length divided by
stride. Other signal/move relationships are rejected, including for non-mux
and short no-op reads. Large count guards are inspected; tests do not allocate
multi-gigabyte sequences to exercise those guards.

The source's behavior is preserved without repairing its indexing. Quality
categories are Q0 through Q7 with score -1, Q8 through Q12 with score +1,
and Q13 upward with score +10. The reverse scan updates on a tied minimum,
sets the candidate end to `i-1`, and exits when cumulative score exceeds `i`.
The candidate becomes an exclusive string end. The excessive-trim threshold
is `floor(float32(length) * float32(0.3))`. Length below 100 is a no-op.
Candidate ends at or beyond `length-5` are also no-ops. Consequently, five
removed bases still do not trim; six is the first accepted removed length.
A candidate end below the 30 percent threshold is a no-op, while equality
passes. The original move-table function locates that base's move position;
zero moves before that position remain. Signal ends at move count times stride,
clamped to the original length as with native tensor slicing.

## Verification

Ordinary Go checks require no native libraries or cgo. The native test and
behavior-mutation test report explicit skips unless their environment is set.

```sh
go test -count=1 -v ./internal/readtrim ./verification/readtrim/...
verification/readtrim/verify.sh
```

`verify.sh` records ordinary tests, race tests, `CGO_ENABLED=0` tests, vet,
and isolated Go behavior mutations. It never enables native qualification by
inheriting `DORADO_SOURCE` alone. Explicit native qualification additionally
requires both environment variables below.

```sh
DORADO_SOURCE="$PWD/external/dorado-source" \
READTRIM_CAPTURE_MANIFESTS="$PWD/artifacts/caller/attempt-20261023T023928Z-Wr38Jl/capture/bb76c31a-36f0-48a0-abd2-107a10e1ade0/manifest.json:$PWD/artifacts/caller/attempt-20261023T033758Z-xTMh1T/capture/bb76c31a-36f0-48a0-abd2-107a10e1ade0/manifest.json" \
verification/readtrim/verify.sh --native
```

Each invocation creates a fresh ignored `artifacts/readtrim/checks-*` directory.
`READTRIM_ARTIFACT_ROOT` overrides its parent. `inputs.json` records the Git head,
Go version, and hashes of the final package, verification tools, and this document.
Each check retains exact arguments, environment overrides, UTC clocks, exit
status, and stdout/stderr hashes beside raw logs. `summary.json` records overall
acceptance. Nested native and mutation attempts remain present on failure.

The native executable compiles these four complete functions byte-for-byte
from hash-pinned original files.

- `find_mux_change_trim_seq_index` and `mux_change_trim_read` come from
  `dorado/read_pipeline/base/read_utils.cpp`.
- `sequence_to_move_table_index` comes from `dorado/utils/sequence_utils.cpp`.
- The string overload of `trim_sequence` comes from `dorado/torch_utils/trim.cpp`.

The read shim exposes only fields accessed by those functions. The original
`ReadCommon.h` is also hash-pinned to audit their types. The signal shim stores
`uint16_t` bits and implements only a one-dimensional positive unit-step slice.
It clamps the end to storage length and copies the selected bits. Every native
process first checks empty, zero-length, prefix, exact-end, overlong-end, and
copy-isolation cases, including signed zero, subnormal, and finite extreme bits.
Native results therefore qualify this explicit storage shim, not general ATen
operations or storage aliasing. The Go result borrows storage while the shim
copies; exact values and ranges are compared, not allocation identity.

Native attempts retain full original source files, exact function slices,
the complete shim/driver, compiler flags and clocks, compiler executable hash,
linked runtime library hashes, executable hash, binary inputs, and native and
Go binary outputs. Output comparison covers every sequence, quality, move,
signal bit, stride, mux flag, event count, pre-trim length, and sample count.
Lengths in the output also establish the exact retained prefix ends.

The 892 deterministic synthetic cases cover positive and no-op reads, Q7/Q8
and Q12/Q13, ties, early-exit input patterns, lengths 99/100, candidate trims
four/five/six, excessive trimming below/at/above threshold, leading/interior/
trailing zero moves, floor/ceiling partial strides, and seeded random qualities.
Three larger cases at lengths 3495262 through 3495264 distinguish native
binary32 rounding from a float64 replacement. Boundary rejection controls
exercise malformed sequence, quality, moves, stride, signal length, NaN,
and infinity. Twelve isolated Go mutations must fail observable assertions.
Altering any of the four pinned source files must prevent native compilation.
No source extraction or metadata output is hand-reimplemented as an oracle.

Retained HAC and SUP inputs validate the manifest payload hashes, model run
metadata, normalized signal bits and sample count, and decoded output bytes.
Both have end reason `signal_positive`. They qualify non-mux identity only.
Their move tables are retained Go producer outputs. [Native CPU beam and stitch
replay](replay-moves.md) now reproduces them from the retained Go CPU13 tensors.
Official stock-binary moves remain unproven because its FASTQ contains no moves.
No positive biological MUX fixture,
whole ATen backend, pipeline integration, basecalling, GPU, performance, or
biological-accuracy claim follows from this package's checks. Verification
performs no download, model load, inference, or stock basecaller invocation.

## Delivery checkpoint

Blocking work was the exact four-function trace and metadata type audit.
Implementation, oracle, and checks form one owned unit because they share the
same input and output contract. No mutable input is shared with another writer.
Caller/model/tensor integration is outside this unit. No nested agents or
commits are required for this handoff. The parent coordinator owns independent
review, repository-wide integration checks, and any later pull request.
