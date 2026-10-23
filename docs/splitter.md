# Simplex DNA PA pore-adapter decisions

`internal/splitter.FindSimplexPA` computes the standalone SQK-LSK114 nonrapid
simplex DNA decision from already normalized binary16 signal, sequence,
quality, binary moves, and stride. It returns half-open spacer intervals. It
does not create children or change read metadata. The caller must select this
explicit chemistry contract before calling it. RNA, duplex, rapid muA, and
other scaling modes have no entry point here.

The source is Dorado `8b8fc5d36a9c0baab262a743cb175b5878e38ca6`.
`dorado/api/pipeline_creation.cpp:92-101` selects simplex mode and the PA
threshold. `dorado/splitter/include/splitter/ReadSplitter.h:46-99` defines its
settings. `dorado/splitter/DuplexReadSplitter.cpp:534-596` runs only
`PORE_ADAPTER` for the nonrapid simplex contract.

## Data and ownership

`Input` borrows `Signal []uint16`, `Moves []uint8`, `Sequence`, `Qstring`, and
`Stride uint64`. Signal contains normalized IEEE binary16 bits. It is not raw
POD5 signal. Calls are pure and reentrant. Callers must keep borrowed slices
unchanged during a call. Every returned slice belongs to the decision.

`Decision` retains the following intermediates.

| Field | Coordinates and meaning |
| --- | --- |
| `Clusters` | Merged sample intervals, exclusive end, later tied argmax, original peak bits |
| `MoveSums` | Inclusive move cumulative sums |
| `PoreCandidates` | One record per cluster with move indices, original base span, one-base argmax interval, local mean and rejection reason |
| `RawCandidates` | All width and quality eligible base intervals, sorted by coordinates |
| `Candidates` | Top ten by native peak and pair ordering, then sorted by coordinates |
| `AdapterChecks` | Pore interval, clipped search interval, first native match interval and acceptance |
| `Spacers` | Candidate intervals accepted by the nearby adapter check |

`PoreCandidate.BaseSpan` becomes meaningful after the move-bound checks pass.
Its `Range` and `MeanQ` become meaningful after the width check passes.
`Eligible` means it passed these local gates. It may still fall below the top
ten or have no adapter. Adapter match coordinates are meaningful only when
`Found` is true. A missing match uses a zero interval.

The boundary rejects empty move tables, mismatched sequence and quality lengths,
nonbinary moves, wrong move base counts, nonpositive or overflowing native
strides, and inconsistent sample counts. It accepts both floor and ceiling move
counts for a partial final stride. DNA is uppercase A, C, G, T, or N. Quality
must be printable positive Phred, matching the source `qscore_mean` assertion.
Nonfinite binary16 signal is rejected before computation. Native sorting of
NaN peak values has no usable ordering contract. Errors return a zero decision
and do not mutate inputs.

## Preserved source behavior

`dorado/splitter/splitter_utils.cpp:125-300` scans after sample 5000 using strict
comparison with `c10::Half(2.8f)`, bits `0x419a`. It merges gaps of 500 samples
or less. Equal peaks choose the later sample and the later merged cluster.
Duration zero keeps single-sample spikes. The Go scan uses the source's signed
binary16 ordering. All admitted peaks are finite and positive, so their raw
bits also give the native peak ordering.

`dorado/utils/sequence_utils.cpp:444-454` supplies inclusive cumulative sums.
`dorado/splitter/DuplexReadSplitter.cpp:147-179` divides both the exclusive
sample end and the argmax by stride. An end move index outside the table
rejects the cluster. A start move with zero called bases also rejects it. The
base start is `move_sums[start] - 1`, the base end is `move_sums[end]`, and the
argmax base is `move_sums[argmax] - 1`. Width above 500 rejects the region
before argmax reduction. The resulting candidate excludes exactly one base.

`DuplexReadSplitter.cpp:106-117,180-199` uses a literal arithmetic mean over
five Phred values, clipped at the sequence end. Binary32 rounds
`10 - numeric_limits<float>::epsilon()` to 10. A mean of exactly 10 passes.
The source sorts `(peak, (start,end))` ascending, takes the last ten, and sorts
those intervals by coordinates. Equal peaks therefore favor later intervals.

`DuplexReadSplitter.cpp:29-53,331-340` aligns the distinct internal adapter
`TACTTCGTTCAGTTACGTATTGCT` with edlib HW/LOC, edit distance at most four. The
target is `[pore.start,min(pore.end+50,sequence.length))`. It uses the first
native location and converts the inclusive alignment end to a half-open end.
N has strict equality here. `internal/align` is shared with the end detector,
which separately requests N wildcard equality.

## Design choices

A configurable splitter would expose settings for unimplemented RNA, duplex,
and rapid branches. The fixed `FindSimplexPA` entry point keeps those modes
outside the type's contract. A spacer-only return would hide whether a
disagreement came from signal detection, coordinates, quality, ranking, or
alignment. The decision retains those intermediates so composition and review
can locate the first disagreement.

The Model the Domain principle shaped the fixed input and decision types.
The throughput checkpoint has four parts. Source tracing and the native
dependency inventory precede implementation. This owner handles the whole
splitter unit, while the sibling owns shared alignment. Production calls share
no mutable state, and verification uses fresh attempt directories. A single
owner keeps each decision and its oracle together. Nested delegation, commits,
and pull requests are parent-owned under this task's scope.

## Native comparison

`verification/splitter` pins eight complete Dorado files and the original edlib
source and header at revision `42aa8fa7051fb2aa1c4f144c5eda22a0e753f026`.
It extracts complete source bodies unchanged. These include
`detect_pore_signal`, `fast_half_comparable`, `move_cum_sums`, `qscore_mean`,
`find_best_adapter_match`, `filter_ranges`, `ExtRead`, `create_ext_read`,
`possible_pore_regions`, and `check_nearby_adapter`. The driver compiles the
original `ReadSplitter.h` and source-built edlib. The production package
imports no native runtime or model backend.

ATen CPU tensors provide the actual binary16 storage and accessor path. The
read field container, class declarations, inline class constructor, and
no-op logger are test-only declarations. They implement no pore, mapping,
quality, ranking, alignment, or filtering algorithm. The oracle uses the
stock PA settings. A separate all-candidate probe changes only
`top_candidates` to `INT32_MAX` to expose pre-ranking eligible intervals. The
stock top-ten result is compared independently.

Each fresh build retains full copied source, unchanged extracted bodies,
compiler identity, accessed ATen header hashes, executable and linked runtime
hashes, exact command, UTC clocks, exits, and raw logs. Runs retain binary
inputs, complete Go decisions, and exact binary outputs. Full source hashes
are checked before and after building. No result cache is used. Native
comparison processes have a 120-second limit. The protocol limits each count
to 67,108,864 elements and a batch to 100,000 reads. Go bounds native stdout
to 512 MiB and stderr to 4 MiB. Binary output is compared as bytes, so forged
output counts cannot trigger allocations in an output decoder.

Repository, ordinary, race, no-cgo, vet, and behavior mutation checks run through the
maintained command.

```sh
verification/splitter/verify.sh
```

Native qualification also requires the genuine capture manifests and their
retained source-verified move receipt. `caller.Replay` validates the captures
without model inference. The checker verifies their complete sequence,
quality, and move endpoints against that move receipt before using them.

```sh
DORADO_SOURCE="$PWD/external/dorado-source" \
SPLITTER_TORCH=/path/to/CPU13/libtorch \
SPLITTER_EDLIB_SOURCE=/path/to/original/3rdparty/edlib \
SPLITTER_CAPTURE_MANIFESTS=/path/to/hac/manifest.json:/path/to/sup/manifest.json \
SPLITTER_MOVES_RECEIPT=/path/to/replay-moves/replay/receipt.json \
verification/splitter/verify.sh --native
```

The wrapper hashes its Go sources and shared alignment before and after all
checks. Failed attempts remain under ignored `artifacts/splitter`. Ordinary
tests skip native qualification when its explicit paths are absent.

## Evidence limits

This unit does not establish child creation, inherited metadata, default caller
composition, whole-read processing, native GPU inference, stock CUDA12 parity,
biological split accuracy, or performance. Retained HAC and SUP reads qualify
genuine no-split controls. Synthetic adapter-supported spacers qualify positive
algorithm decisions. They are not positive biological split evidence.
