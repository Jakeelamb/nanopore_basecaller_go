# Standalone child materialization

`internal/subread` implements one plain simplex DNA split round from supplied,
sorted half-open base spacers. It does not detect pores or adapters, call a model,
or integrate caller output. RNA, duplex and modified-base state have no fields in
its admitted read type.

`Materialize(Read, []Interval)` returns zero or more reads. `Read` owns the actual
read metadata. `Inherited` stores the scalar and string fields that native
`shallow_copy_read` preserves, including calibration and scaling as binary32 bits.
`ClientToken` only preserves an opaque client identity. It does not implement the
native client's behavior. A separate generic metadata map would conceal misspelled
or omitted fields, so the native shim and Go type use explicit fields.

Inputs remain immutable. Returned signal slices borrow the input's normalized
binary16 storage like native ATen views. Actual children own copied move tables.
Strings are immutable slices. No-spacer reads retain their original move buffers.
Callers must keep borrowed signal and move storage unchanged while using it.
Concurrent calls have no shared mutable package state.

The pinned source is Dorado `8b8fc5d36a9c0baab262a743cb175b5878e38ca6`.
The package preserves the following source behavior.

- `splitter/DuplexReadSplitter.cpp:496-531` maps each move equal to one to its
  sample index times stride. It appends signal length as the final map entry.
  Actual splitting starts at the first called base's sample, dropping leading
  zero-move signal. Empty base intervals and intervals with no whole stride are
  omitted. A retained final child includes a partial signal tail, while its move
  slice still ends at floor(signal length / stride).
  With a ceil-sized input table this can leave a child's move sum one below its
  sequence length. A tail-only base with no whole stride remaining is omitted.
- `splitter/splitter_utils.cpp:34-109` slices the child, copies inherited fields,
  resets events to zero and read number to -1, sets pretrim sequence length,
  resets leading trim, and advances sample coordinates. Split point deliberately
  wraps to uint32. A child inherits the existing root parent or the original ID.
- `read_pipeline/base/read_utils.cpp:18-65` omits mean quality start offset from
  its shallow copy. Actual children therefore reset it to zero. The source
  helper's default factory is represented explicitly in the oracle shim.
- `DuplexReadSplitter.cpp:599-627` restores the input's end reason and mux flag
  only on the final emitted child. Earlier children use unknown and false.
- `DuplexReadSplitter.cpp:630-680` bypasses splitting when sequence or moves are
  empty. A root without spacers retains identity and metadata. A nonempty read
  with an existing parent still receives derived ID index zero and count one
  without spacers. Multiple children chain their IDs and retain the parent's
  outer neighbors. One child retains both outer neighbors.
- `utils/uuid_utils.cpp:12-38` hashes the verbatim parent ID plus decimal child
  index with SHA256. The first 16 bytes receive version 4 and RFC variant bits.
  `DeriveUUID` uses Go's standard library and uint64 indices for Linux size_t.
- `subread` rounds relative milliseconds from absolute sample coordinates before
  adding acquisition start. `StartTimeMS` holds those relative milliseconds.
  `StartTime` holds the absolute UTC string with six fractional digits, matching
  `utils/time_utils.cpp`. Root no-spacer clocks are unchanged.

The exported boundary rejects malformed DNA/quality, nonbinary moves, mismatched
nonempty sequence/move counts, inconsistent floor/ceil stride tables, invalid
spacers, zero stride or sample rate, and coordinate or clock overflow. Read counts
and stride are at most int32 maximum, read number fits int32, and sample rate is
at most uint32 maximum. Acquisition start and absolute child timestamps are
limited to nonnegative milliseconds through `9223372036854`, or
`2262-04-11T23:47:16.854000+00:00`. This is the last whole millisecond representable
by the pinned Linux native `std::chrono::system_clock`, whose signed 64-bit
duration counts nanoseconds. `utils/time_utils.cpp:26-28` converts through that
clock before formatting. The next millisecond is outside its defined domain.
The Go boundary rejects it rather than reproducing native signed overflow.
Each actual child's acquisition start plus rounded relative milliseconds must
fit this bound. Its signal end does not constrain its start timestamp. If any
child fails, `Materialize` returns an error and no partial children. No-spacer
and native bypass paths retain the original clock fields without conversion.
Acquisition start and sample-coordinate validation still apply on these paths.
Other native clock representations require separate qualification.
Signal values are opaque uint16 bits, including nonfinite encodings, because
materialization only slices them. Empty sequence or empty moves take the native
bypass before map construction. `EndSample` and `NumSamples` on input remain
independent from normalized signal length because earlier scaling or tail
trimming can change the retained normalized interval.

## Verification

Run from the repository root.

```sh
verification/subread/verify.sh
DORADO_SOURCE="$PWD/external/dorado-source" verification/subread/verify.sh --native
```

The checker runs ordinary tests, the race detector, cgo-disabled tests, vet,
isolated behavior mutations, and the requested native differential gate. Each
fresh attempt under ignored `artifacts/subread` retains command arguments, UTC
clocks, exit codes, raw streams, hashes, and a source-freeze comparison. Failures
remain retained.

The native oracle extracts complete unchanged `moves_to_map`, `shallow_copy_read`,
`assign_subread_parent_id`, `subread`, `subreads`, `apply_split_finder`, and `split`
functions. It compiles full original UUID, OpenSSL SHA256, and timestamp units.
The flat uint16 storage shim implements only shared immutable storage, dimension
zero size, and checked half-open slicing. The split-finder seam returns supplied
spacers and never runs detection. Explicit read containers preserve every field
the source functions access. Modified-base containers remain empty and RNA and
duplex flags remain false. This qualifies the represented fields, not the native
POD5 layout, client implementation, ATen internals or full read container.

Build manifests hash original source, slices, driver, all compiler dependency
headers, compiler executable and version, resulting executable, and resolved
runtime libraries. Differential payloads compare every represented field and
every signal bit. Fixtures cover multiple and zero children, leading zero moves,
floor and ceil partial tails, root identity, existing parent identity, neighbor
chains, uint32 wrap, rounded clocks, mean quality reset, opaque calibration bits,
and deterministic sparse tables. Direct UUID probes cover indices zero, one,
nine, ten and large uint64 values. Rejection controls alter native source,
materialized metadata and payloads. Isolated Go mutations retain a parent's mean
quality offset, drop inherited run ID, or floor a child clock and must fail.

Clock regression tests cover the last admitted and first rejected millisecond,
including acquisition plus rounded sample offsets, leading trim, spacer offsets,
relative-only timestamps, and rejection after an earlier valid child. Accepted
boundary children compare every represented field with unchanged native source.
The native shim statically checks its signed 64-bit nanosecond clock and offers
`--clock-boundary ACQUISITION_MS ROUNDED_OFFSET_MS`. This probe derives its limit
from `system_clock::duration::max()`, calls unchanged native timestamp formatting
at accepted endpoints, and rejects the first overflowing sum with exit 2 before
source conversion. Rejected probes qualify the shim's representability guard,
not the undefined behavior of an overflowing native conversion.

The two hash-pinned retained HAC/SUP captures provide supplied-no-spacer identity
controls only. Their presence does not prove pore detection, native move
production, positive biological splitting, full FASTQ, inference, GPU execution,
accuracy or performance. Synthetic fixtures drive positive splitting. No new
inference or download is needed.
