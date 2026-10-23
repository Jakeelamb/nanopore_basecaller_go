# Plain DNA interval trim and filter

`internal/readtrim.ApplyInterval` implements Dorado commit
`8b8fc5d36a9c0baab262a743cb175b5878e38ca6` plain DNA synchronized trimming.
`Filter` applies the streaming five-base minimum and optional mean-quality cutoff.
These are standalone units. They do not establish default-pipeline parity,
new inference, biological accuracy, or GPU acceptance.

## ApplyInterval

`Interval{Start, End}` retains the half-open base interval. `MuxRead` carries
current sequence, quality, binary moves, normalized binary16 signal, and stride.
`leadingTrim` is the prior sample trim, including scaling trim.

The result borrows strings and slices. The function never modifies input storage.
Copy slices before modifying shared data. Trimmed slices cap their capacity at
the retained end. Full intervals return identity, including signal beyond the
last complete stride and zero moves before the first base. Other intervals remove
leading moves before the retained base, retain following zero moves until the
next removed base, and slice the signal at move coordinates times stride.
Signal slices clamp to the actual length at partial tails.

An interior empty interval clears sequence, quality, moves, and signal without
adding leading trim. An empty read with `[0,0)` returns identity. `[n,n)` for a
nonempty read is an error because the native string helper rejects start equal
to sequence length. Negative, inverted, and out-of-range intervals are errors.
The caller must skip invalid detector intersections before this helper, matching
TrimmerNode's separate no-op policy.

`LeadingTrimSamples` adds removed leading move positions times stride.
`NumSamples` is the current signal sample count plus that cumulative trim.
`PreTrimSequenceLength`, `NumMinknowEvents`, mux state, and stride remain unchanged.
Original start sample, start time, identity, and split metadata remain caller-owned.
RNA, modified bases, barcodes, and duplex signals have no admitted API branch.

Boundary validation accepts uppercase ACGTN, printable quality bytes 33 through
126, binary moves with a sum equal to sequence length, finite binary16 samples,
a positive signed-32-bit stride, and signal lengths whose floor or ceiling
stride count equals the move count. Native string slicing itself accepts arbitrary
bytes. The Go API deliberately narrows that domain. Shape validation reuses the
frozen MUX validator after mapping N to A only for validation. Signed native
leading-sample multiplication overflow and unsigned cumulative sample overflow
are rejected before return.

## Filter

`Filter(seq, q, meanQStart, minQ)` requires equal sequence and quality lengths,
a nonnegative quality offset, the same sequence and quality alphabet above,
and a finite float32 threshold. `FilterResult.Reason` is a bit set with
`FilterShort` and `FilterLowQuality`; both failures can be present. `Keep` is true
only when neither failure is present. Equality at the Q threshold passes.
Zero or negative thresholds disable quality filtering. There is no default Q7
or Q10 cutoff. `MeanQ` is still computed for metadata when filtering is disabled.

When quality length exceeds `meanQStart`, the mean uses the suffix at that offset.
Otherwise it uses the full quality string. `decode.MeanQScore` supplies the
existing float32 accumulation and native float libm semantics. CGO-disabled
builds return `decode.ErrCGORequired` from `Filter`; interval slicing still works.

The native node's constructor takes an integer `size_t` Q threshold. This API
extends its exact comparison predicate to finite float32 thresholds. Integer
thresholds are the source-supported CLI domain. Fractional thresholds are an
explicit API extension, exercised with the same native predicate and float32
parameter. Read-ID exclusion and output-directory pass/fail classification are
outside this API.

## Native verification

`verification/intervaltrim/verify.sh --native` runs ordinary, race, CGO-disabled,
vet, behavior mutation, source mutation, and native differential checks. Run from
the repository root with `DORADO_SOURCE` set to the pinned source root and
`INTERVALTRIM_CAPTURE_MANIFESTS` set to HAC and SUP capture manifests in that
order, separated by the platform path-list separator.

Each attempt retains command clocks, status, raw logs, current Go source hashes,
Go version, and Git head under ignored `artifacts/intervaltrim`. Native builds
retain full pinned source files, seven complete unchanged functions, the exact
ReadFilterNode condition, the driver, compiler hash and version, executable hash,
linked runtime hashes, binary inputs, and exact Go and native outputs.

The oracle compiles the complete `Trimmer::trim_sequence(SimplexRead&,...)`,
`trim_torch_tensor`, string `trim_sequence`, `trim_move_table`, template
`trim_vector`, `mean_qscore_from_qstring`, and `ReadCommon::calculate_mean_qscore`.
Its container shim exposes only accessed fields. RNA helpers throw if reached.
Modified-base probability vectors are inert because mod-base metadata is null.
The signal shim copies unchanged uint16 bits for positive one-dimensional,
unit-step slices and clamps both ends to the signal length. Self-checks cover
nonzero starts, empty intervals, partial tails, and starts beyond the end.
This proves storage slicing without claiming ATen arithmetic.

Synthetic cases include zero moves at both ends, interior and whole intervals,
empty and single-base reads, partial tails, positive prior trims, and random
synchronized inputs. Filtering covers lengths zero, four, and five; float32
threshold neighbors and equality; offset zero and 60; and shorter, equal, and
longer quality strings. Retained HAC `[38,1653)` and SUP `[31,1650)` cases compare
stored normalized signals and original decoded outputs without model execution.
Capture manifest and payload hashes are checked before comparison.
