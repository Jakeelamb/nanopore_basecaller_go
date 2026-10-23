# Default DNA end detection

`internal/ends` implements standalone sequence end detection for the initial
SQK-LSK114 target. It does not run inference, trim a read container, split reads,
or format FASTQ. Caller composition remains a separate qualification.

## API and ownership

```go
result, err := ends.Detect(sequence, sequencingKit, ends.DefaultOptions())
```

`Options{Adapters, Primers}` selects the searches. The zero value disables both.
`DefaultOptions()` enables both. The input sequence must contain uppercase
`ACGTN` and fit signed 32-bit native coordinates. Empty sequences are supported.

The kit is uppercased. `SQK-LSK114` is the admitted stock target. An empty kit or
explicit `ANY` returns `KitMissing`, searches no stock candidates, and retains
the whole sequence. Every other explicit kit returns an error. This admission
boundary deliberately excludes other native kits, custom sequences, RNA, and
extended primer sets. Missing metadata never selects SQK-LSK114 implicitly.

`Result` contains the following values.

- `Adapters` and `Primers` each preserve `Raw` candidate choices, `Accepted`
  choices after trim acceptance, and a half-open `Retained` base interval.
- Each end contains a binary32 `Score`, a `Name`, and inclusive `Start` and `End`
  alignment coordinates. Native negative sentinels remain visible.
- Rejected ends retain their score and coordinates and change their name to
  `unclassified`.
- `PrimerClassification` contains a name and orientation. Forward is `1`,
  reverse is `-1`, and unknown is `0`.
- The final `Retained` interval is the adapter and primer interval intersection.
  `Conflict` identifies an invalid intersection. A conflict retains the whole
  sequence and preserves the independently computed primer classification.

All values belong to one call. There is no shared cache, read metadata, mutation,
or synchronization requirement. The caller owns sequence slicing and subsequent
signal, move, quality, clock, and metadata updates.

## Matching behavior

The stock SQK-LSK114 adapter is `LSK110`. Its front query is
`CCTGTACTTCGTTCAGTTACGTATTGC` and its rear query is `AGCAATACGTAACTGAAC`.
The front and rear adapter windows contain at most 75 bases each.

The default primer is `cDNA`, including for this genomic DNA target. Its source
sequences are `TTTCTGTTGGTGCTGATATTGCTGGG` and
`ACTTGCCTGTCGCTCTATCTTCTTT`. Forward queries use the front sequence and the
reverse complement of the rear. Reverse queries use the rear sequence and the
reverse complement of the front. Primer windows contain at most 150 bases.
`10X_Genomics` is excluded by the source's default `PrimerAux` filtering.

`internal/align.HW(query, target, equality)` aligns the complete query to a target
substring. Its `Location` has edit distance, inclusive coordinates, and `Valid`.
`EqualityExact` matches identical bytes. `EqualityNWildcard` additionally makes
`N` equal to `A`, `C`, `G`, or `T` in either sequence. Empty queries and unknown
equality values return errors. An empty target returns the query length as its
distance, coordinates `-1,-1`, and `Valid=false`, matching native missing LOC
pointers.

The production implementation uses two DP rows and carries the earliest start
through equal-distance cells. It selects the first native end location. Original
edlib has a query-block boundary detail. When query length is divisible by 64,
its unpadded final block does not report the empty target prefix. The Go matcher
preserves that behavior. The final oracle batch includes 63, 64, 65, 127, 128,
and 129-base query controls. The actual admitted end queries are 18 to 27 bases.

A binary32 score is `1 - float32(distance)/float32(query length)`. Across candidate
queries, improvement strictly greater than `0.1` wins. Within `0.1`, a longer
alignment span wins. Exact span ties keep the earlier candidate. The two source
conditions execute independently, including their original floating-point
rounding.

Scores at least `0.8` pass trim acceptance. The retained start is one past the
front match's inclusive end. The retained end is the rear match's start. Empty
or inverted per-search intervals reset to the whole sequence and unclassify
both ends. An invalid adapter and primer intersection also leaves the sequence
untrimmed.

## Source custody and native oracle

The Dorado source pin is `8b8fc5d36a9c0baab262a743cb175b5878e38ca6`.
The edlib pin is `42aa8fa7051fb2aa1c4f144c5eda22a0e753f026`.
`verification/ends/oracle.go` contains full-file SHA256 pins for every used
Dorado source and header and for the original edlib source and header.

The oracle compiles the full unchanged `demux/AdapterDetector.cpp`,
`demux/adapter_info.cpp`, and `models/kits.cpp`. It extracts unchanged candidate
tables, default manager construction and candidate filtering from
`demux/adapter_primer_kits.cpp`. It also extracts the adapter-score overload of
`Trimmer::determine_trim_interval`, scalar reverse-complement functions, source
result types, and the SimplexRead intersection block. It preserves function
bytes and records hashes of whole files, individual detector functions, slices,
shims, compiler inputs, compiler tools, executable, and linked runtime libraries.

Source selection follows these native locations.

- `demux/AdapterDetector.cpp:40-58,74-130,178-285` owns equality, alignment,
  scoring, query transforms, candidate choice, and primer classification.
- `demux/adapter_primer_kits.cpp:29-114,228-248,291-347` owns sequences, kit
  mapping, default construction, and auxiliary-primer filtering.
- `demux/adapter_info.cpp:15-29` maps the GEN10X extension.
- `demux/Trimmer.cpp:93-124` owns threshold acceptance and interval reset.
- `read_pipeline/nodes/AdapterDetectorNode.cpp:153-187` owns kit handling and
  final interval intersection.

The oracle supplies only standard container plumbing, a no-op logger, an unused
HTS include, a no-op tracing-range macro, and protocol serialization. Original
result types are source slices. The original reverse-complement lookup table
comes from its pinned header. Forced standard includes supply the original
build's transitive declarations. Custom-primer construction throws explicitly.
There is no handwritten detector algorithm in the oracle and no system or
prebuilt edlib archive.

The builder requires a new output directory. It verifies the copied source
snapshot before extracting or compiling. Native attempts preserve inputs,
stdout, stderr, clocks, exits, and hash receipts. The checker mutates all 17
pinned source inputs in isolated copies and proves rejection before compilation.
`testdata/captures.json` pins the retained HAC and SUP manifests and outputs.
`implementation-inputs.json` records the tested Go sources, verification inputs,
checkout head, Go runtime, and Go executable hash.

## Reproduce and evidence

Run from the repository root.

```sh
verification/ends/verify.sh
```

The default native edlib input is the read-only existing checkout at
`/home/jake/Projects/nanopore_basecaller/software/source-build-qualification-20261023/source-input/dorado/3rdparty/edlib`.
Use `-edlib PATH` for another checkout with identical pinned source bytes.
Use `-output NEW_DIRECTORY` to name a new retained attempt. No model or GPU is
required. All generated evidence stays outside Git under ignored `artifacts/`.

The accepted local run is `artifacts/ends-parity-attempt-qualified`, recorded by
`artifacts/ends-parity-qualified`. It passed all of the following comparisons.

| Check | Observed result |
| --- | --- |
| Original source mutation controls | 17 inputs rejected |
| Native HW/LOC alignment | 34,248 exact location and distance comparisons |
| Reverse-SHW alternative | Zero differences from native |
| Latest-start alternative | Rejected by 6,419 differences |
| Full native detector | 376 exact structured results |
| Candidate score and span controls | 150 results |
| Threshold and interval controls | 120 results |

The full detector cases include no-hit and `N` matches, clipped and overlapping
windows, front and rear matches, both primer orientations, conflicting primer
names, disjoint accepted intervals, all flag combinations, missing kit, GEN10X
exclusion, seeded cases, and both genuine retained captures. HAC retains
`[38,1653)` and SUP retains `[31,1650)` in this standalone detector check.

These additional commands passed with raw receipts at the corresponding
`artifacts/ends-*-final` paths.

```sh
go test ./internal/align ./internal/ends ./verification/ends -count=1 -v
go test -race ./internal/align ./internal/ends ./verification/ends -count=1
env CGO_ENABLED=0 go test ./internal/align ./internal/ends ./verification/ends -count=1
go vet ./internal/align ./internal/ends ./verification/ends/...
```

Ordinary tests skip native helper comparisons unless `ENDS_NATIVE_ORACLE` is set.
The parity checker sets it, retains the native helper requests and responses,
and requires the named native tests to pass. Source-custody tests skip when the
local pinned source dependencies are absent. Pure Go behavior tests still run.

Earlier attempts remain intact. `artifacts/ends-build-first` exposed the missing
Go exclusion tag for the oracle C++ file. `artifacts/ends-build-second` exposed
selection of the BAM intersection block instead of the SimplexRead block.
`artifacts/ends-parity-second` preserves the initial 64-base empty-prefix
mismatch. The final run follows those corrections and checks both 64 and 128.

This evidence qualifies the standalone sequence detector and its tested shared
alignment behavior. It does not qualify complete default caller behavior,
biological accuracy, complete input processing, inference, GPU execution,
performance, or metadata integration.
