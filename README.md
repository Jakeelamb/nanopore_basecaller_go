# Go nanopore basecaller port

A separate Go port of caller computations from Dorado 2.1.2, pinned to
`8b8fc5d36a9c0baab262a743cb175b5878e38ca6`.

The Go CPU caller reads POD5 and produces HAC or SUP FASTQ. On one predeclared
HG002 development read, both models match unchanged Dorado byte for byte,
including headers, sequence, and qualities. With trimming disabled, HAC emits
1,653 bases and SUP 1,650. Default adapter/primer processing emits 1,615 and
1,619 bases and matches separate unchanged stock controls using the same saved
Go inference outputs.
These bounded results do not qualify other reads, GPU execution, biological
accuracy, or performance.
The original experiment repository remains the stock control.

Implemented in Go:

- Fixed, balanced variable, and transformer chunk planning, plus repeat-padding.
- Standardised DNA pA scaling, binary16 conversion, metadata, and initial trim.
- CPU float32 CRF beam search from supplied logits, guides and posteriors.
- Move-based chunk stitching into sequence, quality, and read moves.
- POD5 selection and metadata through the native POD5 decompressor.
- The pinned HAC convolution/FLSTM and SUP convolution/transformer graphs using
  native CPU tensor primitives, plus Go-owned CRF scans and posteriors.
- A CPU caller CLI with immutable stage captures and replay without inference.
- Source-linked mux-change tail trimming, including its always-on behavior with
  `--no-trim`. Positive fixtures are synthetic; the genuine controls are non-mux.
- Plain DNA SQK-LSK114 pore-adapter splitting, child identity and metadata,
  adapter/primer detection, synchronized trimming, and minimum-length filtering.
  Positive split fixtures are synthetic; the genuine controls do not split.

Native differential tests compile pinned original C++ algorithms or audited
source slices. [The architecture](docs/architecture.md), [port coverage](docs/port-plan.md),
and [decoder qualification](docs/decode.md) describe the boundaries and open work.
The [caller instructions and retained endpoint](docs/caller.md) describe the
working mode. GPU execution, batching, BAM output, and broader input acceptance
remain open.

## Call and replay

Build with `go build -o bin/basecaller ./cmd/basecaller`. The caller takes
`--model`, `--pod5`, `--tensor-library`, `--pod5-library`, and selected
`--read-id` values or explicit `--all`. Use `--device cpu`, `--chunksize 12288`,
`--overlap 600`. Adapter/primer trimming and pore-adapter splitting are enabled
by default for the supported kit. `--no-trim` and `--disable-read-splitting`
disable these stages independently. See the
[complete invocation](docs/caller.md#run-one-selected-read).

`--capture NEW_DIRECTORY` retains raw signal, normalization, scores, guides,
posteriors, decoding, and final output with hashes. Reuse those stages:

```sh
bin/basecaller replay --manifest CAPTURE/READ_UUID/manifest.json > replay.fastq
```

Replay checks payload integrity and repeats normalization, decoding, and
stitching. It loads no model or POD5 library and runs no inference.

## Verify without sequencing computation

Use Go 1.24 or newer with CGO enabled, GCC, a C++20 compiler, Git, and samtools.
The Go decoder uses native float32 libm operations to preserve rounding.
The controlled parity command requires Linux, GCC's default compiler names,
and no external compiler overlays or library-path overrides. It records the
native numerical-library hashes and runs child tests with `GOWORK=off`.
`verify.sh` clears inherited dynamic-library path overrides for that command.
Obtain the pinned source and NVTX headers:

```sh
./verification/setup-source.sh
./verification/verify.sh --source "$PWD/external/dorado-source"
```

For an existing pinned source checkout, supply its path with `--source`.
`portcheck` records code, source and compiler identities, UTC clocks, exit
status, raw logs, native fixtures and hashes under ignored `artifacts/parity/`.
It reuses successful evidence only when the identity and retained outputs match.
`--refresh` forces a new attempt. Skipped or missing native tests fail acceptance.
Failed attempts remain separate. Code changes during a run invalidate it.
This command qualifies the listed CPU-core packages. Tensor and model integration
have separate native acceptance commands and cannot inherit the CPU-core verdict.

`go test ./...` runs the ordinary CPU tests. Native tests explicitly skip unless
`DORADO_SOURCE` points at the pinned checkout, so that command alone is not a
native parity claim. [CPU CI](.github/workflows/cpu-parity.yml) runs the pinned
native algorithm checks. It does not run inference or acquire experiment cohorts.
The nested `artifacts/go.mod` keeps preserved evidence source files out of
normal Go package discovery. All generated evidence remains ignored by Git.

Compare retained output files without basecalling again:

```sh
go run ./cmd/readcompare original.bam candidate.bam
```

The JSON separates raw byte identity from literal SAM-record equality after
sorting records and optional tags. Headers are hashed separately and excluded
from that record comparison. See [the comparator contract](cmd/readcompare/README.md).

## Continue the full caller

The [backend tool](docs/backend-readiness.md) acquires exact development packages
into ignored storage, verifies checksums, and preserves resumable downloads.
Its prerequisite check cannot qualify a working backend. Separate evidence covers
the [CPU tensor ABI](docs/tensor-backend.md), [HAC graph](docs/model-graph.md),
[POD5 reader](docs/pod5-reader.md), and [CRF scan](docs/crf-scan.md).
The qualified development backend uses CPU operations from the pinned CUDA 13
Torch distribution. Its match to one stock CUDA 12 CPU endpoint does not
establish general runtime compatibility. Reuse captured outputs when their pins
match. GPU execution requires a separate implementation and acceptance.
The [SUP QKV half-weight preparation](docs/koi-qkv-weights.md) now matches
native CPU source blocks exactly. It prepares host buffers for future GPU use
and does not run in the current caller.

Changes use feature branches and pull requests targeting `main`. See
[AGENTS.md](AGENTS.md) for the experiment and evidence rules.

Derived Dorado code retains the upstream research-use license in
[LICENCE.dorado.txt](LICENCE.dorado.txt). See [NOTICE](NOTICE).
