# Go nanopore basecaller port

A separate Go port of caller computations from Dorado 2.1.2, pinned to
`8b8fc5d36a9c0baab262a743cb175b5878e38ca6`.

The port is in progress. It does not yet accept POD5 and produce basecalls.
Inference, native guide/posterior generation, input loading, splitting,
scheduling, and final caller output remain open. The original experiment
repository is the stock control.

Implemented in Go:

- Fixed, balanced variable, and transformer chunk planning, plus repeat-padding.
- Standardised DNA pA scaling, binary16 conversion, metadata, and initial trim.
- CPU float32 CRF beam search from supplied logits, guides and posteriors.
- Move-based chunk stitching into sequence, quality, and read moves.

Native differential tests compile pinned original C++ algorithms or audited
source slices. [The architecture](docs/architecture.md), [port coverage](docs/port-plan.md),
and [decoder qualification](docs/decode.md) describe the boundaries and open work.
These CPU checks do not establish a complete caller, GPU parity, biological
accuracy, or performance improvement.

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

`go test ./...` runs the ordinary CPU tests. Native tests explicitly skip unless
`DORADO_SOURCE` points at the pinned checkout, so that command alone is not a
native parity claim. [CPU CI](.github/workflows/cpu-parity.yml) runs the pinned
native algorithm checks. It does not run inference or acquire experiment cohorts.

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
Its prerequisite check cannot qualify a working backend. The next implementation
boundary is the native tensor/kernel ABI, followed by Go model execution and
captured native intermediates. Reuse development outputs when their pins match.

Changes use feature branches and pull requests targeting `main`. See
[AGENTS.md](AGENTS.md) for the experiment and evidence rules.

Derived Dorado code retains the upstream research-use license in
[LICENCE.dorado.txt](LICENCE.dorado.txt). See [NOTICE](NOTICE).
