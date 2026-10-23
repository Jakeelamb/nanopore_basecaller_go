# Go-owned POD5 reader

`internal/pod5` reads POD5 files directly through the POD5 0.3.39 C API.
Production code starts no Python or CLI subprocess. Go owns UUID lookup,
selection order, cancellation, bounds, and the lifetime of copied outputs.
Native POD5 owns Arrow decoding and signal decompression. This qualifies an
input reader, not native GPU inference, full input processing, basecalling,
biological accuracy, or performance.

## API and ownership

```go
reader, err := pod5.Open(libraryPath, filePath, pod5.Options{
    MaxReads: 1_000_000,
    MaxSamples: 100_000_000,
})
if err != nil {
    return err
}
defer reader.Close()
metadata, err := reader.Metadata(ctx)
if err != nil {
    return err
}
_ = metadata
return reader.Iterate(ctx, selectedIDs, func(read pod5.Read) error {
    return consume(read)
})
```

`OpenContext` also accepts a context during indexing. `IDs` returns copied UUIDs
in file storage order. `Read` accepts a binary `ID`, and `ParseID` accepts the
canonical UUID spelling. `Iterate` uses the supplied order and preserves
duplicates. A nil selection visits the entire file. A nonnil empty selection
visits nothing. An absent UUID fails before callbacks start.

`Read.Signal` contains unchanged signed 16-bit ADC samples. `NumSamples` must
match both the row metadata and complete native signal length. Calibration
scale, offset, median-before, and open-pore-level retain their float32 bits.
The reader preserves NaN open-pore-level values. It does not substitute a
predicted value or convert samples to picoamps.

`Read` embeds `RunInfo`. `RunID` means `protocol_run_id`, matching
`dorado/data_loader/DataLoader.cpp` lines 114-255 at Dorado source revision
`8b8fc5d36a9c0baab262a743cb175b5878e38ca6`. `AcquisitionID` remains separate.
Other metadata includes channel, well, read number, start and end samples,
end reason, pore type, sample rate, flow cell, kit, timestamps, context tags,
and tracking IDs. Empty experiment names fall back to `protocol_group_id`.
`EndSample` is the checked sum of start sample and sample count.

`Metadata` reads file identity, original on-disk version, and every run record
without decoding a signal. `FileMetadata.OriginalVersion` is not the producer's
migrated logical version. Chemistry inputs are available as flow cell product
code, sequencing kit, and sample rate. Chemistry classification remains the
caller's responsibility.

The default `MaxReads` is one million and bounds the metadata index. A file
beyond that bound is rejected rather than silently truncated. The default
`MaxSamples` is 100 million per selected read. Both values must fit the host
address space. Metadata queries do not apply `MaxSamples` because they decode
no signal. Pore-type and end-reason names have a 1,024-byte native buffer limit.
Longer names return an explicit native error.

All returned slices, strings, and maps belong to Go. The C wrapper keeps its
read batch and run-info owners alive until Go finishes copying. It then frees
all native result data. C retains no Go pointer. Each wrapper captures POD5's
thread-local error before returning to Go or running native cleanup.

A reader serializes native operations and `Close` with one mutex. Callbacks run
outside that mutex and may close the reader. `Close` is idempotent. Methods
called after close return `ErrClosed`. An unknown UUID returns `ErrNotFound`.
The process loads one canonical library path. Alternate library images are
rejected after initialization. Global Arrow registration happens once under a
C mutex. The library stays loaded for process lifetime, including after a
failed initialization attempt, and readers never call `pod5_terminate`.

Cancellation is checked before and after native read calls and between indexed
rows, run records, and callbacks. A running native call is not interrupted.
The caller must close each opened reader. Copying a reader after first use is
unsupported. Other platforms and `CGO_ENABLED=0` return `ErrUnsupported`.

## Pinned native interface and independent fixtures

`verification/pod5/source-pin.json` pins the native 0.3.39 distribution, copied
interface headers, shared library, and original fixture inputs. The upstream
archive is
`https://github.com/nanoporetech/pod5-file-format/releases/download/0.3.39/lib_pod5-0.3.39-linux-x64.tar.gz`.
The archive SHA256 is
`7fcf1d9f1da6f1bd13167782e900462fbadb0db57c666911bdba9bbdfd4e7e06`.
The shared-library SHA256 is
`d8c1d9380c66439520a4fc814acf82300c144c8e1332035b8351b643d351879c`.
Copied `c_api.h` and `pod5_format_export.h` remain unchanged. Their license is
retained beside them in `internal/pod5/pod5_format/LICENSE.md`.

The maintained fixture producer uses the stock workspace's locked
`pod5==0.3.49` and `lib_pod5==0.3.49`. This differs from native 0.3.39 deliberately.
It provides an independently maintained reader implementation and wrapper
against the same pinned inputs. It uses the POD5 library family, so it is not
an independent implementation of the file format.

The fixture producer reads one split-read fixture and exactly one HG002
smoke-selected raw signal. It also enumerates UUID metadata for both files.
No other HG002 signals are decoded. The selected read is the first UUID in
`data/splits/smoke.txt`. HG005 is unused.

| Fixture | Selected UUID | Samples | Raw little-endian int16 SHA256 |
| --- | --- | ---: | --- |
| Split read | `c31a90ab-b786-4dcd-b996-2d05d97e7e53` | 365,157 | `eff0dc8d5784dd8cab85ee59f36e8336aa514fddb670812ae91b733fb40b0cc3` |
| HG002 smoke | `bb76c31a-36f0-48a0-abd2-107a10e1ade0` | 19,856 | `931078657d3ef96ef9faeacd3fb0dad3197abeff4bf577a9cf8710b2b99bce4b` |

Calibration scale bits are `3e3f8f3d` for both. Offset bits are `c36a0000`
and `c3870000`, respectively. Open-pore-level bits are `7fc00000` for both.
All compared read and run metadata match the independent producer.

## Reproduce the bounded reader check

Use a fresh output directory for each oracle attempt. Run from the Go repository.
The original repository stays read-only. The producer rejects output paths
inside that workspace and existing output directories.

```sh
uv run --project /home/jake/Projects/nanopore_basecaller --locked python \
  verification/pod5/generate.py \
  --stock /home/jake/Projects/nanopore_basecaller \
  --output "$PWD/artifacts/pod5/fixtures-new"

NANOPORE_POD5_LIBRARY="$PWD/external/backend/pod5/lib/libpod5_format.so" \
POD5_FIXTURE_DIR="$PWD/artifacts/pod5/fixtures-new" \
  verification/pod5/verify.sh
```

The native tests require both environment variables. Without them they print
an explicit skip reason. The maintained script requires them and fails if
either is absent. It pins the loaded library hash and original input hashes.
It retains commands, exit codes, UTC clocks, source hashes, and raw logs in a
new `artifacts/pod5/verify-*` directory. No download or GPU is required.
The generated manifest and signals stay under ignored `artifacts/pod5`.

The runner executes these checks against the same source files.

```sh
go test -count=1 -race -v ./internal/pod5
CGO_ENABLED=0 go test -count=1 -v ./internal/pod5
go vet ./internal/pod5
```

Native acceptance checks byte-for-byte raw signal identity, exact float32 bits,
all compared metadata, every stored UUID in order, metadata-only reads, copied
buffer lifetime, missing UUIDs, malformed and empty files, bounds, cancellation,
callback errors, repeated close, concurrent read and close, and two-reader global
initialization lifetime. The default CPU test does not establish native parity.

The first expanded file-metadata acceptance attempt failed. Python's
`file_version` reports a migrated logical version, while the C API reports the
original on-disk version. Tracing the maintained Python properties identified
`original_file_version` as the matching field. Both fields remain recorded in
the fixture manifest. The failed receipt is retained at
`artifacts/pod5/verify-20261023T022451Z-cxFxDe/native.log`.
The corrected native race check, explicit no-cgo rejection, and vet check passed
at `artifacts/pod5/verify-20261023T022558Z-ckSzGQ/`.

## Design checkpoint

The pinned C API was traced before implementation. A C++ adapter would add a
second build without improving the read contract, so the reader uses a thin C
`dlopen` wrapper. A Go-owned index avoids persistent native traversal state.
This keeps filtering and selection order under Go ownership.

Blocking work was source tracing and independent fixture generation. This unit
has one owner for `internal/pod5`, `verification/pod5`, and this document. Other
port units use separate paths. Only POD5's process-global registration requires
shared serialization. Per-file handles and returned Go data remain separate.
The parent owns independent review and integration. No nested workers or commits
were created for this unit.
