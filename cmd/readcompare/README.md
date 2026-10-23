# Exact canonical record comparison

```sh
go run ./cmd/readcompare LEFT RIGHT
```

Each input is SAM text or BAM. Gzip/BGZF and uncompressed BAM magic identify BAM
without a filename extension. A `.bam` extension also requests BAM decoding.
The only subprocess is `samtools view --no-PG -h -- PATH`, which decodes existing
alignments. No basecaller or GPU computation runs. Samtools must be on PATH for
BAM inputs. Plain text inputs need no native tools.

The JSON report compares multisets of complete alignment records. It preserves
all 11 mandatory fields literally, sorts optional fields, and tolerates record
order changes. Duplicate QNAMEs and supplementary records remain valid. Duplicate
complete records retain multiplicity. Optional tag names must be unique within
one record. Malformed fields or tags produce a JSON error and exit status 1.

Headers are excluded from equality. Their decoded text is hashed separately.
`raw_sha256` hashes input file bytes. `canonical_sha256` hashes sorted canonical
record lines, including repeated lines and LF delimiters. `byte_identical` is
separate from `canonical_equal`. The report does not normalize reverse strands,
CIGARs, coordinates, numbers, or reference names. It is not a claim of biological
or general alignment equivalence.

Exit status 0 means canonical record equality. Exit status 1 means mismatch,
invalid input, decoding failure, or incorrect arguments. The command reports
at most five different-record examples, each truncated to a 512-byte prefix
with its full canonical record hash and counts. It bounds lines at 64 MiB,
canonical input text at 256 MiB, and records at 1,000,000 per input. Decoded header
hashes use LF endings. Floating tag values must be finite.

Run `go test ./internal/recordcompare ./cmd/readcompare`. The BAM integration test
skips explicitly if samtools is unavailable. Reordering, field/tag changes,
record multiplicity, malformed input, raw hashes, and bounded examples have unit
coverage. A stock BAM self-comparison verifies artifact ingestion only. It does
not demonstrate a Go basecaller producing equivalent records.
