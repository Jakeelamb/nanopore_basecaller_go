# Unaligned BAM writer

Status is host-verified source implementation. The writer has been compared
against one retained stock simplex parent record through htslib, not against a
native BAM. This is not GPU evidence, not end-to-end basecalling and not
biological evidence.

`internal/bam` owns BAM header text, record serialization, BGZF framing and the
typed source tag order in pure Go. `verification/bam` derives a writer manifest
from a retained SAM, writes a BAM from it, and separates the canonical, header,
literal-record and raw-byte verdicts.

## Source pins

- `read_pipeline/base/messages.cpp:108-212` fixes the simplex tag order, types
  and emission conditions, and `:404-424` fixes the mean-quality window.
- `read_pipeline/base/messages.cpp:425-520` fixes the mandatory unmapped fields,
  quality conversion and read-group attachment.
- `read_pipeline/nodes/ReadToBamTypeNode.cpp` fixes the final-record contract
  the FASTQ and BAM writers share.
- `hts_utils/bam_utils.cpp:43-80` fixes the `@RG` text and the
  `runid_model[_barcode]` identity rule.
- `hts_utils/header_utils.cpp` and `hts_utils/hts_file.cpp` fix header text,
  BGZF framing and finalization.

## Contract

The header is deterministic: `@HD VN SO`, one caller `@PG` (`ID PN VN [CL]
[DS]`), optional `@RG` with `PU PM DT PL DS LB tm` in that order, then `@SQ`
lines. Empty source values render as `unknown`; `experiment_id` is omitted when
absent. The read-group ID is `runid_model`, plus `_barcode` for a classified
barcode; `unclassified` never joins.

A record is unmapped (`flag=4`), `refID/pos/next_refID/next_pos = -1`,
`mapq=0`, `bin=4680`, no CIGAR, zero template length, a NUL-terminated QNAME,
4-bit packed sequence and phred bytes converted from the FASTQ quality string.
Tags are appended in the pinned order with their pinned type bytes: `qs:f`,
`du:f`, `ns:i`, `ts:i`, `mx:i`, `ch:i`, `st:Z`, optional `TS:A`/`RX:Z`,
`rn:i`, `fn:Z`, `sm:f`, `sd:f`, `sv:Z`, `dx:i`, optional `RG:Z`, optional
`pi:Z`+`sp:i`, optional `mv:B:c` (first element is the model stride), optional
`po:Z`, optional `er:Z`, then `me:I`. `Tags` emits `mv` only when `Moves` is
non-nil and rejects a missing stride. RNA poly-tail tags (`pt`/`pa`) are outside
the pinned plain-DNA path and are not emitted.

BGZF uses one deflate member per 0xff00-byte block, a stored-block fallback when
compression would exceed 64 KiB, a 28-byte EOF marker, and first-error
propagation. Short writes, `ENOSPC`, double close and writes after close never
report success.

## Commands

```sh
go run ./verification/bam/cmd manifest -sam ORACLE.sam -out NEW_ATTEMPT
go run ./verification/bam/cmd verify -plan PLAN.json -attempt NEW_ATTEMPT -out NEW_RESULT
```

The manifest is bound to the oracle SHA256 and is rejected if it differs from
the manifest derived from that oracle. `verify` writes `unaligned.bam`, the
normalized oracle, the htslib-decoded SAM, per-mutation control BAMs and, last,
`receipt.json`.

## Verdicts

- `quickcheck-unmapped`: `samtools quickcheck -u` exits zero.
- `canonical-records`: `internal/recordcompare` sees the same canonical record
  multiset (tags sorted).
- `decoded-header`: the htslib-decoded header text hash matches the oracle.
- `literal-records`: decoded record text is byte-identical, which covers tag
  order and formatting.
- `aux-order`: the declared tag order survives htslib.
- `bam-deterministic`: a second serialization has the same SHA256.
- `raw_bam`: unclaimed. No retained native unaligned BAM exists; the oracle is
  SAM text transcoded by samtools.

## Controls

`comparator-self`, `changed-tag`, `tag-reorder` (canonical-positive,
raw-negative), `duplicate-record`, `missing-record`, `header-mutation`,
`truncated-bgzf` and `truncated-record` all run on every verification. The
comparator self-check is not evidence of a Go writer. Writer faults are covered
by `internal/bam TestWriterShortWrite` and `TestWriterENOSPC` (`/dev/full`).

## Host evidence

```sh
go test -count=1 ./internal/bam ./verification/bam
go test -race -count=1 ./internal/bam ./verification/bam
go test -count=1 ./...
```

The retained oracle is
`/home/jake/Projects/nanopore_basecaller/logs/stock-baseline-audit-20261022/foreign-native-parent.sam`
(SHA256 `ec703a0eb17d890efc99a4b498275cde95c7a85ef6fa30ba2fcf77b7ca19b11e`).
Its one record decodes to the same literal SAM line as the Go-written BAM, and
the BAM passes `samtools quickcheck -u`. Receipts are under
`artifacts/gpu-plan/bam-writer-host-*`.

## Open

The oracle has a single simplex parent with no `mv`, `pi`, `sp`, `TS`, `RX` or
duplex-parent tags, so those paths are unit-tested but not oracle-differential.
Raw BAM identity, larger read sets, other modes, a native BAM oracle and
independent review remain open. The writer is not yet wired to caller output;
`ReadToBamTypeNode` metadata beyond the current postcall seam is still missing.
