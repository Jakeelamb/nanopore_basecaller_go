# Unaligned BAM writer

Status is host-verified source implementation with one native payload
differential and one native conditional-tag differential. The Go writer
reproduces the exact **uncompressed payload** of one retained stock simplex BAM:
same 733,623 bytes, same SHA256, including QNAMEs and exact float32 bits. A
second oracle compiles the pinned C++ tag emitters against htslib and shows
byte-identical aux for every conditional path, including `TS`, `RX`, `pi`,
`sp`, `mv`, `BC`/`bv`/`bi`, `pt`/`pa` and the duplex tag set. Compressed
bytes and BGZF member topology are explicitly not claimed. This is not GPU
evidence, not end-to-end basecalling and not biological evidence.

`internal/bam` owns BAM header text, record serialization, BGZF framing and the
typed source tag order in pure Go. `verification/bam` parses the retained native
BAM, derives a writer manifest from its exact records, writes a BAM from that
manifest, and separates the payload, compressed, BGZF-topology, header,
record-bit, canonical and literal-text verdicts.

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

The header is deterministic and pinned to `@HD VN:1.6 SO:unknown`, one caller
`@PG` (`ID PN VN [CL] [DS]`), optional `@RG` with `PU PM DT PL DS LB tm` in that
order, then `@SQ` lines. Empty source values render as `unknown`;
`experiment_id` is omitted when absent. A wrong `VN`/`SO`, a second `@PG`, a
`PP` link or an unsupported header field is rejected rather than normalized.

A record is exactly `flag=4`, `refID/pos/next_refID/next_pos = -1`, `mapq=0`,
`bin=4680`, no CIGAR, zero template length, a NUL-terminated QNAME, 4-bit packed
sequence and phred bytes converted from the FASTQ quality string. Any other flag
is rejected.

Tags are appended in the pinned order with their pinned type bytes: `qs:f`,
`du:f`, `ns:i`, `ts:i`, `mx:i`, `ch:i`, `st:Z`, optional `TS:A`/`RX:Z`,
`rn:i`, `fn:Z`, `sm:f`, `sd:f`, `sv:Z`, `dx:i`, optional `RG:Z`, optional
`pi:Z`+`sp:i`, optional `mv:B:c` (first element is the model stride), optional
`po:Z`, optional `er:Z`, then `me:I`. `Tags` emits `mv` only when `Moves` is
non-nil and rejects a missing stride. `B:i` and `B:f` arrays are supported for
the tags that use them. RNA poly-tail tags (`pt`/`pa`) and barcode tags
(`BC`/`bv`/`bi`) are not emitted by the default plain-DNA path and are rejected
by omission.

The writer rejects unsupported or inconsistent requests instead of silently
mis-encoding them: duplex reads (`dx=1` tag set is not ported), moves without a
stride, `Trimmed` outside `[0, NumSamples]`, a `Duration` that disagrees with
`NumSamples/SampleRate`, an `RG` tag that does not name the declared read
group, and B-array subtypes other than `c`/`i`/`f`.

BGZF uses one deflate member per 0xff00-byte block, a stored-block fallback when
compression would exceed 64 KiB, a 28-byte EOF marker, and first-error
propagation. Short writes, `ENOSPC`, `EPIPE`, double close and writes after
close never report success.

## Commands

```sh
go run ./verification/bam/cmd manifest -bam ORACLE.bam -out NEW_ATTEMPT
go run ./verification/bam/cmd verify -plan PLAN.json -attempt NEW_ATTEMPT -out NEW_RESULT
go run ./verification/bam/cmd tags-oracle -dorado-source ROOT -out NEW_RESULT
```

The manifest is bound to the native BAM SHA256 and is rejected if it differs
from the manifest derived from that BAM. `verify` writes `unaligned.bam`, the
oracle and Go payloads, the native and Go decoded SAM, per-mutation control
BAMs and, last, `receipt.json`.

## Verdicts

- `payload` (**claimed**): the inflated BAM payload is byte-identical to the
  native BAM payload.
- `compressed` (**unclaimed**): Go `flate` is not zlib, so compressed bytes are
  reported but never claimed.
- `bgzf` (**unclaimed**): native member boundaries follow the producer flush
  schedule; the native file has 17 members and the Go writer 13. Only the
  inflated payload is claimed.
- `quickcheck-unmapped`: `samtools quickcheck -u` exits zero.
- `header-text-identity` / `header-bits`: the Go header text and parsed header
  struct match the native header.
- `record-bits`: every record matches, including QNAME, sequence, quality,
  tag order, tag types and exact float32 bit patterns.
- `qname-set`: the QNAME sets are equal. This closes the earlier redacted-SAM
  gap.
- `rg-consistency`: every record's `RG` names the declared `@RG`.
- `canonical-records` / `literal-records`: the htslib-decoded SAM is equal as a
  canonical multiset and as literal text.
- `aux-order`, `bam-deterministic`, `bgzf-eof`: tag order survives htslib,
  a second serialization has the same SHA256, and the EOF marker is present.

## Controls and faults

Twelve controls execute on every verification: `comparator-self`, `changed-tag`
(1 ULP float), `tag-reorder` (payload-negative, canonical-positive),
`duplicate-record`, `missing-record`, `record-reorder` (payload-negative,
canonical-positive), `header-mutation`, `dangling-rg`, `duplex-reject`,
`flag-reject`, `truncated-bgzf` and `truncated-record`.

Five writer faults execute against synthetic writers: short write, `ENOSPC`,
`EPIPE`, double close and write after close. A control or fault that does not
behave as declared fails the receipt. The tags oracle adds two mutation
controls and thirteen cases, with no skip. No required test skips.

## Host evidence

```sh
gofmt -l internal/bam verification/bam
go vet ./internal/bam ./verification/bam/...
go build ./...
go test -count=1 ./internal/bam ./verification/bam
go test -race -count=1 ./internal/bam ./verification/bam
go test -count=1 ./...
```

The oracle is the stock run's stdout:
`/home/jake/Projects/nanopore_basecaller/logs/20261022T211637Z-stock-9be0f8151f6b42108bed076f7772c664-4ZOlqN/stdout`
(426,057 B, 32 records, 17 BGZF members, SHA256
`2345bdbbba817a3c42ace948abacb873a12931041dea01a3b05551f82a14ef81`).
Its inflated payload is 733,623 B with SHA256
`d8729e8b2b1f714f42b1e134651467ee287e49a30bd5cce485841343478e75c9`.
The Go writer produces the same payload hash. Receipts are under
`artifacts/gpu-plan/bam-writer-native-*` (ignored).

The earlier receipt `artifacts/gpu-plan/bam-writer-host-20261023T192601Z` used
`stock-baseline-audit-20261022/foreign-native-parent.sam` as its oracle. That
SAM is a lossy derivative of the native BAM: its QNAMEs were redacted to zeros
and its float tags were rounded (`qs 0x41a37b96` became `0x41a37b7f`, `sm
0x42bbffff` became `0x42bc0000`). Those passes were self-consistent but
non-probative for native bytes. That receipt is superseded.

## Conditional-tag oracle

The retained native BAM has no `mv`, `pi`, `sp`, `TS`, `RX`, `BC`, `bv`, `bi`,
`pt` or `pa` tag, so the payload differential cannot reach those paths. A
second, narrower oracle does: `verification/bam/native/tags_oracle.cpp` is a
template whose marked regions are replaced by byte-exact slices of the pinned
source, compiled against real htslib 1.24 and run on synthetic `ReadCommon`
values.

Pinned whole-file inputs, verified before any slice is taken:

| file | SHA256 |
| --- | --- |
| `dorado/read_pipeline/base/messages.cpp` | `5db12169f5eb376bfa70425a5f17fa31eab6be5d027d681caec81dd105ba35e0` |
| `dorado/utils/sequence_utils.cpp` | `d421325f78981c6710c67d50f9a6bc417701009f49bfd49373ac0fbab66b40f2` |
| `dorado/hts_utils/include/hts_utils/hts_types.h` | `653f2f71de8bc57847349154bd45a6f66120cb56fcaf847a8ccc9b694a1ae65b` |
| `dorado/read_pipeline/base/include/.../ReadCommon.h` | `a8ff4bced80c61f07070d6f5ed17ad3c2367c4f7d2a4564f54b5a7e47aec01cb` |

The spliced functions are `generate_read_group`, `generate_read_tags`,
`generate_duplex_read_tags`, `calculate_mean_qscore`, `extract_sam_lines`,
`find_rna_polya`, `mean_qscore_from_qstring`, `to_char`, `Attributes`,
`PrimerClassification`, `BarcodeScoreResult` and `StrandOrientation`. The
receipt records each slice offset and hash.

Shims, all documented in the template: the `ReadCommon` class body,
`get_raw_data_samples` (the pinned accessor reads an `at::Tensor` dimension, so
the oracle stores the same count), `generate_modbase_tags` (the pinned body
returns early whenever `mod_base_info` is null, which is every case here), and
`trace_log` (a logging helper with no effect on tag bytes). The oracle is a tag
differential, not a full Dorado build.

Thirteen cases run, twelve ported and **byte-identical**:

- Simplex: minimal, primer+UMI (`TS`/`RX`), split (`pi`/`sp`), moves (`mv`),
  full (`TS RX pi sp mv po er`), duplex-parent (`dx=-1`).
- Duplex: `duplex-read` (`qs dx mx ch st RG pi po`) and `duplex-minimal`
  (`qs dx mx ch st`). The duplex set omits duration, sample, move, scaling,
  read-number and minknow tags, matching `generate_duplex_read_tags`.
- Barcode: `barcode` emits `BC`, `bv` and the seven-float `bi` array; the
  writer derives the seven floats from typed score fields using the source's
  `front_length`/`rear_length` arithmetic. `barcode-unclassified` proves the
  `unclassified` sentinel emits no barcode tags.
- RNA: `rna-polytail` emits `pt`/`pa`, and `rna-polytail-window` has a trailing
  poly-A run so `find_rna_polya` returns 10 and `qs` must use `qstring[:10]`.
  Its `pt` value is `-1` (`POLY_TAIL_NOT_FOUND`), which the source still emits
  because only `-2` (`POLY_TAIL_NOT_ENABLED`) suppresses the tags.

`qs` is recomputed in Go with `decode.MeanQScore` and the source's window rule,
including the RNA poly-A prefix, so it is a real differential rather than an
echoed input.

`barcode-unclassified-score` is the one **rejection**: a caller that pairs the
`unclassified` sentinel with a barcode score is refused, because the pinned
source would silently drop the score. The writer does not drop data silently.

Two controls require the Go aux to move, so a pass cannot come from comparing a
blob with itself: `split-point-1` increments `sp`, and `rna-window` rescores the
whole `qstring` instead of the poly-A prefix. Both change the aux bytes. The
comparison uses real writer output: the Go side writes a BAM and reads back the
aux region, rather than re-implementing the encoder in the harness. The only
harness-side reimplementation is `findRNAPolyA`, used to compute the `qs` input
window; the native side computes its own, so the comparison still covers it.

## Open

Multi-`@RG`/multi-run headers, `modbase_models` (`MM`/`ML`/`MN`),
`model_stride=-1`, larger read sets, other modes, compressed/topology policy and
independent review remain open. The writer is not yet wired to caller output;
`ReadToBamTypeNode` metadata beyond the current postcall seam is still missing.
