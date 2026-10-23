# CPU HAC and SUP caller

`internal/caller` composes the Go-owned POD5, scaling, chunking, HAC/SUP models,
CRF scan, beam decoder, and stitching units. `cmd/basecaller` emits FASTQ with
the pinned stock CPU metadata tags.
The native boundary provides CPU tensor operations and POD5 decompression.
Production does not invoke Dorado, Python, or a basecalling subprocess.

The admitted models are the exact pinned
`dna_r10.4.1_e8.2_400bps_hac@v6.0.0` configuration with 50 weights and
`dna_r10.4.1_e8.2_400bps_sup@v5.2.0` with 139 consumed weights.
GPU execution, adapter trimming, read splitting, BAM output, and automatic
chemistry/model selection are unsupported. The caller rejects unsupported
model or chemistry before inference. Read chemistry must match the source's
DNA R10.4.1 e8.2 400bps 5000Hz kit and flow-cell combinations.

## API and lifetime

```go
c, err := caller.Open(ctx, caller.Config{
    TensorLibrary: tensorLibrary,
    ModelDir: modelDirectory,
    CaptureDir: newCaptureDirectory,
})
if err != nil {
    return err
}
defer c.Close()
called, err := c.Call(ctx, read)
```

`Config.ChunkSize == 0` selects all model chunk defaults and requires
`Overlap == 0`. A nonzero chunk size honors overlap literally, including zero.
HAC chunk sizes must be multiples of stride 6 from 6 through 12,288, with
overlap a smaller multiple of 6. SUP chunk sizes must be multiples of 768 from
768 through 12,288, with overlap a smaller multiple of its inner stride 12.
The CLI defaults to explicit 12,288 and 600 values. These bounds limit the CPU working
set. The backend uses its qualified one-thread initialization.

`Caller` owns its engine and model. Calls and close serialize on one mutex.
Close releases the model before the engine and is idempotent. Inputs are
borrowed and remain unchanged. Outputs own their sequence, quality, and moves.
`CalledRead.FASTQ` owns the metadata for `CalledRead.WriteFASTQ`. It carries the
channel, read and protocol clocks, protocol run ID, model name, flow-cell ID,
and mean-quality start position. The formatter rejects malformed records and
tag text before writing any bytes.
Each inference closes all intermediate tensor handles on success, failure, or
cancellation. Context checks occur between stages and within the model and CRF
loops. A running native operation and beam decode are not interrupted.

The pipeline preserves Dorado's float32 pore correction and binary16 scaling.
Expected pore levels are 199.21 pA for PRO114 variants, 197.61 pA for MIN114
variants, and 200 pA for FLG114 variants. Case conversion matches
`models/kits.cpp`. A NaN open-pore level or an unknown expected level gives
zero correction. Unknown chemistry still fails caller admission.

Standardized DNA scaling removes a ten-sample prefix only when the signal is
longer than ten samples. This scaler prefix operation still occurs with
`--no-trim`, which disables adapter trimming. Fixed chunks repeat-pad binary16
bits to the configured size, then expand them exactly to CPU float32 input
`[1,1,T]`. The Go HAC graph returns NTC scores. The caller transposes and copies
them to contiguous TNC for Go-owned CRF scans. Beam settings are width 32,
cut 100, blank score 2, and the model's quality bias and scale. Stitching uses
normalized signal length and stride 6.

The source is Dorado revision `8b8fc5d36a9c0baab262a743cb175b5878e38ca6`.
The pore correction comes from `dorado/read_pipeline/nodes/ScalerNode.cpp`
lines 117-140 and 189-262. Chemistry admission comes from
`dorado/models/kits.cpp` lines 197-265 and the 5000Hz chemistry entry.

FASTQ follows the admitted unbarcoded, unsplit DNA CPU path. The selected tags
retain BAM insertion order. They are `qs`, `ch`, `st`, optional `RG`, optional
`PU`, `DT`, and `tm`. Values use tab separators. CPU omits the GPU `DS` tag.
`RG` uses the protocol run ID followed by an underscore and the model name.
An empty model name uses `unknown`, and an empty run ID omits `RG`.
An empty flow-cell ID omits `PU`. Disabled adapter trimming emits `tm:Z:none`.

`DataLoader.cpp` derives read time from acquisition Unix milliseconds plus
`start_sample * 1000 / sample_rate` with integer truncation. `time_utils.cpp`
formats read and protocol timestamps as UTC with six fractional digits and
`+00:00`. Go rejects times outside the pinned native nanosecond clock range.
`ReadCommon::calculate_mean_qscore` selects the quality suffix at model
`MeanQStart`, which is 60 for HAC. Reads of at most that length use all qualities.
`decode.MeanQScore` ports the float32 error lookup, accumulation, division,
`log10f`, and clamp to 1 through 50 from `utils/sequence_utils.cpp`.
The FASTQ float field uses six significant digits, matching HTSlib formatting.
The helper shares the decoder's native `powf` and `log10f` boundary and fails
with `decode.ErrCGORequired` when cgo is unavailable.

The source chain is `ReadCommon::generate_read_tags`,
`ReadToBamTypeNode`, `WriterNode`, `HtsFileWriter::prepare_item`,
`StreamHtsFileWriter`, and `hts_utils/hts_file.cpp`. The structured writer uses
the same preparation method. The older `HtsWriterNode` also appends flow-cell
and protocol clocks. `verification/caller/oracle.go` pins all these sources.

## Run one selected read

```sh
go build -o bin/basecaller ./cmd/basecaller

bin/basecaller \
  --model /home/jake/Projects/nanopore_basecaller/models/dna_r10.4.1_e8.2_400bps_hac@v6.0.0 \
  --pod5 /home/jake/Projects/nanopore_basecaller/data/raw/HG002/PAW70337/PAW70337_66b2eea5_de8117b1_33.pod5 \
  --read-id bb76c31a-36f0-48a0-abd2-107a10e1ade0 \
  --tensor-library artifacts/tensor/attempt-005-sup-abi3/libnanopore_tensor.so \
  --pod5-library external/backend/pod5/lib/libpod5_format.so \
  --device cpu --chunksize 12288 --overlap 600 \
  --no-trim --disable-read-splitting \
  --capture artifacts/caller/new-capture > called.fastq
```

The capture directory must be new and its parent must exist. Repeat `--read-id`
for multiple distinct selections. The CLI requires either selected UUIDs or
explicit `--all`. It rejects an implicit whole-file run and duplicate UUIDs.
Both `--no-trim` and `--disable-read-splitting` are required. FASTQ is the only
stdout content. Errors, capture paths, and the count of completed reads go to
stderr. An error returns a nonzero exit status and retains partial captures.

## Capture and replay

A capture records model config bytes, the interpreted model spec, all consumed weight
pins, the tensor-library hash, chunk settings, and beam options. Each read has
its original raw int16 samples, normalized binary16 samples, metadata, and exact
float32 calibration and pore-level bits. Each finished chunk has contiguous
float32 scores, forward guides, backward guides, posteriors, and decoded sequence,
quality, and moves. Every referenced payload has a byte count and SHA256.
A final manifest exists only after the read completes. Partial attempts retain
finished chunk records without a success manifest. Files are created exclusively
and never overwritten.

Replay requires neither model weights nor the tensor or POD5 libraries.
It validates payload hashes, rejects paths outside the capture directory, checks
tensor shapes and coordinates, and validates the exact captured model config.
It repeats raw normalization, beam decoding, and stitching. Chunk and final
sequence, quality, and move bytes must match the capture. Referenced payloads
are integrity checked; this format does not authenticate an untrusted manifest.
New captures store `CalledRead.FASTQ` and verify it against derived metadata.
Older schema-1 captures derive this field from their pinned input and model
metadata. Replay preserves every retained capture file and its hash.

```sh
bin/basecaller replay --manifest artifacts/caller/new-capture/bb76c31a-36f0-48a0-abd2-107a10e1ade0/manifest.json \
  > replay.fastq

# The equivalent flag form is also accepted.
bin/basecaller --replay artifacts/caller/new-capture/bb76c31a-36f0-48a0-abd2-107a10e1ade0/manifest.json
```

## Observed bounded endpoint

The predeclared plan is `provenance/cpu-caller-control-plan.json`. The parent ran
one unchanged stock CPU producer and retained it at
`artifacts/native-cpu-control/stock-hac-dev1-001/`. The Go producer ran once at
`artifacts/caller/attempt-20261023T023928Z-Wr38Jl/`. Both exited zero and emitted
exactly one record for development UUID
`bb76c31a-36f0-48a0-abd2-107a10e1ade0`. HG005 was unused.

The input has 19,856 samples. Scaling retains 19,846 after its prefix trim.
The fixed chunk starts are 0 and 7,560. Each chunk has 2,048 score blocks and
256 CRF states. The stitched read has 1,653 bases. The Go output matches the
retained stock CPU UUID, sequence, and quality bytes exactly. Replaying the
capture reproduces the Go sequence, quality, and moves exactly.

| Endpoint | Observed SHA256 |
| --- | --- |
| Sequence bytes | `e6292c4cc2d67b842cc7f61bcb175a08be9e01eb8189d1fffae9264265cbe453` |
| Quality bytes | `cbaba7569be04c74dcb9223af10688dae468b56d7636622305df64e7cf01ce0b` |
| Captured Go moves | `568a7ea8b158788b4a1b3393345e25e8b12eaf863273ddb54349f6bdcf32cca3` |

The original producer's raw FASTQ headers differed. Stock included description
tags, and that Go revision emitted only `@UUID`. The original attempt and its
canonical comparison remain retained. The formatter now derives those tags
from the same capture. The CLI replay matches the entire unchanged stock FASTQ
file, including the header, separators, plus line, and final newline.
The shared raw FASTQ SHA256 is
`53ec2ca57acff8ce0041576a6bade2724a623110421867827032990d8df860cd`.
Stock FASTQ does not include moves, so this comparison makes no native
move-table claim. The comparator reports raw byte identity as its endpoint
and retains canonical UUID, sequence, and quality comparisons separately.
This one development read does not qualify GPU parity, biological accuracy,
all-read processing, performance, trimming, or splitting.

The separate SUP plan is `provenance/sup-cpu-caller-control-plan.json`. Its one
unchanged stock CPU producer is retained at
`artifacts/native-cpu-control/stock-sup-dev1-001/`. The Go producer completed
once at `artifacts/caller/attempt-20261023T033758Z-xTMh1T/`, selecting the same
UUID, chunk size, overlap, and disabled trimming/splitting. Its raw FASTQ already
matches stock. The SUP result has 1,650 bases and the same two fixed starts and
2,048 blocks per chunk, with 1,024 CRF states.

The subsequent comparator initially rejected replay because validation still
required HAC's 256 states. The retained red check is
`artifacts/caller/sup-replay-state-regression-red/`. Replay now derives its
state count from the pinned model output channels. No inference was repeated.
The maintained consumer acceptance passed at
`artifacts/caller/check-20261023T034106Z-N6svDo/`, including raw `cmp`, race,
negative header mutation, native metadata/decoder checks, and traced CLI replay.

| SUP endpoint | Observed SHA256 |
| --- | --- |
| Raw stock and Go FASTQ | `dbb28c99d9c29d9808bc27c56f96a1057cfdf0b5fe197ac940e898d380b994c0` |
| Sequence bytes | `fe3925bb44e22494303c313a401f0a3f6f97e5c8bb93326c0ddf5b15fb35c9ab` |
| Quality bytes | `2ff15b9b720454439608210450d89d9d8bd165d1bef1d657c04f31ea1e6cc17b` |
| Captured Go moves | `3fb4bfdfa128ef931fcec22993245c7137e81015863d16925c2c4c441f0be42e` |

SUP also uses fixed chunks on the CPU. `ModelRunnerBase::variable_chunk_sizes`
defaults to false; `BasecallerNode` then selects `generate_chunks`. Its config
normalizes chunk size by inner stride times 64 (768) and overlap by inner stride
(12). The Go caller rejects misaligned settings instead of silently changing
the requested values. Explicit zero overlap remains a Go experiment extension;
stock CLI normalization raises it to one inner stride. No native endpoint
parity is claimed for zero overlap. The two observed controls use overlap 600.

Mux-change tail trimming runs after stitching even with `--no-trim`, matching
the source node. The two POD5 end-reason enum values 1 and 2 trigger it; the
data-service unblock value 3 does not. The [source-linked trim unit](readtrim.md)
and caller wiring tests cover positive synthetic tails and no-op conditions.
`NumSamples` retains the original POD5 sample count, `Trimmed` counts the leading
scaler prefix, and `TailTrimmedSamples` counts removed normalized tail samples.
Replay applies the same transform before checking the captured output. Older
non-mux captures remain unchanged. Both genuine inputs end with `signal_positive`,
so no positive biological mux-change endpoint is claimed.

## Maintained verification

`verification/caller/run.sh` creates a new attempt, builds the Go CLI, runs the
bounded development read, captures intermediates, and compares with the existing
stock output. `MODEL_DIR` and `STOCK_CONTROL` select the corresponding SUP model
and retained control. It never runs the stock producer. Reuse retained captures for
consumer changes instead of rerunning inference.

```sh
CALLER_CAPTURE_MANIFEST="$PWD/artifacts/caller/attempt-20261023T023928Z-Wr38Jl/capture/bb76c31a-36f0-48a0-abd2-107a10e1ade0/manifest.json" \
CALLER_STOCK_FASTQ="$PWD/artifacts/native-cpu-control/stock-hac-dev1-001/stdout" \
  verification/caller/verify.sh
```

The verification script records exact commands, exit codes, UTC clocks, source
hashes, and logs in a unique ignored directory. It runs race checks, native
metadata and decoder oracles, the retained native endpoint replay, explicit
no-cgo tests, vet, and the CLI replay comparator. It also runs `cmp` against
the stock FASTQ and rejects an output with only the channel tag changed.
When installed, `strace` records the built CLI's `execve` and `openat` calls.
The traced output must also pass `cmp`. Before and after hashes verify the
stock output and retained capture manifest remain unchanged.
It rejects missing cached acceptance inputs. Normal `go test` explicitly skips
the retained native endpoint when those two environment variables are absent.
Unit controls cover unsupported chemistry/model shape, default and explicit
zero-overlap settings, cancellation, close, immutable files, malformed manifests,
changed payload hashes, invalid shape, missing chunks, and CLI selection guards.
No verification command starts GPU work or downloads data.

The metadata oracle extracts complete source functions for mean quality, read
groups, quality substring selection, BAM tag construction, writer metadata,
and trim flags. It compiles the original `time_utils.cpp` and uses the pinned
FASTQ tag-selection list. Only data containers are shimmed. System HTSlib
1.24 performs oracle serialization, so this oracle alone does not establish
the stock binary's linked HTSlib identity. The retained stock raw comparison
provides the separate byte-level check for the development read.
The oracle covers 153 records with short-read fallback, the 60-base boundary,
mixed qualities, clamp endpoints, empty optional fields, leap-day timestamps,
fractional-sample truncation, and the native clock limits. It compares raw
FASTQ and mean-quality float32 bits. Source mutation, malformed record,
timestamp overflow, legacy capture, and changed captured metadata controls
must reject invalid input.

The first metadata-oracle attempt at
`artifacts/caller/check-20261023T030448Z-RoCnai/` failed at an out-of-range
year-9999 timestamp. The native intermediate clock overflowed. That failure
is retained. The formatter now rejects that unsupported range, and the
supported clock-boundary cases pass. The subsequent raw FASTQ check at
`artifacts/caller/check-20261023T030557Z-QPqcKp/` passed, including the built
CLI, changed-channel rejection, unchanged capture pins, and replay trace.
The final check at `artifacts/caller/check-20261023T030726Z-Eki21a/` also
records exact float32 mean-quality bits for all 153 oracle cases and retained
decoder oracle receipts. Every positive command exited zero. The changed
channel comparator exited one as required, while its canonical endpoint
remained exact. No model inference or stock producer was repeated.

The unit has one owner across caller, CLI, verification, and this document.
Model, tensor, CRF, and POD5 units have separate owners. Each instance owns its
engine, model, and capture path. Only native tensor registration remains global.
The parent owns independent review and integration.
