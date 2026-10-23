# CPU HAC caller

`internal/caller` composes the Go-owned POD5, scaling, chunking, HAC model,
CRF scan, beam decoder, and stitching units. `cmd/basecaller` emits basic FASTQ.
The native boundary provides CPU tensor operations and POD5 decompression.
Production does not invoke Dorado, Python, or a basecalling subprocess.

The admitted model is the exact pinned
`dna_r10.4.1_e8.2_400bps_hac@v6.0.0` configuration and its 50 pinned weights.
SUP, GPU execution, adapter trimming, read splitting, BAM output, and automatic
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
Chunk sizes must be multiples of stride 6 in the inclusive range 6 through
12,288. Overlap must be a smaller multiple of 6. The CLI defaults to explicit
12,288 and 600 values. This initial bound limits the 1,024-channel CPU working
set. The backend uses its qualified one-thread initialization.

`Caller` owns its engine and model. Calls and close serialize on one mutex.
Close releases the model before the engine and is idempotent. Inputs are
borrowed and remain unchanged. Outputs own their sequence, quality, and moves.
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

## Run one selected read

```sh
go build -o bin/basecaller ./cmd/basecaller

bin/basecaller \
  --model /home/jake/Projects/nanopore_basecaller/models/dna_r10.4.1_e8.2_400bps_hac@v6.0.0 \
  --pod5 /home/jake/Projects/nanopore_basecaller/data/raw/HG002/PAW70337/PAW70337_66b2eea5_de8117b1_33.pod5 \
  --read-id bb76c31a-36f0-48a0-abd2-107a10e1ade0 \
  --tensor-library artifacts/tensor/attempt-004/libnanopore_tensor.so \
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

A capture records model config bytes, the interpreted model spec, all 50 weight
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

The raw FASTQ headers differ. Stock includes description tags; Go emits only
`@UUID`. This is separate from the predeclared primary endpoint. Stock FASTQ
does not include moves, so this comparison makes no native move-table claim.
The compare report retains both raw headers and each canonical byte hash.
This one development read does not qualify GPU parity, biological accuracy,
all-read processing, performance, SUP, trimming, or splitting.

## Maintained verification

`verification/caller/run.sh` creates a new attempt, builds the Go CLI, runs the
bounded development read, captures intermediates, and compares with the existing
stock output. It never runs the stock producer. Reuse retained captures for
consumer changes instead of rerunning inference.

```sh
CALLER_CAPTURE_MANIFEST="$PWD/artifacts/caller/attempt-20261023T023928Z-Wr38Jl/capture/bb76c31a-36f0-48a0-abd2-107a10e1ade0/manifest.json" \
CALLER_STOCK_FASTQ="$PWD/artifacts/native-cpu-control/stock-hac-dev1-001/stdout" \
  verification/caller/verify.sh
```

The verification script records exact commands, exit codes, UTC clocks, source
hashes, and logs in a unique ignored directory. It runs race checks, the retained
native endpoint replay, explicit no-cgo tests, vet, and the CLI replay comparator.
It rejects missing cached acceptance inputs. Normal `go test` explicitly skips
the retained native endpoint when those two environment variables are absent.
Unit controls cover unsupported chemistry/model shape, default and explicit
zero-overlap settings, cancellation, close, immutable files, malformed manifests,
changed payload hashes, invalid shape, missing chunks, and CLI selection guards.
No verification command starts GPU work or downloads data.

The unit has one owner across caller, CLI, verification, and this document.
Model, tensor, CRF, and POD5 units have separate owners. Each instance owns its
engine, model, and capture path. Only native tensor registration remains global.
The parent owns independent review and integration.
