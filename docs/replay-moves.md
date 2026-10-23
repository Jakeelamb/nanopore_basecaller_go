# Native move parity from retained scores

`verification/replaymoves` verifies native decoder and stitch outputs without
loading model weights or rerunning inference. It closes the retained-input move
evidence gap identified in [the default pipeline trace](default-pipeline-trace.md).
The result qualifies the captured untrimmed, unsplit endpoints for the pinned
float CPU source oracle. It does not qualify stock CUDA12, GPU execution,
post-call splitting or trimming, biological accuracy, or performance.

The retained CRF scores, guides, and posterior probabilities came from the Go
caller's CPU13 LibTorch path. These native source oracles consume those same
arrays. The official stock CUDA12 FASTQ endpoint comparison is separate evidence
for sequence and quality. Those FASTQ records contain no move table, so this
check makes no claim about the official binary's moves.

## Run

From the Go repository root, run the maintained check with the genuine retained
HAC and SUP manifests.

```sh
verification/replaymoves/verify.sh external/dorado-source \
  artifacts/caller/attempt-20261023T023928Z-Wr38Jl/capture/bb76c31a-36f0-48a0-abd2-107a10e1ade0/manifest.json \
  artifacts/caller/attempt-20261023T033758Z-xTMh1T/capture/bb76c31a-36f0-48a0-abd2-107a10e1ade0/manifest.json
```

The script creates an ignored `artifacts/replay-moves/attempt-*` directory. It
retains the exact command, environment, UTC clocks, exit status, raw test logs,
and log hashes. The replay directory retains native build results, executable
and dependency hashes, compiler identity, Go executable and source hashes,
manifest and payload hashes, protocol inputs, native stdout and stderr, Go
outputs, and per-command receipts. Native replay processes have a 120-second
timeout. Large decimal float-bit inputs stream to files and then to the native
process rather than accumulating in memory.

A direct replay-only CLI is also available.

```sh
go run ./verification/replaymoves/cmd -source external/dorado-source \
  -output artifacts/replay-moves/new-output CAPTURE_MANIFEST [CAPTURE_MANIFEST ...]
```

The output directory must not exist. The CLI runs parity checks. The maintained
script also runs rejection controls. Ordinary `go test` skips the genuine-input
checks unless their environment variables are set. A skipped test is not native
parity evidence.

## Contract

`caller.Replay` validates the complete capture schema, payload paths, regular
file types, byte counts and hashes, pinned model configuration, raw and
normalized sample counts, recomputed normalized binary16 bits, fixed chunk
coordinates, and captured Go endpoints. The checker reuses that validation
before and after native replay. A self-consistent capture manifest is not an
authenticator of its producer or historical provenance.

`decodeoracle.Build` verifies Dorado commit
`8b8fc5d36a9c0baab262a743cb175b5878e38ca6`'s pinned `beam_search.cpp` bytes and
extracts its existing source slice. The existing protocol instantiates the float
CPU beam search and quality generator. Each chunk reuses its retained scores,
backward guides, posterior probabilities, and decode options. It compares exact
sequence, quality, and move bytes with the retained chunk and Go `Decode`.
Native and Go states, final score, and quality data also require exact float32
bits. Forward guides are hash-validated by replay but do not enter beam search.

`stitchoracle.Build` verifies its existing source pins and compiles the untouched
`stitch.cpp` translation unit with the existing `ReadCommon` container shim.
The checker passes native decoded chunks through that existing protocol and
compares native stitching with both Go stitching and the retained output.
These captures have 19,846 normalized samples, stride 6, fixed chunk offsets
0 and 7,560, raw chunk sizes 12,288 and 12,286, and granularity 12,288.

There is no positive-result cache. Every invocation rebuilds both oracles and
runs every requested capture. Missing payloads or receipts cannot qualify by
cache reuse. Both source pins and Go verifier source identities are checked
again before a pass is recorded. The script never calls Dorado or an inference
producer. It does not need a reference, GPU, model, or download.

The rejection controls alter copied captures only. They cover changed scores,
short scores, a short score payload with repaired manifest hash and size,
changed normalized bits, path traversal, changed moves, and changed stride.
Separate controls reject mutations of decoder and stitch source before native
compilation. A truncated native decoder input must exit with code 2 and report
`truncated float bits`. Basic tests also reject symlinks, bad payload hashes,
wrong tensor shapes, and reuse of an output directory.

## Retained result

The first complete check passed on 2026-10-23 UTC. Its raw attempt is
`artifacts/replay-moves/attempt-20261023T035505Z-niop51/`. The primary receipt is
`replay/receipt.json`. Raw test output is `stdout.txt`; process clocks and exit
status are in `command.txt`. The short-protocol control retains its own expected
exit-2 receipt under `replay/truncated-native-input/`.

All four chunks have 2,048 blocks and 2,048 move bytes. The HAC chunks have
256 states, and the SUP chunks have 1,024 states. Every row below matched all
move bytes exactly across native source, Go, and retained output. Chunk state,
score, and quality-data comparisons also passed with exact bits.

| Endpoint | Bases | Move bytes | Move SHA256 |
| --- | ---: | ---: | --- |
| HAC chunk 0 | 996 | 2048 | `fda5d6ca17273c2dad7ebdca43aaf9fc0408a6745b144f1a1a169617c38c734b` |
| HAC chunk 1 | 1050 | 2048 | `85265563e30d1e5d2af999f71b96c015c21687353391d111f88895d7131b7f38` |
| HAC stitched | 1653 | 3307 | `568a7ea8b158788b4a1b3393345e25e8b12eaf863273ddb54349f6bdcf32cca3` |
| SUP chunk 0 | 994 | 2048 | `a4a8fb944135dfb4fef1767aa0a0ec2f001e5200878a77971d19a095de4ffd31` |
| SUP chunk 1 | 1054 | 2048 | `d72e51e2ddaddb662636e2347d620a118efc5615b0e16c7c36fe822cfacbc042` |
| SUP stitched | 1650 | 3307 | `3fb4bfdfa128ef931fcec22993245c7137e81015863d16925c2c4c441f0be42e` |

The stitched sequence and quality hashes are also exact across all three paths.

| Model | Sequence SHA256 | Quality SHA256 |
| --- | --- | --- |
| HAC | `e6292c4cc2d67b842cc7f61bcb175a08be9e01eb8189d1fffae9264265cbe453` | `cbaba7569be04c74dcb9223af10688dae468b56d7636622305df64e7cf01ce0b` |
| SUP | `fe3925bb44e22494303c313a401f0a3f6f97e5c8bb93326c0ddf5b15fb35c9ab` | `2ff15b9b720454439608210450d89d9d8bd165d1bef1d657c04f31ea1e6cc17b` |

The receipt retains every native protocol input and output hash and all captured
CRF payload hashes. The common raw-signal hash is
`931078657d3ef96ef9faeacd3fb0dad3197abeff4bf577a9cf8710b2b99bce4b`.
HAC normalized bits hash to
`fd4d291b78f5cfa5a69c77a09a7a4472cabda65d42e9b4b168133c7fbeef5dd0`.
SUP normalized bits hash to
`42c3a60fcaeb15937a66219fabcf398e2b75a31d88e029dfb04efa832a1e69c2`.
The different model standardisation pins explain why the normalized arrays have
the same length but different bits.
