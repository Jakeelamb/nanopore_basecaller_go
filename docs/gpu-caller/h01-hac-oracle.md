# H01: capture every HAC GPU stage once per declared case

Status is **device-captured on the admitted host**. H01 is the single native
producer every downstream HAC card consumes: one verification-only run of the
untouched pinned HAC graph that records every convolution layer, every recurrent
layer, the projection and the scores, plus the complete raw parent arena for each
stage. It is not a port, and it is not end-to-end basecalling or biological
evidence.

## Declared case

| Item | Value |
| --- | --- |
| Model | `dna_r10.4.1_e8.2_400bps_hac@v6.0.0` |
| Graph | CRF + FLSTM, factorised decoder (`scale = 5`, `out_features = 128`) |
| Batch | 64 (the pinned `get_batch_size_granularity` for non-TX models) |
| Chunk | 192 samples |
| Stride | 6 (`conv0` stride 6), so `T_out = 32` |
| Input | 12288 normalised samples from pinned fixture read 1 (`hg002_smoke`) |
| Device | ordinal 0, `GPU-807d7ae7-6a99-c795-12c2-23fc801d38e7`, driver 615.78.08 |

Batch 64 is not a convenience. `CudaCaller::get_batch_size_granularity` returns
64 for non-TX models (`CudaCaller.cpp:211-214`) and `koi.h` documents
`host_factorised_linear` as requiring `N % 64 = 0`. At `N = 1` the kernel returns
status 2 and leaves the output untouched, which is why
`artifacts/tensor/hacgpu-capture-005` is preserved as a failed attempt: it is a
"successful" capture of an all-zero projection.

`T_out` must be a multiple of 32 for the same projection kernel. `T_out = 48`
(chunk 288) never returns; the probe is preserved under
`/tmp/opencode/h01p`. Production `T_out` values are 2048 and 1024, both multiples
of 32, so `T_out = 32` exercises the same tile path. The development fixture has
19856 samples, and `64 x 192 = 12288` is the largest batch-64 geometry that fits
in real signal.

## Pinned source

| Source | Lines used |
| --- | --- |
| `dorado/basecall/model/CRFModel.cpp` | `87-145` literal `run_koi` replay; `51-70` decoder construction |
| `dorado/nn/ConvStack.cpp` | `131-158` reserve/run; `191-215` per-layer reservation; `215-373` `ConvLayer::run_koi` |
| `dorado/nn/FLSTMStack.cpp` | `97-123` reserve/run dispatch; `125-295` `forward_koi` layer loop |
| `dorado/nn/FactorisedLinearLayer.cpp` | `52-106` projection |
| `koi.h` (libkoi 0.7.5) | `261-272` `host_factorised_linear` contract |

Source revision `8b8fc5d36a9c0baab262a743cb175b5878e38ca6`; `-unit sources`
re-pins every file before a capture runs.

## How the graph is driven

`verification/hacgpu/native/oracle.cpp` drives the graph **twice**:

1. **Untouched reference.** `convs->run_koi`, `rnns->run_koi`,
   `linear1->run_koi` are called exactly as `CRFModelImpl::run_koi` calls them.
   The digest of each stack output is recorded as `stack_sha256`.
2. **Per-layer replay.** The same plan is executed one layer at a time:
   `convs->layers[i].run_koi` for each convolution, and a literal transcription
   of `FLSTMStackImpl::forward_koi`'s layer loop
   (`verification/hacgpu/native/oracle.cpp`, `flstm_replay`) for each recurrent
   layer, capturing `wm.current` after every layer.

The replay must reproduce the untouched stacks byte for byte, and it is the
oracle itself that enforces it: after the convolution loop and after the
recurrent loop it compares the live tensor digest against `stack_sha256` and
aborts on any difference. `Import` re-checks the same relation from the recorded
digests, so a capture whose per-layer replay drifted from the untouched graph is
rejected even if its own payloads are self-consistent.

The FLSTM replay reuses the device weight caches that the untouched pass
populates. It never re-quantises them: a second quantisation would be a second,
different graph. `flstm_replay` aborts if the caches are not already populated.

The per-layer replay uses a `#define private public` include shim confined to
`oracle.cpp` to reach `FLSTMStackImpl::layers_` and its device caches. The stock
source is unchanged, no object layout changes, and the shim never leaves this
translation unit.

## Stages

`Import` derives the expected stage list from the declared graph
(`StageNames(convs, lstmLayers, linear2)`) and rejects any missing, extra or
reordered stage. For this case:

`input, conv0, conv1, conv2, flstm0..flstm6, linear1, scores`

| Stage | DType | Layout | Bytes | Meaning |
| --- | --- | --- | --- | --- |
| `input` | Half | NTC | 24576 | normalised `[64, 192, 1]` |
| `conv0` | Half | NTC | 393216 | `[64, 192, 16]` swish |
| `conv1` | Half | NTC | 430080 | `[64, 192, 28]` swish |
| `conv2` | Char | CUTLASS_TNC_I8 | 2424832 | quantised `[37, 64, 1024]`, `T_out + 5` CUTLASS rows |
| `flstm0..6` | Char | CUTLASS_TNC_I8 | 2424832 | evolving in/out buffer, one per layer |
| `linear1` | Half | NTC | 4194304 | factorised projection, `[64, 32, 1024]`, `tanh x 5` |
| `scores` | Half | NTC | 4194304 | `wm.current`; identical to `linear1` in this branch |

`scores` and `linear1` are the same buffer here because the factorised branch
registers one projection and `linear2` is never constructed
(`CRFModel.cpp:51-70`). Clamping to `[-5, 5]` is performed by `CUDADecoder` on
read, not by the model, so the recorded scores are the projection output. The
range is source-derived: the kernel applies `tanh` and scales by `SCALE = 5`.

Every stage records its own payload, the complete raw parent backing dump, the
byte interval inside that backing, the working-memory `N/T/C`, layout, reservation
and current-bytes counters, and the source-declared `is_input_to_rev_lstm` flag.
`Import` requires the payload to be the exact slice of the backing dump at the
declared interval, so the arena and the stage cannot disagree.

### Unwritten rows are prospective exclusions

`WorkingMemory::allocate_backing_tensor` uses `at::empty`, so the arena is not
zeroed. The source explicitly initialises only some of it: `FLSTMStack.cpp:139-147`
zeroes rows `0`, `T+1` and sets row `T+2`, and `temp_bfr.zero_()` zeroes the
scratch buffer. Everything the kernels do not write is **not evidence**. Each
stage therefore declares its unwritten rows along the `T` axis, derived from the
source rather than assumed:

| Rule | Source | Effect |
| --- | --- | --- |
| small-conv padding | `ConvStack.cpp:239-242` allocates `T_out + 2 * output_T_padding` rows and narrows to `[padding, padding + T_out)` before the kernel writes | `conv1` declares `[0,9)` and `[201,9)` |
| CUTLASS conv | `ConvStack.cpp:283-288` allocates `T_out + 5` rows and writes only the `T`-row NTC view, offset `is_input_to_rev_lstm ? 1 : 2` | `conv2` declares `[0,1)` and `[33,4)` |
| FLSTM guard rows | `FLSTMStack.cpp:139-147` initialises rows `0`, `T+1`, `T+2`; the kernel writes the `T`-row window at offset `is_input_to_rev_lstm ? 1 : 2` | alternating `[1,1) [35,2)` and `[35,2)` |

`input`, `linear1` and `scores` declare no exclusions: the copy and the
projection kernel write their whole buffers. `Import` validates every declaration
(inside the `T` axis, no self-overlap), so an exclusion cannot silently grow to
swallow evidence. Consumers receive the mask and must not treat those rows as
compared values.

Separately, the capture is repeated and every recorded byte — every stage payload
*and* every full backing dump, exclusions included — is required to match. That is
`-unit compare`, and it passed for `hacgpu-capture-006`, `-007` and `-008`. So the
excluded rows are reproducible in this environment; they are still not claimed as
evidence, because reproducibility of unwritten storage is not a property the
source guarantees.

## Admitted capture

Host lock held for the whole run; no competing compute process was observed.

| Item | Value |
| --- | --- |
| Build | `artifacts/hacgpu/build-010`, ABI 4, CUDA family 13.0 |
| Oracle SHA-256 | `bcd3afa1c7c0a4e7eadc6f401563928935808b2cd2f25c3eb62517a25f75bc39` |
| Capture | `artifacts/tensor/hacgpu-capture-008`, status `stage_capture_passed` |
| Repeat captures | `hacgpu-capture-006`, `-007`, `-008` are identical in every recorded byte |
| Stages | 13 |
| Reservation | 7782400 bytes |
| Scores SHA-256 | `56b26176b4272906612eb75690087182a72527f94a9171d9937e5dd80b0d12db` |
| Scores shape | `[64, 32, 1024]`, 22362 distinct half values in `[-5, 5]` |
| Input SHA-256 | `b76f42a90771d88d893f653e2bcefe5243c73abb3a1125a37e7d2d9efe3e4428` |
| Normalisation | mean 94, stdev 24, calibration scale `0.1870698481798172`, offset `-270`, trim 10 |
| Input pins | 94 (build, model weights, fixture, sources, produced bytes) |
| Runtime pins | 23 mapped objects, re-hashed outside the child |
| Commands | 5 recorded with clocks and exits; 0 failing |

The oracle reports the shared objects it actually mapped
(`runtime-paths.json`); the orchestrator hashes those exact files and requires
`libtorch_cpu`, `libc10`, `libcudart`, `libcublas` and the host `libcuda` driver,
rejecting any CUDA-12 or foreign Torch path.

## Rejection controls

`verification/hacgpu/h01-controls.sh` copies an admitted capture, applies one
mutation, and requires `-unit verify` to reject it. All eleven fire with a
self-describing reason:

| Control | Observed rejection |
| --- | --- |
| `corrupt-payload` | payload digest does not match the recorded digest |
| `missing-payload` | stage payload file is absent |
| `extra-stage` | capture has 14 stages, want 13 |
| `stack-digest-drift` | per-layer `conv2` does not reproduce the untouched `convs` stack |
| `scores-digest-drift` | scores stage does not match the recorded scores digest |
| `batch-not-granular` | batch 63 is not a multiple of the 64-wide kernel granularity |
| `layer-relabelled` | stage `conv2` has layer index 9 |
| `t-out-not-tiled` | stride 6, T_out 48 and chunk 192 are not a valid projection geometry |
| `unwritten-drift` | stage `flstm0` declares unwritten rows [30,50) outside the 37-row T axis |
| `graph-shape-drift` | capture declares `linear2` with `linear1` kind `factorised` |
| `foreign-source` | capture source commit is not the pinned revision |

Go-level controls, run by `go test ./verification/hacgpu/`:

- `TestImportRejectsCorruptStage` — flipped byte, missing stage, truncated backing.
- `TestImportRejectsForeignSchema` — commit drift.
- `TestImportRejectsGraphShapeDrift` — declared decoder shape drift.
- `TestImportRejectsStackCrossCheckDrift` — drifted or absent stack digest.
- `TestImportRejectsConstantScores` — the silent "graph did not run" shape.
- `TestImportRejectsUnwrittenDrift` — an exclusion that escapes the `T` axis or
  overlaps itself.
- `TestStageNamesFollowsTheDeclaredGraph` — the stage list is derived, not assumed.
- `TestCompareRejectsDrift` — the determinism gate compares real bytes.
- `TestFixtureNormalisationDecodesBitPatterns` — `float_bits` are uint32 bit
  patterns, not JSON numbers.
- `TestNormaliseRejectsNonFiniteOutput` — a mis-decoded calibration cannot be
  captured as an all-infinity graph run.

## Preserved failed attempts

| Attempt | Failure |
| --- | --- |
| `hacgpu-capture-001` | assumed a fixed 6-stage list including `linear2`; HAC v6 is factorised and has no `linear2` |
| `hacgpu-capture-002` | assumed every stage is `Half`; the conv/FLSTM stages are `Char` |
| `hacgpu-capture-003` | decoded fixture `float_bits` as JSON numbers, giving a ~1e9 calibration scale and an all-`+inf` input; 5 of 8 `rnns` values distinct |
| `hacgpu-capture-004` | required exact mapped sonames (`libcublas.so`), which never match versioned names |
| `hacgpu-capture-005` | batch 1; `host_factorised_linear` returned 2 and `linear1` was 2097152 copies of `0.0` |
| chunk 288 probe | `T_out = 48` hangs in `linear1`; the run never returns |

## Reproduce

```sh
go run ./verification/hacgpu/cmd -unit sources -dorado-source external/dorado-source
go run ./verification/hacgpu/cmd -unit build \
  -build-dir /home/jake/Projects/nanopore_basecaller/software/source-build-qualification-20261023/build-gcc15 \
  -dorado-source external/dorado-source -out artifacts/hacgpu/build-010
go run ./verification/hacgpu/cmd -unit capture \
  -build artifacts/hacgpu/build-010/manifest.json \
  -dorado-source external/dorado-source \
  -model /home/jake/Projects/nanopore_basecaller/models/dna_r10.4.1_e8.2_400bps_hac@v6.0.0 \
  -fixture artifacts/pod5/fixtures-v3/manifest.json -read 1 \
  -signal artifacts/pod5/fixtures-v3/hg002_smoke.signal.i16le \
  -batch 64 -chunk 192 -ordinal 0 -runtime-admitted \
  -out artifacts/tensor/hacgpu-capture-009
go run ./verification/hacgpu/cmd -unit compare \
  -out artifacts/tensor/hacgpu-capture-008/stages \
  -compare artifacts/tensor/hacgpu-capture-009/stages
verification/hacgpu/h01-controls.sh artifacts/tensor/hacgpu-capture-008/stages
```

## What this does not prove

- It is not a Go port. All arithmetic is the stock library's.
- It is not end-to-end basecalling, read accuracy, or biological evidence.
- It covers one declared case (batch 64, chunk 192). Other `T_out` values and
  VCS/fused dispatch conditions need their own declared cases once I01 fixes the
  variable-chunk geometry.
