# Fixed physical capacity and real publication

This caller source keeps HAC at N64 and SUP at N32, with chunk12288 or
chunk6144. Every graph, guide, posterior, beam and quality call computes the
physical profile. `Batch.Real` and its placements identify the real prefix to
publish. No decoded work is constructed for absent rows.

`gpuInvocation.prepareInput`, used by `Call`, allocates fresh zero-filled bytes
for the entire physical input on each call. It RepeatPads each occupied row and
uploads the complete allocation. HAC receives `PaddedSlots` as `ActiveSlots`;
SUP already consumes physical input. `decodedWork`, also used by `Call`, requires
exactly physical N quality rows and extracts only the first Real through their
placements. Positive Real bounds, physical geometry, duplicate IDs, placement
coordinates, pending identity/profile and source slice bounds are checked before
storage access. Real move extents and live sequence/Q bytes are checked before
publication.

Cancellation checks, result-owner retention on failure, owner closure before
publication, stream-first drain, scheduler sticky failures and EOF accounting
retain their existing paths. Graph and decoder errors in any physical row remain
errors. In particular, an inactive beam collision can fail a partial call.
Skipping absent quality material during extraction does not mask a computation
error.

## Release boundary

`gpuBundleRecord.partialProfiles` is a separate per-profile approval set. Nil,
missing and false entries hold partial calls even if `profiles` approves the full
profile. Every true partial entry must have matching true full approval and fixed
HAC N64 or SUP N32 full/short geometry. Malformed records are rejected at bundle
load and invocation construction; `Call` checks membership before native work.
There is no runtime override. `approvedGPUBundles` remains empty.

Private synthetic host records only test this boundary. They grant no production
qualification.

## Native source authority

The pinned source is Dorado `8b8fc5d36a9c0baab262a743cb175b5878e38ca6`.
Paths below are relative to its `dorado/` directory:

- `basecall/CudaModelRunner.cpp:13-60` retains physical input and overwrites only
  occupied fixed rows; preceding inactive values can survive.
- `basecall/decode/CUDADecoder.cpp:20-28,171-191` obtains fixed N from physical
  scores and materializes all N decoded rows.
- `read_pipeline/nodes/BasecallerNode.cpp:189-214` copies and publishes only real
  chunks, while accounting physical padded samples.

Go zeroing is a deterministic input policy. Native source does not establish its
numerical independence from occupied rows.

## Verification

The host tests cover both architectures, both chunk sizes, every Real from 1 to
N, physical quality cardinality, invalid/duplicate placements, distinct invalid
absent decoded material, independent partial approval, fresh input across
full -> 1 -> N-1 -> full, and host publication/failure accounting. They do not run
the GPU. On 2026-10-29, the focused controls and full CGO1 `go test ./...`
passed: 852 root tests and 1,970 subcases passed, with 87 unselected opt-ins
skipped. Maintained parity passed its CPU-package/native-algorithm scope.
The original source reproduces duplicate membership reaching the storage
boundary; the corrected regression rejects it before storage. Both independent
same-model source reviewers reported no findings.

Genuine fixtures must cover full/short physical profiles, prepared half inputs,
weights/config/runtime pins, real membership, preceding calls, zero and finite
poison/stale absent rows, occupied-slot permutations and complete real scores,
guide/posterior, beam, quality/moves/sequence and emitted parent records. Active
corruptions and altered placement/pin controls must fail. Nonfinite inactive
input must retain the whole-input finite rejection. Legal inactive collision
failures must remain visible until independently qualified decoder behavior can
support them.

Tiny SUP chunk768/T64 evidence with 191 events is not full/short partial proof.
Packed HAC/SUP/decoder, collision scheduling, complete caller/cohort evidence,
biology, ALT retention and timing remain separate unfinished work. HG005 is
excluded from selection. Host tests confer no native or biological acceptance.
