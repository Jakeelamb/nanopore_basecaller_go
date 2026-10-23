# GPU postcall reuse and replay

Status is host-verified source implementation. No genuine GPU capture exists yet,
so producer-free replay has run only on synthetic captures. This is not native
parity and not biological evidence.

`internal/caller/gpu_metadata.go` feeds a decoded GPU read into the accepted
default postcall pipeline. `internal/caller/gpu_capture.go` defines the versioned
replay envelope. `verification/gpucaller/postcall.go` replays a capture and writes
a receipt without a model, POD5 file or GPU.

## Source pins

- `read_pipeline/base/messages.cpp:108-220` fixes the read tags, mean-quality
  offset and read-group construction reused here.
- `read_pipeline/nodes/ReadToBamTypeNode.cpp` fixes the final record contract that
  the FASTQ writer and the future BAM writer share.
- `internal/caller/{caller,process,mux,fastq}.go` already carry the accepted CPU
  postcall seam; this unit reuses it instead of forking policy.
- `internal/postcall/postcall.go` owns trim, split, filter and publication order.

## Contract

`GPURead` carries the POD5 metadata, the normalized half-precision signal, the
model stride and the decoded stitch result. `gpuStage` rebuilds the same
`coreStage` the CPU caller reaches after inference: MUX trimming, FASTQ clock and
model metadata, mean-quality offset and sample accounting. `ProcessGPURead` then
runs the accepted postcall policy. No inference, no model load and no policy fork.

`GPUCapture` (`gpu-capture/1`) pins the runtime producer, device, compiler, tensor
ABI and source commit, the lane and chunk geometry, the postcall metadata, the
normalized signal, every decoded chunk and the requested policy. `Seal` writes a
SHA256 over the canonical payload; `DecodeGPUCapture` rejects unknown fields,
trailing JSON, a missing or wrong digest, a CPU producer, incomplete runtime pins,
an unadmitted kit or flow cell, geometry that does not match the pinned lane model,
and chunk coordinates, granularity, sequence, quality or move mismatches.

Replay resolves the exact admitted spec with `model.Pinned` and requires the
requested policy to equal the captured policy, so a capture cannot be silently
reinterpreted. `Assemble` delegates coverage and overlap validation to the
accepted `internal/stitch` unit.

## Commands

```sh
go run ./verification/gpucaller/cmd -unit postcall-check -mode replay -input CAPTURE -out NEW_ATTEMPT
```

The replay writes `reads.fastq` and, last, `postcall-receipt.json` with the capture
digest, model config pin, policy, child/read counts, FASTQ digest and a
determinism result. The receipt is synced and linked after the payload, so a
partial attempt never looks complete.

## Host evidence

```sh
go test -json -count=1 ./internal/caller ./internal/model ./verification/gpucaller -run 'TestGPU|TestProcessGPU|TestPinned|TestPostcallReplay'
go test -race -count=1 ./internal/caller ./verification/gpucaller -run 'TestGPU|TestProcessGPU|TestPostcallReplay'
go test -count=1 ./...
```

The tests cover the four trim/split combinations, MUX trimming, positive split
child identity and clocks, metadata rejection, capture sealing, digest
determinism, policy swap, CPU-as-GPU rejection, tampering, unknown and trailing
JSON, chunk geometry mutants and fresh-output rejection. Receipts are under
`artifacts/gpu-plan/gpu-postcall-host-20261023T190731Z/`; 22 tests pass with no
failures.

## Open

The card still needs a genuine GPU capture, a replay against the stock default
headers and records in each mode, the DS-tag predicate, child-clock and leading
trim controls on real captures, and independent review. No GPU, host lock, model
file or native compiler was used for this checkpoint.
