# GPU decoded chunk assembly

Status is host-verified source implementation. Genuine GPU decoded captures and
the decoder units remain unqualified.

`internal/caller/gpu_assembly.go` adds the collector that receives completed GPU
chunks in completion order and restores source order before the accepted
`internal/stitch` unit. It does not decode, infer or mutate accepted stitch
logic.

## Source pins

- `read_pipeline/base/chunk.cpp` and `stitch.cpp` define chunk coordinates and
  the asymmetric overlap cuts.
- `read_pipeline/nodes/BasecallerNode.cpp:420-491` assigns per-read chunk indices
  and stitches only after every chunk completes.
- `internal/stitch` already carries the accepted 726 source cases.

## Contract

`DecodedWork` carries read identity, source chunk index, raw signal offset and
length, padding granularity, sequence, quality and moves. Raw length excludes
repeat padding; granularity is the lane alignment used to derive padded model
length.

`Collector` fixes the expected chunk count per read up front. It rejects an
unknown read, an index outside the expected range, a duplicate chunk, a chunk
outside the signal, and a granularity that is not stride-aligned. `Assemble`
requires every expected chunk, orders by source index, and delegates coverage,
overlap, move and quality validation to `stitch.Stitch`. Completion order cannot
change the result.

## Host evidence

```sh
go test -json -count=1 ./internal/caller -run '^TestGPUAssembly'
go test -count=1 ./...
go vet ./internal/caller
```

The tests cover shuffled completion, independent reads, missing and duplicate
chunks, cross-read identity, bad coordinates and granularity, incomplete
coverage and preserved padding granularity. Receipts are under
`artifacts/gpu-plan/gpu-assembly-host-20261023T185028Z/`.

These six Go unit checks are not captured GPU decoder output and not native
parity. The card's genuine decoded-chunk replay remains open until D01-D04
produce retained decoder captures. No GPU, lock or native compiler was used.
