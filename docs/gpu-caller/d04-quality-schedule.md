# D04 quality scheduling candidate

`internal/gpudecode/quality_schedule.go` moves state-neighbor selection and
move-associated frame grouping into Go. It is source/host work. Posterior
scans, GPU quality arithmetic, native differential comparison and final decoder
output qualification remain open.

The source is the pinned decoder member `930b35afa9e352b412b58e07a43d693630280ed9f422dc324106b2b8e700758f`,
selected SM120 PTX `7e46f59645025bca2cf7a519c9f0334b5435942ca9006d0dec5faa6d3e63b01a`.
Line references below use the retained whole `decoder_lib.sm_120.ptx` in
`artifacts/decoder-static-20261025-001/`, rather than recovered C++ source.

For state `s` and state count `S` (256 or 1024), neighboring posterior states
are visited in order `floor(s/4)+k*S/4`, `(4*s)%S+k`, for `k=0..3`.
The center and previously included neighbors are excluded. PTX lines4735–4944
show the conditional additions for the compact-output branch; lines5192–5405
repeat them for move padding. The neighbor sum starts at positive zero. The
center posterior is added after that sum, rather than before it. Go selects
indices only; it does not replace FTZ addition or approximate arithmetic with
CPU math.

Traceback path frame `i` reads posterior row `i+1`. Before grouping, the source
retains the full state for posterior lookup and writes `state%4` back to path.
Each group ends at the last frame or immediately before the next nonzero move.
Its base is taken from the final frame of that group. Compact output uses the
next emitted index; padded output writes at the group's first frame index.
See lines4700–4735,4958–5018 and5415–5485. `PlanQuality` copies input path and
moves, checks the complete state/move domain, and requires the first move to be
one, as established by traceback. It does not enforce additional biological
consistency between states and moves.

The plan does not claim the source result-word semantics of padded decoding.
In particular, the padded source branch resets the compact emitted counter to
zero. Neither final result words nor padded output tails are synthesized by
this scheduling unit.

Bounded CGO0 host002 passed four named tests and vet, including all1280 HAC/SUP
state neighborhoods, literal order/deduplication cases, move groups, both output
placements, owned-input mutation checks and malformed trace rejection.
The receipt is `artifacts/d04-quality-schedule-host-20261025-002/receipt.json`.
Host001 failed before a Go child launched because its wrapper expected a
nonexistent `go.sum`; it remains preserved. Neither attempt loaded native code
or executed GPU/model work. An independent PTX review and original decoder
stage comparison are still required.
