# D03 beam ordering candidate

Status: partial source/host implementation only. No D01 native capture has been
consumed and no GPU beam parity is established.

`internal/gpudecode/beam_order.go` implements two recovered policies for W32:

- `CompactBeam` keeps inclusive-cutoff survivors in the source warp's lane-major
  order, capped at 32. It accepts the already expanded and merged front, with
  five candidates per previous beam.
- `TerminalBeam` returns the earliest finite maximum in compacted order. The
  inspected final-frame partial merge emits only one winner per pair; it does
  not establish a fully sorted or permutation-preserving scratch array.

Comparisons flush float32 subnormals to signed zero as the source FTZ predicate
does. Retained score bits and record fields are copied unchanged. Nonfinite
scores and out-of-scope front capacities are rejected. Wire encoding remains
explicit; a Go struct memory layout is not the source record ABI.

The independent static trace and PTX-block interpreter are retained in
`artifacts/parity-resume-review-20261024/d03-compaction-sort-review.md`.
They correct the initial left/right register interpretation and exercise 48
selection and 32 compaction cases. The Go host checks in
`artifacts/d03-beam-order-host-20261025-001/` pass literal all-tie order,
strict maximum, FTZ equality, record preservation and domain rejection cases.
These are static and host observations, not native numerical evidence.

Initial and two-bit extended CRC32C keys are also implemented and checked
against independent standard-library byte updates. Their source seed and raw
CRC conventions are preserved; this does not qualify collision merging.

`TraceBeamHistory` now translates the successful traceback substep in
`decode_step.ptx:117-260` (original PTX:4542-4685). It starts at terminal slot
zero, walks initialized history rows T through 1, copies full state IDs and tests
the stay byte against zero. It then forces the first move to one. Its path is
the temporary full-state path before quality aggregation replaces entries with
base IDs. Unvisited scratch records and the unused frame-zero predecessor are
not interpreted. Three host tests pass in
`artifacts/d03-traceback-host-20261025-001/`, including a selected predecessor
chain with poisoned unvisited records, a one-frame SUP boundary and rejection of
uninitialized selected slots without returning a partial trace. Native comparison
remains open. Independent partial-unit source/host review passed in
`artifacts/parity-resume-review-20261024/d03-go-policy-review.md`; it also checks
the raw CRC rules against the pinned PTX. This does not close full D03.

Expansion and collision merging, numerical cutoff construction,
guide subtraction, history production and full chunk decisions remain unfinished.
Native all-stay/tie/near-tie/collision/cut controls and genuine HAC/SUP comparisons
remain required before D03 can close. The actual runtime-selected Koi image is
also still unobserved.
