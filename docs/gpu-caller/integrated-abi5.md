# Integrated decoder ABI 5 candidate

This source candidate combines the accepted HAC and SUP graphs with D03 beam
search and D04 posterior and quality computation. Source integration is complete.
Build, host, native, GPU, full caller, and biological acceptance remain pending
for this combined revision.

Go owns the decoder decisions. `GuidePlan` describes the fixed score geometry.
`BeamPlan` selects beam operands, CRC matches, cutoffs, compaction, and history.
`Posteriors` owns alpha initialization, frame order, old-alpha selection, and the
separate destructive score parent. Quality code owns traceback, neighbor order,
groups, output slots, and counters. `internal/tensor` validates typed storage and
marshals requests. Native PTX performs only the accepted numeric operations.
The existing collision rejection and both attention masks are unchanged.

`NT_ABI_VERSION` is 5. This version requires explicit SUP geometry, the beam
program and numeric symbols, and the post numeric symbol. A version 4 library
is incompatible. The loader rejects it before calling the changed SUP signature.
The maintained build command reports ABI 5 without claiming runtime acceptance.
Current fake libraries report 5. Their argument projections are host controls.

The guide module now links guide, forward, posterior, and quality PTX. Its linked
cubin needs a fresh expected digest. The beam module has its own digest and
owner. Both retain module and linker handles through partial initialization or
close failure. The old guide digest and historical backend002/ABI4 receipts do
not qualify this source.

The combined build input list retains SUP and adds all six D03/D04 native files.
The closed controller already pins every `.h`, `.cpp`, and `.inc` under the two
tensor directories. Its backend export check requires guide program and frame,
beam program and numbers, post numbers, and SUP. Its output remains unexecuted.
A new controller, plan, dependency sets, objects, and linked artifact are needed.

The H08 `nativeFixtureDirectory` helper and its load-weight fixture selection
remain. The shared storage fake contains both numeric implementations and SUP.
Its compile uses the D04 receipt helper. The combined host request sets both
D03 and D04 receipt roots to the same attempt directory. Missing-symbol fakes
also include SUP, so they reach the intended decoder symbol rejection.

Historical `verification/tensor/check`, `verification/tensor/cuda`, and
`verification/hacgpu` ABI4 qualification checks remain unchanged. They cannot
qualify this ABI5 candidate. Their future producer and consumer contract needs
separate review. This candidate does not promote retained ABI4 evidence.

The source freeze and exact pending host request are under
`artifacts/integration-source-001/`. The request requires a fresh coordinator
grant. C-only fake-loader tests do not load a genuine tensor backend or run CUDA.
The maintained repository test suite and parity command remain later gates.
