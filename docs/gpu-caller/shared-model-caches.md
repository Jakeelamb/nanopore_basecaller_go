# Persistent GPU model caches

The candidate caller owns one HAC or SUP model for a run. Each profile owns its
queue, stream, checked forward plan and input buffer. Weights and caches upload
once; selecting another chunk size does not construct a second model. This
follows the pinned source's persistent `CudaCaller::m_module` and local forward
working memory. No process-wide cache or mutable current-profile field is used.

`NewHACGPU` and `NewSUPGPU` construct persistent owners without a profile.
Every forward, observation and HAC inactive-slot probe requires the explicit
immutable plan produced by `PlanHACGPU` or `PlanSUPGPU`. Current qualifier callers
pass their declared tiny plans. Model forward methods validate the plan before
allocating activation storage. Supported profile geometry and numerical
operations are unchanged.

SUP builds the full 4096-position RoPE table once. Each forward borrows a prefix
with its selected time dimension while cache observations retain the full table.
Persistent QKV scale bindings live until model close. The first encoder entry
reports cold prepared-before provenance and downloads the uploaded after cache.
Later entries emit `cache.warm` observations downloaded from the persistent
owners; they cannot replay the host cold snapshot. Cancellation before entry
does not consume the first-entry flag; entry commits before callbacks because
the uploaded cache already exists, even if an observer then fails. This does not
qualify native abort/resume behavior. Invocation cleanup detaches only its own
activation bindings. A failed scale detach keeps the entire owner
set so a needed scale cannot become unreachable.

Profile shutdown drains its stream and releases its inputs and result owners.
After every profile has drained, run shutdown closes the shared model before the
device and engine. Failed owners survive for a cleanup retry. If a profile drain
fails, direct shared-cache teardown waits for device or engine recovery. The
original failure still makes the run incomplete even if recovery succeeds.
Partial initialization returns its remaining owner with the error when cleanup
cannot finish. HAC temporary padding, workspace, fused and projection owners
also remain reachable after failed cleanup.

Host controls use a synthetic memory-backed ABI5 loader to check allocation
identity across both profile orders, unchanged full RoPE and prefix descriptors,
scale-binding lifetime, changed-plan rejection and failed-close recovery. They
stop after input transport and do not compute a GPU model. Synthetic cache values
are not native weight or numerical evidence.

The production integration adds checks for distinct HAC invocation workspaces,
one input-stage allocation per call without cache reuploads, independent SUP
model instances, activation/view scale cleanup and top-level constructor-owner
retention. These passed in the capped `t-2b3921/host002` run, together with caller
failure/accounting controls and vet. The initial full suite exposed a CPU13
selection fixture missing the overlay's three required imports; the fixture now
declares them and the production verifier's strict import checks are unchanged.
The full suite also requires pinned-source files beneath `external/dorado-source`;
their absence remains a failure, not a skipped or qualified native endpoint.

Native qualification remains required for both models and both full/short orders
on their distinct streams: compare every declared initialized range, cache and
activation descriptor, graph stage, scores and output extraction against the
pinned source. Check upload counts and actual peak device memory. Neither a host
pass nor this ownership change establishes GPU parity, variable chunks, collision
scheduling, complete-input output, biology or throughput.
