# Finish the Go GPU caller

Use the [feature cards](CARDS.md) to assign one bounded change to one owner. The
machine-readable source is [cards.json](cards.json). This is a work plan, not a
runtime qualification or an instruction to launch its native commands.

## Define completion before dispatch

The initial target is the pinned Dorado simplex DNA pipeline for HAC v6.0.0 and
SUP v5.2.0 on the qualified RTX 5070 Laptop GPU. It includes fixed chunks and the
default variable-chunk path, selected reads and complete declared POD5 inputs,
default SQK-LSK114 processing, FASTQ, and unaligned BAM records and tags. Freeze
device capability, precision dispatch, batch policy, quality parameters, and
supported CLI options before capture. A required path cannot become optional
because its implementation is difficult.

Go owns model recognition, graph loops, weight preparation, workspace planning,
chunk scheduling, decoder decisions, read assembly, and output policy. Native
POD5 decompression and individually named tensor or fused model kernels remain
explicit dependencies. A whole-model, whole-caller, or opaque whole-decoder bridge
does not close this target. The four opaque Koi decode entrypoints are reference
oracles until their contracts and replacement decisions are established.

Completion requires reviewed code and all declared runtime, input-accounting,
output, biological, and performance gates. Exact stage comparison comes first.
Record canonical equality, header equality, raw FASTQ/BAM equality, and accuracy
artifact equality separately. A tolerance, reordered artifact, or canonical
comparison cannot silently replace a literal-byte gate. Preserve failed gates
and report an exactness blocker when it remains. Speed has no fabricated target
or result. Predeclare its acceptance rule before measuring it.

Retain complete captured allocations, but compare source-declared live outputs,
initialized regions, sentinels, and guards. Source allocations can contain
uninitialized bytes. Declare excluded ranges from source before execution.
Never derive a comparison mask from observed mismatches.

Other models, RNA, modified bases, duplex, barcoding, aligned output, other kits,
multi-GPU execution, Volta, and unselected precision/device families are outside
this initial completion claim. Reject unsupported requests explicitly. They need
separate cards before expanding the claim to all Dorado functionality.

## Preserve existing work

The starting public head is `5cb228cdf97dfb397fd58b5917936ea2c538229f` on
`port/go-caller`. The reviewed SUP int8 capture unit is frozen and uncommitted in
`verification/koiquant/**` and `docs/koi-qkv-int8.md`. It has source-only evidence.
Its C++ adapter remains uncompiled and its native arithmetic remains uncaptured.
Do not overwrite it when creating worker branches.

Retain the existing CPU caller, its HAC/SUP fixtures, source-linked processing,
FP16 QKV preparation, and ABI4 CUDA primitive qualification. Their exact limits
remain in [the port plan](../port-plan.md), [architecture](../architecture.md),
and [CUDA review](../independent-review-cuda-abi4.md). The historical kernel trace
describes an older ABI3 starting point. Current ABI4 already supplies synchronous
half buffers, views, streams, transfers, and source half GEMM. It still lacks the
full model and decoder operations.

Reuse the original checkout's retained assets and source qualification read-only.
The CUDA13 source build and official CUDA12 package are distinct references.
The original source qualification still fails six literal accuracy artifacts due
to alternate-alignment row order. Its numerical agreement is useful evidence,
but it is not an overall artifact-equivalence pass. Never select using HG005.

## Assign owners without sharing files

Give each implementation card one owner and a separate worktree. The owner keeps
the card through implementation, verification, review fixes, and handoff. Name
the exact base commit and the dependency commits in every dispatch. An independent
reviewer checks the resulting head and actual receipts. A same-family second
agent is an independent review, not a cross-family panel.

Only the ABI integration owner changes `internal/tensor/abi.h`, the loaders,
shared backend registration, and shared build/source-pin files. Other owners add
their assigned files behind the frozen interface and hand patches to that owner.
Only the caller integration owner changes shared caller and CLI entrypoints.
P05 owns the thin finite `verification/gpucaller` command and handler registry.
Operation and acceptance owners submit serialized registration patches to that
owner. H01, S00, and D01 own their respective model/decoder checker entrypoints.
Their consumers add assigned files and request shared wiring, rather than edit
the common command concurrently.
Only the coordinator changes this board, acceptance policy, and source baselines.
Overlapping ownership means serialization or a new boundary, never concurrent
edits in one checkout. Parent-owned files are read-only to dependent workers.

Card paths are proposed ownership reservations. They do not imply those files or
commands already exist. Cards distinguish existing checks from checks that the
owner must implement. A dependency must pass its stated acceptance before native
downstream execution. Independent source-oracle preparation can proceed in its
own directory while a runtime dependency is held.

## Queue native work separately

Caller-stock's temporary native-yield request ends at **2026-10-23 17:18 UTC**.
That deadline releases only that request. Check caller-reference reservations,
other holds, running processes, and all existing admission policies afterward.
Never terminate another producer or treat a free lock as permission to bypass a
reservation.

The latest proposed caller-harness2 window is **17:18 to 18:23 UTC** on that date,
with re-coordination if not ready by **17:23 UTC**. Caller-reference reports no
conflict after 17:18 and plans a safe checkpoint by 17:15. Retrieval has no native
producer or reservation. The Go coordinator directly notified the live
`caller-harness2` agent at `w3N:p3` that this source-only planning pass has no
conflicting native reservation. Coordinate any later Go capture directly with
that owner. Do not infer permission from the scheduled end without current intake.

All native producers use the existing original
`/home/jake/Projects/nanopore_basecaller/results/stock/host.lock`, opened without
creation or truncation and acquired nonblocking. One resource coordinator admits
one bounded native unit at a time. CPU-heavy builds, memory-heavy captures, and
timed CPU work share this queue when required by the active hold. Preserve power,
cooldown, disk, process, and profile gates. Source-only investigation and small
pure checks do not reserve the GPU.

Capture a native boundary once with exact source, input, library, device, command,
clock, exit, and raw-log pins. Import, downstream implementation, and independent
review reuse that capture. Repair an importer from retained bytes. Do not rerun a
producer to repair a report. New producer execution requires an actual changed
input or an identified missing observation and a new immutable attempt.

## Dispatch in dependency order

The throughput checkpoint is the ABI/dispatch contract, source-oracle handshake,
then one bounded runtime unit. Independent model kernels, weight preparation,
decoder contract research, scheduling, output, and verification tooling can have
separate source owners. Shared interfaces and native execution remain serial.
Dozens of total owners can cover the board across waves. This session exposes
four concurrent agent slots including the coordinator. Do not claim dozens of
simultaneous local agents or a measured speedup from planning.

Inventory actual agent handles and active worktree claims before every restart.
Do not respawn an accepted or running producer. Every handoff names its commit,
owned files, exact checks, retained evidence, failed attempts, limits, and next
action. PRs target `main`; stack on dependency branches while the initial port PR
is unmerged. No card authorizes merging, force-pushing, deleting artifacts, or
writing the original controls.

## Read the cards

Validate the dependency graph and ownership reservations from the module root.

```sh
jq -e -f verification/gpu-plan/check.jq docs/gpu-caller/cards.json
jq -r -f verification/gpu-plan/render.jq docs/gpu-caller/cards.json > docs/gpu-caller/CARDS.md
```

The checker validates structure, dependency cycles, exact duplicate ownership
strings, and evidence-field presence. It does not resolve overlapping globs or
verify evidence payloads and verdict hashes. The coordinator must inspect actual
ownership and pinned evidence before dispatch or acceptance. A plausible evidence
string is not proof. Ordinary source tests must not launch native producers or
initialize a GPU. Native consumer verification uses explicit admission; host-only
import/replay does not.

Inspect one card or find the native queue without changing the board.

```sh
jq '.cards[] | select(.id == "S02")' docs/gpu-caller/cards.json
jq '[.cards[] | select(.resources | index("gpu-capture")) | {id,title,depends_on}]' docs/gpu-caller/cards.json
```

All remaining cards start as planned or source-ready. Source-ready means a
reviewed source checkpoint exists, not that the feature is accepted. A card closes
only with its independent verdict and required runtime evidence. Card count is
not a completion percentage. Research unknowns, especially opaque decoder ties
and quality arithmetic, prevent a credible seconds-to-completion estimate.

The Model the Domain principle makes dependencies and acceptance fields explicit
in the card data. Sequence Work into Verifiable Units keeps each card bounded by
a source capture, implementation, and reviewable result. Build the Lever makes
the board checkable and regenerable from that data. Prove It Works keeps source,
native, full-caller, biological, and performance evidence separate.
