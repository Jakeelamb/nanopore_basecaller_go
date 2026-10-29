# Packed source-layout flow: candidate A

Selected design, derived from the supplied Phase A report and this worktree's
source. The first pure host unit is implemented in `internal/packed/plan.go`;
the forward/decode/owner APIs below remain proposals. No runtime acceptance is
asserted. The controller owns formatting, checks, evidence and integration.

## Implemented host entry

`NewDefinition(DefinitionInput) (Definition, error)` validates and copies the
interpreted config, source/model/config pins, convolution definitions, basecaller
maximum chunk and complete decoder scalars. A later model adapter supplies these
facts without a packed-to-model import. `Definition.Input` returns a copy.

`PlanBatch(Definition, scheduler.Batch, []Member) (Plan, error)` is the sole batch
geometry constructor. `Member` contains chunk ID, parent ID, parent offset and
raw length; sample buffers belong to the later forward boundary. Identities may
arrive in any order: placements determine source order. The plan checks exact
rounded lengths, offsets, membership and capacity against the maintained handoff;
it does not reconstruct scheduler admission or reset its persistent HAC state.

`Plan.Members`, `Input`, `Decoder`, `Frames`, `HAC`, `SUP`, `Definition`,
`Sequence`, `SourceOrder` and `UnboundContracts` provide actual public read
access. All reject a zero plan; all returned slices are defensive copies.
`Validate` rejects the zero value. HAC/SUP are a private sealed sum, with a
cross-family getter error. No score lease, device owner or native binding exists.

The ordinary `CudaModelRunner` input capacity is `N*chunk` samples. SUP's
populated prefix is `sum(rounded)+(logicalCount+1)*FirstConvPadding`; its scheduler
capacity charge instead uses `MaxConvPadding`. These gaps do not enlarge the
runner allocation. SUP's first convolution sees the full input view, including
unpopulated capacity of unknown initialization. HAC uses only its populated
prefix. The separately padded native `create_empty_input` timing helper is a
different call shape, not substituted here. Maximum RoPE extent derives from
`DefinitionInput.MaxChunk`, not the potentially shorter scheduler queue chunk.

`Decoder.Requirements` records segmented numerical shape/type requirements,
without claiming qualified program IDs. Source allocation formulas and positive
posterior bounds are host facts; `UnboundContracts` leaves native read/write,
initialization, interval mutation, attention, scratch, posterior mapping,
collision ordering, numerical authentication and ownership evidence unresolved.
NativeReference descriptor capacity includes undefined rows beyond its explicitly
initialized logical prefix. Source allocation capacity never implies reads.

`TestRetainedCPUAuxOptIn` uses the maintained `hacvcsaux.CompareNative` manifest
reader when both `PACKED_HAC_AUX_EVIDENCE` and
`PACKED_HAC_AUX_EVIDENCE_SHA256` are supplied. It accepts only the maintained
authenticated `first`/`all` selection, checks its word-corruption and independent
pin rejection, and verifies Plan's four arrays against that same matrix's first
auxiliary fixture. It produces no native capture. Paths and pins must remain
valid; absence of selectors skips this optional test, never qualifies this unit.

## Caller usage first

1. `internal/caller/gpu_invocation.go`, replacing only the variable branch of
   `Call`; fixed full/short and independently admitted fixed partial keep their
   own admission and graph contracts. `scheduler.Invocation.Call` is unchanged.

```go
func (i *gpuInvocation) callPacked(ctx context.Context, b scheduler.Batch) error {
    // Resolve pending IDs to parent coordinates and raw samples before enqueue.
    members, err := i.resolvePackedMembers(b)
    if err != nil { return err }
    p, err := i.packedModel.Plan(b, members) // copies and validates all geometry
    if err != nil { return err }
    scores, err := i.packedModel.ForwardPacked(ctx, p, members)
    if err != nil { return err } // invocation already retains every allocation
    decoded, err := i.packedDecoder.DecodePacked(ctx, scores)
    if err != nil { return err }
    work, err := i.bindPackedResults(p, decoded) // exact logical IDs and lengths
    if err != nil { return err }
    // Complete joins score retirement, stream and release errors before publish.
    // Any active-batch error becomes retained; preflight errors leave it idle.
    if err := i.owner.Complete(ctx); err != nil { return err }
    if err := ctx.Err(); err != nil { return err }
    return i.run.publish(b, work) // existing fallible collector commit semantics
}
```

2. `internal/model`, the HAC implementation of `ForwardPacked`; SUP is a separate
   sealed implementation selected when loading the model, never nullable fields.

```go
func (m *packedHAC) ForwardPacked(ctx context.Context, p *packed.Plan,
                                members []packed.Member) (*packed.Scores, error) {
    batch, err := m.owner.Begin(p, members) // owner records batch before allocation
    if err != nil { return nil, err }
    // Graph dispatch belongs to Go; these are private model operations.
    // Repeat-pad each raw member; upload flat input and four independent arrays.
    // conv0/1 + boundary repair; conv2 fast/fallback + scatter + repair.
    // Seven direction-specific encoded FLSTMs; row3 factorised gather.
    return m.runSourceGraph(ctx, batch) // [1,sum(t_i),1024], source layout lease
}
```

3. `internal/gpudecode`, a segmented schedule over borrowed source scores.

```go
func (d *PackedDecoder) DecodePacked(ctx context.Context,
                                    scores *packed.Scores) (packed.Decoded, error) {
    scan, err := d.bind(scores) // same plan, owner, stream and numeric program pins
    if err != nil { return packed.Decoded{}, err }
    // Go runs guides backwards over active logical segments, including terminals;
    // then beam histories/merges, forwards/posteriors, traceback and quality.
    // Frame operands are bounded views/row selections, not fixed Decode per chunk.
    return d.runSegmented(ctx, scan)
}
```

## Coordinate flow and membership

The immutable plan retains source packing coordinates through decode. HAC scores
are the source projection's compact NTC view. SUP scores retain padded encoder
block order after upsampling: a global score row maps to `(block,local score row)`
by division/remainder by the model-derived score rows per block (128 for this
SUP configuration). No second owned ragged score arena or model-output repack
terminates these coordinates. Decoder guide/history/posterior/output workspaces
have their own explicit coordinates, derived once from logical segment lengths.

`Member` binds chunk ID, parent offset, raw length and rounded placement to the
plan. Publication iterates this membership, never physical lanes or padding.
Each result has exactly `t_i` binary moves; sequence and quality lengths equal
`sum(moves)`. Parent offset never derives from an input, block or score offset.
Logical count may exceed HAC physical lanes or SUP configured batch capacity.

## Authority and integration

- Existing `gpubatch.NewAux`, `PackLSTM` and `DeviceTable` remain host packing
  authority. `RestoredIntervals` is the host multiplication helper, not evidence
  of opaque native interval mutations. Capture those transitions before freezing
  executable repair contracts; fresh per-batch aux needs no replay restoration.
- Source authority: `AuxiliaryData.cpp:24-155`, `ConvStack.cpp:193-324,376-459`,
  `FLSTMStack.cpp:110-145,219-236`, `FactorisedLinearLayer.cpp:30-106`,
  `TxModules.cpp:540-575,837-900`, `TxModel.cpp:20-50`, and
  `CUDADecoder.cpp:20-169`, `CudaCaller.cpp:609`, `CudaModelRunner.cpp:13-58`, and
  `WorkingMemory.cpp:83-95`, under the authorized pinned Dorado tree.
- Existing barriers in `gpu_invocation.go` and `gpu_run.go` stay until the new
  complete plan/backend/decoder tuple is qualified. Source inspection supplies
  geometry and call contracts; opaque Koi internals still require native evidence.

## Genuine producer handoff, proposed only

Extend the maintained HAC observer with a distinct `packed-source` variant:
construct real `AuxiliaryData` from rounded development/validation sizes, capture
the four host uploads, create device aux, and traverse authentic CRFModel with
aux non-null. Capture flat convolutions before/after interval repair, scatter,
all seven encoded recurrent layers with backing/slack, row3 gather and scores.
Capture interval words before/after each repair and any replay restoration.
Capture both the genuine fast convolution and source fallback dispatch variants.

Extend SUP `source-graph` with a distinct `packed-source` producer that calls
`model.forward(input, aux)` and its source-staged equivalent using the same real
aux. Record input gaps, sample table, every convolution's LUT/padding/M extents,
weight tilings, RoPE table/positions, all 18 attention layers, upsample and scores.
Include logical chunks spanning several 64-step blocks; fixed `nullptr` aux or
fixed T64 attention cannot produce this evidence.

Extend the decoder observer with genuine variable descriptors for S256 and S1024,
using the corresponding retained authentic score output and aux; capture all four
original stages, descriptors/results, scratch/path, posterior prefix and output.
These stages are oracles only: production Go retains schedules and decisions.

Freeze independent source/model/weights/Koi/build/library/kernel/input/program
pins before output. Include tiny/unequal/max sizes, ties/interleaving, multiple
32-lane HAC parts, logical count above physical N, full/short alternation, SUP
block totals around multiples of four, cross-block attention and max RoPE position.
Poison uninitialized capacity and inspect guards; never infer a read mask from
observed values. Reject corrupt units/layouts/directions/descriptors, unsafe
aliases/spans, wrong dtype/tags/scales/device/stream and wrong pins before launch.
Retain raw exits, clocks, logs and failed attempts in ignored evidence storage.
This assignment generated no producer artifacts or runtime receipts.

## Selected integration rules

The completion barrier and a final cancellation check precede entry to existing
publication. Publication is synchronous and fallible; a later Collector.Add
error can follow earlier successful adds. Preserve nonzero failure and accounting,
without claiming atomic rollback. A separate batch-commit change needs its own
contract. Failed synchronization retains all possibly referenced owners and weight
leases; later computation is held until cleanup. Preflight rejection before Begin
leaves the invocation idle. Decoder parameters and numerical program identities
are frozen with the plan; active-frame schedules derive from its segment lengths.
Host geometry never binds unknown native read/write/attention/collision contracts.

## Implemented host unit and evidence

`internal/packed` now provides immutable `NewDefinition` and `PlanBatch` values.
The maintained scheduler supplies membership; existing auxiliary packers remain
the authority for host arrays. The plan separates sample, score and guide-row
coordinates and derives active-frame CSR without scanning finished members.
HAC scratch bounds reject before packing allocations. SUP definition checks
enforce the source's first, later and final weight views with checked products.

At the frozen source reviewed on 2026-10-29, both configured independent
same-model reviewers reported no findings after the three corrections. Focused
host tests passed, including highly unequal lengths through the maintained
scheduler. The retained genuine CPU auxiliary comparison and its corruption
controls passed, followed by a comparison of the plan's first-case arrays.
No native producer was repeated.

`CGO_ENABLED=1 go test -p 2 -json ./...` passed 877 root tests and 1,929 subcases;
93 opt-ins were skipped. The maintained `verification/verify.sh` command passed
its CPU-package/native-algorithm scope. Raw receipts and retained payloads stay
outside Git under ignored artifact paths.

This unit does not dispatch GPU work. Packed native call bindings, ABI6,
segmented decoding and genuine HAC/SUP graph evidence remain open. The approved
GPU registry remains empty; these checks do not qualify basecalling or biology.
