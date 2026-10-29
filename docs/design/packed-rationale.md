# Source-layout flow rationale

## Problem
Default HAC/SUP VCS changes coordinates between flat input, recurrence or encoder
blocks, and logical decoder chunks. Existing fixed plans cannot express these
changes. Go already implements host aux/four-array packing and ABI5 I32 storage;
the missing work is packed GPU boundaries and segmented Go decoding. Fixed
full/short and independently gated fixed partial must remain valid separate paths.

## Usage (caller's view)
The three call sites in [packed-usage.md](packed-usage.md) are the specification:
plan from scheduler membership, forward a source-faithful graph, decode the score
lease, then publish by original chunk identity. Scheduler Call/Drain stay intact.

## Shape
One private immutable `packed.Plan` owns all coordinate conversions and logical
membership; sealed HAC/SUP variants avoid required-but-optional fields. Source
score coordinates remain live into decoder views; terminal rows, posterior
prefixes and output extents derive from the same segments. Model and decoder
cannot invent a second geometry. Per boundary-discipline, validate external
placements and native operands at entry; per separate-before-serializing, each
invocation owns its stream, aux and scratch. Interface depth is high: three calls
hide packing, convolution repair/scatter, encoded recurrence/gather, SUP LUT/RoPE
and cross-block attention, four decoder stages and resource retirement. Native
wire types stay inside tensor; Go owns dispatch/state/decisions/publication.

## Synthesis decision
The controller selected A with explicit completion/failure/weight ownership and
complete frozen decoder parameters plus derived active frames from B. The
selection record and manual graft below remain authoritative. This implementation
reads the selected design and cross-judge; it does not reopen the arena.

## Tradeoffs accepted
- We accept source-layout indexing inside decode in exchange for direct native
  descriptor comparison and avoiding a second full score materialization.
- We accept a new independently qualified ABI6 tuple in exchange for freezing
  ABI5's existing CPU13/SUP compiled tuple and keeping incompatibility explicit.
- We accept explicit unqualified kernel spans/collision cases as implementation
  blockers in exchange for preventing guessed Koi behavior from becoming authority.

## Alternatives considered
An owned ragged score arena would end model coordinates at a gather boundary and
simplify decoder views, but adds allocation/copy/rounding/lifetime semantics and
hides less of the actual source handoff; it is the distinct controller-assigned
alternative, not part of this candidate. Fixed Decode per chunk exposes scheduling
and cannot preserve packed native semantics; a Real mask cannot encode remapping.
A whole native caller hides computation by surrendering Go control and fails the
port's ownership requirement.

## Open questions and risks
Which exact Koi interval mutations, read/write/scratch ranges, attention windows
and legal CRC collision schedules will the genuine producer establish? Can each generalized
numeric primitive retain its existing mathematical contract with segmented rows?
These are evidence questions; this design does not answer them by assumption.
Source references and proposed observers/rejections are in packed-usage.md.

## First implementation unit and next step
`internal/packed/plan.go` now implements the immutable host plan, copied model
definition boundary, source coordinates and decoder schedules, with host test
source in `plan_test.go`. These tests have not run. Definitions contain actual
public constructors and readers rather than requiring another package to set
private fields. No ABI6 schema, backend, forward/decode computation, resource
owner or publication rewrite belongs to this first unit.

Source inspection distinguishes ordinary SUP's full runner input view from HAC's
populated prefix and the separately padded native timing helper. The host plan
records ordinary-runner geometry and leaves unpopulated native reads/initialization
unbound. Basecaller maximum chunk is separate from full/short queue chunk to
preserve maximum RoPE position. Decoder program requirements describe shapes and
types without manufacturing program identities. No full score arena is added.

The controller must format/check this source, review the contract and bind retained
CPU auxiliary evidence with matching pins and paths. Native observers then need
aux-bearing HAC/SUP model and variable-descriptor decoder captures before any GPU
unit is admitted. Later consumers use this single host geometry authority.

This first unit adds host source and test source; no formatter, build, test,
native/device work or timing was executed. Runtime qualification, port closure, end-to-end caller acceptance,
HG005 selection, biology and ALT retention remain unclaimed and unchanged.
