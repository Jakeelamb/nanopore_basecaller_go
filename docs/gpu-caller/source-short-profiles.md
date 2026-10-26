# Default simplex short profiles

The Go graph planners now represent the pinned source's extra 6144-sample queue.
`SourceShortHACGPUProfile` is N64/chunk6144, with 1024 output frames.
`ShortFixedSUPGPUProfile` is N32/chunk6144, encoder T512 and score T1024.
The constructors use the existing actual graph operations at those dimensions.
The caller keeps the normalized parent on its source-selected profile and calls
the matching fixed graph constructor. No qualification or bundle approval was
added, and no native/GPU operation was run to establish these profiles.

`CudaCaller.cpp:410-465` inserts half the requested chunk size for default simplex
unless variable chunks and transformer are both selected. The default profile
lists remain `[12288,6144]` for fixed HAC, variable HAC and fixed SUP, and `[12288]`
for variable SUP. The literal `BasecallerNode.cpp:93-105` comparison selects 6144
for normalized length 6143, and 12288 for lengths 6144 and 6145. Packed HAC still
requires both source profile sizes, but a new fixed profile is not a packed GPU
implementation. Packed graph/indexing and genuine partial-lane work remain open.

SUP's typed boundary adds `SUPGeometryShort=3` without changing ABI5, the existing
wire values 1/2, operation numbers, scalar policy or function signatures. The
matching C enum and native `sup_roles`/entry validation admit value 3 and derive
N32/T512/chunk6144. The native changes are limited to those dimension/validation
sites. An old backend that only understands values 1/2 cannot qualify this
profile. Its new native build plan must freeze the updated `abi.h` and
`verification/tensor/sup_primitive.inc`, together with the complete actual source
and compiler/runtime closure. Host fake-loader geometry transport is not that
native build or numerical qualification.

## Ownership and allocation qualification

The source creates one `CudaCaller::m_module` at `CudaCaller.cpp:153`, warms every
profile through it at lines 160-180 and uses it for queued tasks at line 731.
Model parameters and transformed GPU weight caches therefore survive profile
changes. For example, `nn/FLSTMStack.cpp:167-218` populates device caches only on
the first layer use. This is shared model ownership across the profile queues.

The HAC `WorkingMemory` object itself is local to each `CRFModelImpl::run_koi`
call at `basecall/model/CRFModel.cpp:92-106`, which reserves the current geometry
and allocates one backing tensor for that call. It is not a permanently shared
workspace spanning all input sizes. The Go HAC graph similarly creates its
profile's workspace in `HACGPU.forward` per invocation. SUP forward intermediates
are also invocation-owned; its uploaded parameters and RoPE table are model-owned.

The current Go caller has one engine/device and a separate stream/graph for each
used profile. Each graph constructor loads/uploads its own parameter caches,
and each profile invocation retains its input buffer. Using both queues therefore
duplicates persistent model/cache and input ownership relative to the source's
single module. The synchronous run permits one transient graph/decode call at a
time, so it does not imply simultaneous full and short transient workspaces.
Source-compatible cache sharing is not supplied by an existing constructor.
Qualify this explicit ownership difference, or implement a real shared model/cache
owner before accepting allocation/lifetime equivalence. Score agreement alone
cannot close full initialized-range, alias, guard or cache-lifetime endpoints.

## Required native evidence

Use the pinned source and predeclared development inputs with
`DORADO_EXTRA_CHUNK_SIZES` unset. Retain a full-active N64 HAC6144 native graph
oracle and N32 SUP6144 oracle with every declared intermediate, parent allocation,
initialized/live range, cache, descriptor, guard and sentinel. HAC's source reserve
traversal uses T1027 after reserve padding while actual recurrence uses T1024;
do not collapse them. SUP must retain all convolution/encoder/upsample/score stages
at encoder T512 and preserve its invariant 4096-row RoPE table.

Compare Go inputs and all retained graph values with these new references. Add
foreign-profile rejection controls, the exact normalized 6143/6144/6145 routing
controls, alternating full/short batches against one native module, and independent
native partial-lane controls before partial batches are admitted. Decoder plans,
moves, sequence/Q and postcall/output endpoints still need genuine differentials.
Do not reuse a full/tiny profile's numerical qualification for the short profile.
The production bundle registry stays empty until actual independent pins exist.
