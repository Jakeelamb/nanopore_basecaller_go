# Caller port and acceptance

The target is a Go implementation of the stock simplex DNA caller, starting with
the two pinned models below. A subprocess wrapper around Dorado does not meet
this target. The original experiment workspace remains the stock control.

| Model | Network | Effective stride |
| --- | --- | --- |
| dna_r10.4.1_e8.2_400bps_hac@v6.0.0 | Three convolutions and seven factorized LSTMs | 6 |
| dna_r10.4.1_e8.2_400bps_sup@v5.2.0 | Five convolutions and 18 transformer layers | 6 |

## Throughput checkpoint

- Blocking first steps: pin the source, map production call sites, compile the
  first source oracle, and preserve the stock checkout before implementing.
- Independent workstreams: chunk planning and padding, standardised DNA pA
  scaling, and decoded chunk assembly have separate owners and directories.
- Shared mutable state: the coordinator owns source pins, the parity command,
  coverage, and integration. Workers own separate packages and oracle adapters.
- Smallest safe decomposition: verify each production computation against a
  native oracle before joining it to the next stage. Capture inference outputs
  once when that boundary exists. Replay them for decoder and assembly work.

## Completion sequence

The Go-owned HAC and SUP GPU graphs, decoder call chain, explicit cross-read
batching, shared postcall, and FASTQ/BAM output are integrated in source.
No production GPU bundle has been approved. The empty bundle registry prevents
the CUDA CLI from completing a basecall. The [production run boundary](gpu-caller/production-run.md)
records the implemented path and excluded domains. The remaining work follows
these verification boundaries. Component or host checks do not complete a later
boundary.

1. Qualify the integrated fixed HAC GPU command and maintained tensor tests.
   Run the genuine CPU fixtures and fake-backend rejection controls, then compare
   all 64 lanes and the padding probe against the pinned native graph.
2. Use the qualified fixed HAC native GPU decoder reference recorded in
   [current evidence](cpu-abi5-acceptance.md#bounded-hac-native-gpu-reference).
   Its 32 buffers cover four stages and five numeric identities at score shape
   `[64,32,1024]` with 256 states. Compare guides, beam decisions, posteriors,
   moves, sequence, and quality against the Go-owned candidates. SUP references
   and numerical-payload and mask rejection controls remain open. The external
   anchor corruption control checks authentication only. This reference does
   not complete D01, full graphs, packed or partial batches, or the GPU caller.
3. Qualify the integrated SUP graph and full-size HAC and SUP geometry.
   Compile the native numerical operations and compare every declared retained
   stage before running complete graphs on full chunks. Qualify shared full/short
   model caches and stream-order alternation separately.
4. Preserve the bounded [native HAC host-packing evidence](gpu-caller/hac-vcs-aux.md)
   at its recorded paths and pins. Source import does not transfer its receipt.
   Implement and compare the remaining packed GPU operations and decoder indexing
   for both models. Qualify mixed and partial batches, lane identity, ownership,
   and the remaining legal beam collision schedule.
5. Qualify the existing graph/decoder connection to the Go caller, batch scheduler,
   collector, and shared postcall path. Compare complete declared development
   and validation inputs, input accounting, cancellation, records, tags, FASTQ,
   and BAM artifacts against stock. Preserve raw-artifact failures separately
   from semantic record comparisons.
6. Run the registered biological and performance acceptance in
   [GPU acceptance](gpu-acceptance.md). Freeze the required timing schedule and
   budget before clocks. Finish with `go test ./...`, the maintained parity
   command, and an independent review of the integrated result.

The GPU path requires genuine native graph and decoder comparisons and independent
approval of a complete production bundle. Decoder reference preparation and
packed GPU implementation can proceed independently. The coordinator serializes
compiler and GPU execution through the existing stock host lock. Each worker owns its unit through numerical evidence
and integration handoff. Shared caller and tensor interfaces have one integration
owner. Imported graph and decoder source does not inherit runtime acceptance
from isolated candidates or change the stock workspace.

## Exactness contract

Try exact equality first. Compare sample calibration and half bits, chunk
coordinates, model input bytes, logits, guide and posterior tensors, moves,
sequence, quality, split intervals, and tags at their own boundaries. A tolerance
at one tensor boundary requires an explicit decision and does not qualify exact
downstream output. CPU decoding does not establish CUDA decoding parity.

Compare canonical records by read identity for semantic BAM parity. Test raw BAM
hash equality separately. Headers, concurrent record ordering, and compression
are part of the raw artifact contract. Dorado enables deterministic Torch
algorithms, so do not assume stock inference is nondeterministic.

`BasecallerNode.cpp` repeat-pads short chunks. The comment in the chunk planner
about zero padding does not describe the consuming implementation.

## Boundary evidence and remaining work

| Boundary | Observed evidence | Remaining acceptance |
| --- | --- | --- |
| POD5 loading | Two independent native fixtures, copied signal/calibration/UUID/metadata and selection guards | Broader input versions and complete-file processing |
| Scaling | Standardized DNA formula and half bits; one whole caller endpoint | Full native ScalerNode/SIMD and other scaling modes |
| Chunk planning and padding | All three source algorithms, repeat padding, fixed CPU chunks and GPU profile scheduling source | Genuine integrated variable and transformer execution; packed GPU paths remain unfinished |
| Model config and weights | Exact HAC/SUP configs and 50/139 consumed weight hashes/shapes | Other models and upstream revisions |
| CUDA tensor primitives | ABI4 ownership and raw half transfer; 37 exact source-GEMM/copy cases with guards; all 32 float32 precision mutants differ | Model operations, Koi descriptors/kernels, other devices and wider arithmetic/layout domains |
| GPU batch geometry | Host-tested fixed and variable membership, chunk tables, LSTM/transformer layouts and granularity transitions for both lanes; bounded native HAC host packing in [HAC VCS auxiliary packing](gpu-caller/hac-vcs-aux.md) | Remaining native geometry comparisons, device table bytes and genuine mixed/partial batches; packing receipts retain their original path bindings |
| GPU decoded assembly | Host-tested completion-order collector restoring source order before the accepted stitch unit | Genuine GPU decoded-chunk replay after the decoder units |
| GPU postcall replay | Host-tested reuse of the accepted postcall seam from decoded GPU reads, versioned sealed captures and producer-free replay receipts | Genuine GPU capture, stock header/record differential in all four modes, DS-tag predicate and independent review |
| HAC inference | 55 source-oracle fixtures and one genuine-weight score capture | Other inputs and device paths |
| Fixed HAC GPU graph | Bounded native unit evidence for convolution, fused recurrence/cache and projection; Go-owned fixed composition source, caller connection and an unloaded closed-build backend | Current guarded CPU fixtures, actual all-64 graph comparison and padding probe; half fallback runtime, full chunks, VCS and genuine caller qualification remain open |
| Source-default fixed HAC geometry | Integrated N64/chunk12288/T2048 plan and runtime dimensions; independently reviewed isolated candidate had 23 host roots and vet passed with retained input snapshots | Full-size native oracle and runtime comparison; current H08 witness still uses chunk192 |
| SUP inference | 12 source-oracle fixtures, 287 stages, 15,228,786 exact float32 values and genuine 18-layer weights | Other inputs and device paths |
| SUP GPU weight preparation on host | QKV interleave, FP16 tiling and rounding; S02 int8 packing matches all 34,992 native rows; [S05 FP8 packing](gpu-caller/s05-fp8-caches.md) matches 72 native before/after buffers across all 18 layers; graph cache composition exists in source | Genuine GPU upload and remaining cache comparisons; S05 composition does not separately qualify every intermediate half cast |
| Fixed SUP GPU graph | Integrated implementation follows the fixed ATen convolution route and seven explicit primitives per encoder layer; independently reviewed isolated candidate had 14 base host checks and two later uploaded-cache observation controls passed | Native compilation and all 733 retained tensor comparisons; runtime/current CGO1/CPU fixtures, full chunks, VCS and genuine caller qualification remain open |
| Source-default fixed SUP geometry | Integrated two-profile implementation derives chunk12288/T1024 and scoresT2048; independently reviewed isolated candidate had 15 CGO0 and 19 CGo host roots plus vet passed | Full-size native oracle/graph qualification. Host fake evidence retains an eight transient type-probe output gap and is not a closed compiler artifact |
| GPU guide scan candidate | Go reverse-time scheduling and typed one-frame SM120 PTX boundary; host/fake checks only | Native build/JIT, D01 numerical differential, packed indexing, posterior scans and independent review |
| GPU beam and posterior/quality candidates | Integrated caller call chain; independently reviewed isolated Go scheduling, ordering and quality logic with closed numerical kernel operations; source and host controls passed | Native compilation/JIT, authenticated decoder oracle, complete numerical comparisons, legal beam collision schedules and genuine caller qualification |
| CRF decoding | CPU N=1 scans, posteriors, beam diagnostics and quality bits; N=1..4 lane composition implemented, replay pending compatible backend; fixed GPU decoder composition exists in source | General batched scans, direct CPUDecoder multi-chunk parity, packed GPU indexing and genuine GPU kernel qualification |
| Stitching | Source algorithm fixtures including partial strides | Integration with variable chunk paths |
| Split and trim | Source-linked MUX trim, pore-split decisions, child metadata, adapter/primer detection, interval trim and filtering; two genuine default FASTQ controls | Positive biological split/MUX controls, other kits and modes |
| Scheduling | Selected reads, CPU batch one, cancellation, ownership, immutable captures and replay; bounded synchronous cross-read GPU batching, profile dispatch and ordered retirement in source | Genuine integrated scheduling/input-accounting qualification and partial fixed batches; automatic sizing and concurrent timeout scheduling are absent |
| Shared GPU model ownership | Integrated run-owned HAC/SUP caches with explicit per-forward full/short plans; synthetic host ownership controls and qualifier API migration | Genuine native full/short alternation on both stream orders, every graph stage and score, persistent upload/allocation counts and device memory; see [shared caches](gpu-caller/shared-model-caches.md) |
| BAM and FASTQ output | Raw FASTQ identity for HAC and SUP on one frozen UUID, including headers; Go-owned unaligned BAM writer reproduces the exact 733,623-byte uncompressed payload of one 32-record native stock BAM, including QNAMEs and float bits, with 17 checks, 12 controls and 5 executed faults, plus a pinned-source htslib oracle showing 12 byte-identical conditional-tag cases (TS/RX/pi/sp/mv, barcode, RNA poly-tail, duplex) | Compressed bytes and BGZF member topology (unclaimed), modbase tags, larger read sets, other modes |
| Complete caller | Two CPU models on one predeclared development read, including default processing from retained core stages; gated GPU composition exists in source | Complete declared input sets, live default producer acceptance beyond the shared-stage source trace, genuine GPU endpoints and independently approved production bundle |

The table retains bounded unit evidence, including historical isolated-candidate
checks. HAC/SUP full/short plans, graph and decoder composition, cache ownership,
batch scheduling, and output are now connected in source. Packed GPU graphs,
decoder indexing, and partial fixed invocation remain unfinished.
The [HAC VCS host-packing unit](gpu-caller/hac-vcs-aux.md)
records bounded original-host comparisons; its receipts retain their original
path bindings and do not qualify a packed GPU graph in this checkout.

The linked unit documents describe the exact fixture and runtime limits. A
bounded pass does not close the remaining column. Package tests and CPU CI do
not qualify a full caller, GPU behavior, biological accuracy, or speed.

## GPU boundary

### Native model stage checkpoint

- Blocking first steps: verify the reused development distribution and prove an
  in-process native allocation, matrix operation, copied result, and rejected
  invalid handle before running a model graph.
- Independent workstreams: one owner builds the tensor ABI and typed Go owner.
  Another traces the exact model graphs and weight mappings. Go model execution
  starts against their agreed interface.
- Shared mutable state: development libraries and stock weights are read-only.
  New builds and receipts use this repository's ignored storage. The concurrent
  original source build and its GPU schedule are outside this port's ownership.
- Smallest safe decomposition: native tensor operations, a synthetic factorized
  LSTM comparison, then the pinned HAC graph. SUP has its own graph and oracle.
  A CUDA 13 CPU backend does not qualify stock CUDA 12.8 GPU parity.

Two designs were considered: a whole-caller C ABI and a narrow tensor C ABI.
A whole-caller ABI would preserve the original implementation behind a Go
entry point. The selected design gives Go ownership of caller decisions and
keeps native tensor operations behind a typed backend boundary. The CPU float32
backend has a qualified allocation, layout, synchronization, and lifetime
contract. ABI4 additionally qualifies owned CUDA streams, half buffers and views,
raw transfers, and the source half-GEMM operation on 37 declared cases. The
[independent review](independent-review-cuda-abi4.md) records its device, runtime,
negative controls, and limits. GPU model graphs and their caller composition
exist in source. Complete native graph/decoder qualification remains open,
alongside packed GPU operations, partial fixed batches, and legal beam collision
schedules. No production GPU bundle has been approved.

The local Dorado distribution contains runtime libraries but no Torch headers.
Koi inference and decoding kernels are native dependencies. Rewriting those
kernels in Go would change the numerical and performance target. Preserve their
identity where possible and record every retained native dependency.

## Maintenance

Keep the pinned source manifest, oracle drivers, replay command, and coverage
document together. An upstream source change invalidates old parity evidence.
Review a source-pin update as a separate change and rerun affected boundaries.
Keep generated fixtures and raw receipts under ignored `artifacts/` or external
storage. Keep their hashes and concise qualification summaries in Git.

Never use HG005 to choose a port implementation. Reuse HG002 development and
validation evidence. Do not run or restart stock acquisition, bootstrap,
full-reference jobs, or full GPU cohorts as part of a cheap parity check.
