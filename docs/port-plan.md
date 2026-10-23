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
| Chunk planning and padding | All three source algorithms, repeat padding, fixed caller chunks | Integrated variable and transformer scheduling |
| Model config and weights | Exact HAC/SUP configs and 50/139 consumed weight hashes/shapes | Other models and upstream revisions |
| CUDA tensor primitives | ABI4 ownership and raw half transfer; 37 exact source-GEMM/copy cases with guards; all 32 float32 precision mutants differ | Model operations, Koi descriptors/kernels, other devices and wider arithmetic/layout domains |
| GPU batch geometry | Host-tested fixed and variable membership, chunk tables, LSTM/transformer layouts and granularity transitions for both lanes | Extracted native oracle, device table bytes and captured mixed batches |
| GPU decoded assembly | Host-tested completion-order collector restoring source order before the accepted stitch unit | Genuine GPU decoded-chunk replay after the decoder units |
| GPU postcall replay | Host-tested reuse of the accepted postcall seam from decoded GPU reads, versioned sealed captures and producer-free replay receipts | Genuine GPU capture, stock header/record differential in all four modes, DS-tag predicate and independent review |
| HAC inference | 55 source-oracle fixtures and one genuine-weight score capture | Other inputs and device paths |
| SUP inference | 12 source-oracle fixtures, 287 stages, 15,228,786 exact float32 values and genuine 18-layer weights | Other inputs and device paths |
| SUP GPU weight preparation on host | QKV interleave, FP16 tiling and rounding match three full source fixtures and all 65,536 half patterns; int8 source/input preparation reviewed | Int8 arithmetic captures and Go implementation, FP8 preparation, other caches, device upload and graph integration |
| CRF decoding | CPU N=1 scans, posteriors, beam diagnostics and quality bits | Batched scans and GPU kernels |
| Stitching | Source algorithm fixtures including partial strides | Integration with variable chunk paths |
| Split and trim | Source-linked MUX trim, pore-split decisions, child metadata, adapter/primer detection, interval trim and filtering; two genuine default FASTQ controls | Positive biological split/MUX controls, other kits and modes |
| Scheduling | Selected reads, CPU batch one, cancellation, ownership, immutable captures and replay; owned CUDA streams at primitive boundary | Batching, GPU model scheduling, complete input accounting |
| BAM and FASTQ output | Raw FASTQ identity for HAC and SUP on one frozen UUID, including headers; Go-owned unaligned BAM writer reproduces the exact 733,623-byte uncompressed payload of one 32-record native stock BAM, including QNAMEs and float bits, with 17 checks, 12 controls and 5 executed faults, plus a pinned-source htslib oracle showing 12 byte-identical conditional-tag cases (TS/RX/pi/sp/mv, barcode, RNA poly-tail, duplex) | Compressed bytes and BGZF member topology (unclaimed), modbase tags, larger read sets, other modes |
| Complete caller | Two models on one predeclared development read, including default processing from retained core stages | Complete declared input sets, live default producer acceptance beyond the shared-stage source trace |

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
negative controls, and limits. GPU model graphs and Koi kernels remain unfinished.

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
