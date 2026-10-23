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

## Remaining full caller boundaries

| Boundary | Acceptance required |
| --- | --- |
| POD5 loading | Native signal bytes, calibration, UUID and metadata fixtures, invalid-input rejection |
| Scaling | Full native ScalerNode and host SIMD parity, including open-pore adjustment and all supported modes |
| Chunk planning and padding | All three production algorithms and exact model input bytes |
| Model config and weights | Exact architecture and tensor names, shapes, dtypes and content |
| HAC inference | Pinned factorized LSTM native forward tensors on development inputs |
| SUP inference | Pinned transformer native forward tensors on development inputs |
| CRF decoding | Native guide, posterior, beam tie behavior, quality and move outputs |
| Stitching | Exact sequences, qualities and moves including partial strides and variable chunks |
| Split and trim | Native signal split identities, intervals, adapters and quality filtering |
| Scheduling | Batching, cancellation, bounded memory, deterministic replay and error accounting |
| BAM and FASTQ output | Canonical records plus separately demonstrated byte identity |
| Complete caller | Same declared development reads, all parents accounted for, end-to-end native comparison |

Each boundary remains open until its own evidence exists. The package tests and
CPU CI cannot qualify a full caller, GPU behavior, biological accuracy, or speed.

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
backend now has a qualified allocation, layout, synchronization, and lifetime
contract. GPU streams, half precision execution, and Koi kernels remain
unimplemented and need separate qualification.

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
