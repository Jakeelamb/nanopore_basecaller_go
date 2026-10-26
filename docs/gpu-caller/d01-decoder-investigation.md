# D01 decoder static investigation

Status on 2026-10-25 UTC is **partial static contract recovery**. D01 is not closed. No target, GPU, compiler, model or decoder execution occurred. Static analysis identifies recoverable arithmetic and layouts, but does not establish numerical parity, complete tie semantics or runtime-selected code.

This investigates pinned Dorado `8b8fc5d36a9c0baab262a743cb175b5878e38ca6` and only the selected `decoder_lib.cu.o` archive member. The Go checkout was HEAD `3c88e3442940f3b69f5a00fbe762c602b7527199` with existing dirty work preserved.

## Retained identities and scope

Raw evidence and reproducible static scripts are in `artifacts/decoder-static-20261025-001/`. These are ignored local artifacts, not source or newly generated native fixtures for Git. Commands have argv, UTC start/end, exit code and stdout/stderr files. Tool executables and output manifests are SHA-256 pinned. Analysis ran with CPUs 0 and 1 and address-space limit 3 GiB. See `effective-analysis-status.txt` and `effective-analysis-address-space-kib.txt`.

| Input | SHA-256 |
| --- | --- |
| Selected libkoi.a | `17cf3083c5db538f1e275b0a41ac19560b5986c143634e8a25540b9735f5805f` |
| decoder_lib.cu.o | `930b35afa9e352b412b58e07a43d693630280ed9f422dc324106b2b8e700758f` |
| Extracted SM120 PTX | `7e46f59645025bca2cf7a519c9f0334b5435942ca9006d0dec5faa6d3e63b01a` |
| Extracted SM120 cubin | `292621a0efb7da3a9345797b469d73fb04ee04a88ad1aa0e52c94944214da938` |
| CUDADecoder.cpp | `e913013043ad0e2d352551b21adc74a7dcca7417e4b89a791978ce36a985e297` |
| Matching koi.h | `8b4115c2e92ebf1acf1db2aa76bdc9b3a22768cb595b92363f88e44ff92d3b88` |

Archive path is `/home/jake/Projects/nanopore_basecaller/software/source-build-qualification-20261023/dependencies/libkoi-0.7.5-Linux-x86_64-cuda-13.0/libkoi-0.7.5-Linux-x86_64-cuda-13.0/lib/libkoi.a`. No other archive member was extracted or disassembled. The selected object contains cubins for SM75, SM80, SM86, SM89, SM90a, SM100a, SM120 and SM120a, plus PTX 9.0 targeting SM120. Only SM120 code was selected for semantic inspection. Its PTX retains original source-line mappings and inlined function names. Those mappings are compiler metadata, not recovered original source text.

REA was consulted through cached `npx --offline -y rea-agents@6.4.0`. Its `inspect-binary-layout` command returned `provider_unavailable` because `REA_PWNTOOLS_PYTHON` was unconfigured and the required pwntools provider was unavailable. The complete response is `rea-layout.stdout`, exit 1 in `commands.tsv`. No provider was installed or global wrapper repaired. No Ghidra analysis was launched because line-annotated selected PTX and relocatable host disassembly answered this bounded slice without a whole-object analysis session. Findings below are ordinary static-tool evidence, not REA Evidence IDs.

Two failed static attempts remain. An RLIMIT_NPROC=128 limit blocked shell forks and was removed while retaining CPU/address-space limits. Initial SASS function filters incorrectly used C++ mangled prefixes and returned no functions; the warning remains in `sm120-sass.stderr`. Correct C symbol filters succeeded. `selected-sass.stdout` includes SM120a as well because cuobjdump's architecture selection included it; `exact-sm120-sass.stdout` instead disassembles the separately pinned SM120 cubin and contains only the four selected kernels.

## CPU and kernel boundary, observed

`external/dorado-source/dorado/basecall/decode/CUDADecoder.cpp:20-135` builds descriptors and buffers, then invokes back guide, beam search, posterior and final decode in that order. Its public matching header is `koi.h:469-544`. All scores are half; scalar arithmetic parameters are float32. The four host functions return `int` CUDA status and differ only in entry point and final `move_pad` parameter.

Their shared arguments, in order, are stream, chunks, chunk_results, num_chunks, half scores, score_clamp, score_channels, aux, path, moves, optional weights, sequence, qstring, q_scale, q_shift, beam_width, beam_cut and fixed_stay_score. Final decode appends integer move_pad. Dorado passes null optional weights.

The selected x86-64 object symbols are `host_back_guide_step` at `.text+0x1270`, `host_beam_search_step` at `+0x1610`, `host_compute_posts_step` at `+0x19c0`, and `host_run_decode` at `+0x1d80`. Relocations in the four corresponding `host_*.stdout` files show `__cudaPushCallConfiguration`, a matching device stub, and `cudaGetLastError`. Arithmetic is in the device code, not these wrappers. Wrapping those four functions in Go is an oracle boundary, not a completed decoder port.

For positive score channel count C, the host launch code selects one block per chunk for back guides and posterior scans, with `min(C/4,1024)` threads. Beam uses one 32-thread block per chunk. Final decode uses 32 threads and `ceil(num_chunks/32)` blocks. Beam dynamic shared memory is the maximum of `8*(C/4)` and `120*beam_width` bytes; posterior dynamic shared memory is `12*(C/4)` bytes. These formulas are recovered from wrapper disassembly, and are not independent support qualification for arbitrary values.

## Scratch and output bounds

Let S=C/4 be state count, W be beam width, and chunk i have score-frame start a_i and length L_i. Source fixed descriptors are four int32 words `[i*T,T,i*T,0]`. Packed descriptors copy the source chunk table and set the third word to its start and fourth to zero. PTX accesses shown in this investigation use the first two words for the relevant stage bounds. Do not infer that the other two words are an interchangeable application schema.

For source fixed allocation, aux capacity is `N*(T+1)*(C+4*W)` bytes. Source path capacity is `N*(T+1)` int32 values. Moves, sequence and quality each have N*T byte capacity and begin zeroed. Descriptor storage is N*16 bytes; result storage is N*32 bytes and begins uninitialized.

The PTX addresses establish the following scratch partition. Define `U=a_last+L_last+N`, where last is descriptor N-1. Descriptor ordering and nonoverlap therefore matter.

| Region | Byte offset within aux | Addressed content |
| --- | --- | --- |
| All backward guides | 0 | U*S float32 values |
| Guide rows for chunk i | `4*(a_i+i)*S` | `(L_i+1)*S` float32 values |
| All packed beam history | `4*U*S` | U*W four-byte records |
| History for chunk i | `4*U*S + 4*(a_i+i)*W` | `(L_i+1)*W` four-byte slots |

Pointers are proven by `back_guide_step.ptx:45-63` and `beam_search_step.ptx:64-100`, with their original PTX line correspondence in `kernel-index.json`. Guide terminal row is initialized to zero. Kernel loops write valid rows, rather than defining every capacity byte. History slots beyond the retained beam count are not automatically initialized evidence. Capture masks must follow actual source writes; an allocation dump alone does not justify comparison of all bytes.

Each history record is read as a 16-bit state and two bytes. Final traceback sign-extends the state into path, uses the first byte to select a preceding beam slot, and derives move as `second_byte==0`. This establishes a four-byte wire record. The proposed semantic names are state, predecessor and stay flag, with the latter two names inferred from these uses. The temporary beam front is a distinct 12-byte shared record, with a 32-bit merge key at offset 0, float32 score at offset 4, 16-bit state at offset 8 and two bytes at offsets 10 and 11. Do not confuse the two records.

`compute_posts_step` writes float32-derived, round-to-nearest half values back into the score allocation. This is an in-place destructive stage, proven by its output pointer originating from parameter 3, the same half score pointer that earlier stages read. Final quality reads half values from that changed allocation. Retain an immutable original score payload before any oracle execution. Do not treat the original model score digest as the post-posterior score digest.

The 32-byte chunk result has observed writes at offsets 0/4 for start/length, +12 for beam terminal score, and later +8/+16/+20/+24/+28. Decode's arithmetic supports interpreting +8 as emitted count, +16/+20 as stay/move fractions, +24 as zero in the inspected path, and +28 as an error summary, but these latter semantic labels remain provisional until staged oracle controls. Preserve raw eight-word results with explicit bit interpretation rather than inventing a fully established public struct.

Packed HAC and transformer capacities differ from the fixed formulas. See `CUDADecoder.cpp:36-91`. This pass does not close packed/VCS indexing or all capacity branches. Start D01's first oracle with fixed HAC and SUP layouts.

## Recovered arithmetic and beam decisions

The following are static observations, not numerical equivalence results.

- Back guide PTX has terminal zeros, reverse timestep order, four transition successors plus fixed stay, half-to-float loads, symmetric clamp, float32 FTZ arithmetic and ordered five-term log-sum-exp. Clamp zero selects finite FLT_MAX as the limit. The log-sum-exp uses EX2/LG2 approximate instructions and a final FMA. See original `decoder_lib.sm_120.ptx:127-416`.
- A reconstruction of its recurrence uses `b=(4*s)%S`, `r=floor(4*s/S)`, stay `blank+beta[t+1,s]`, and four terms `clamp(score[t,4*b+r+4*k])+beta[t+1,b+k]`, k=0..3. This is a data-flow reconstruction from the inspected registers, and needs differential validation before becoming an accepted Go rule.
- Beam initializes from guide scores and scans state indices upward while admitting up to W candidates. Initial threshold ties are admitted by a not-less-than comparison. Its cut parameter is converted to natural-log space when positive; nonpositive or unordered values use a finite negative maximum sentinel. See original PTX:506-516 and 789-807.
- CRC32C-derived 32-bit keys are present. Initialization uses an inlined `crc32c<32>` over state bits; transitions update two bits through `crc32c<2>`. The observed reflected polynomial constant is `0x82f63b78`, signed -2097792136. See original PTX:815-1144 and 1308-1350.
- The candidate merge path compares these 32-bit keys for equality. It does not compare full emitted strings at that point. The inspected equal-key merge chooses the greater score's representative; equality takes the else side. A lower-score slot is invalidated with finite negative maximum. Exact collision effects across all candidate-generation and pruning paths remain unqualified. See original PTX:1583-1713.
- A subsequent independent pointer trace corrected the initial comparator interpretation. `%r675` is the left pointer and `%r673` is the right pointer; PTX:2041-2047 compares right against left and copies left on finite equality. The invocation runs only on the last frame and emits **one winner per pair**, because `r151=min(end,start+1)` bounds each output run. This is terminal top-one selection, not a fully sorted beam or a guaranteed permutation. Before it, survivor compaction is lane-major (lane, lane+32, ...), rather than ascending candidate index. The exact pointer flow, 48 partial-selection cases, 32 compaction cases and discriminating tie control are retained in `artifacts/parity-resume-review-20261024/d03-compaction-sort-review.md`. These are static/PTX-block results; native tie and full beam acceptance remain open. The earlier interpretation is preserved in that review evidence.
- Two-argument log-add used during merge has a magnitude-difference cutoff of 17. Below it the emitted PTX includes approximate exponent plus an explicit polynomial/log path and FTZ FMA sequence. Above it the correction is zero. Treating this as Go `math.Log1p(math.Exp())` would discard the inspected numerical contract. See original PTX:1607-1658 and 1662-1710.
- Posterior code has forward scans, a warp/block softmax and float32-to-half rounding. Compiler metadata names `compute_next_fwd_scan`, `compute_posts_for_timestep`, `threaded_softmax`, `warp_log_sum_exp`, and `warp_max_element`. The presence of those names does not prove their entire high-level reconstruction. See original PTX:2223-4425.
- Quality aggregation reads half posteriors, deduplicates several neighboring-state contributions, clamps a resulting probability to [0,1], applies exponent 0.4 through approximate LG2/EX2, then averages over a move-associated group. Exact neighbor deduplication and group boundary behavior remain to be fully reconstructed. Both move_pad branches are present. See original PTX:4710-5010 and 5260-5460.
- Final quality-character calculation is statically explicit. It forms error `1-p`, computes `-10*log10(error)` through approximate LG2 and float32 constants, applies `fma(q_scale,Q,q_shift)`, clamps to [1,50], adds 33.5, truncates toward zero and stores a byte. This differs from assuming generic ties-to-even. See original PTX:4970-4996 and 5439-5459. The SM120 cubin includes MUFU and contracted operations, so merely translating PTX expressions to CPU math does not establish bit equality.

## Smallest useful Go/kernel design

Go should own chunk validation, scan timestep order, beam expansion order, CRC key construction, merge representative/tie decisions, pruning and traceback. Preserve explicit fixed-size semantic records in Go, not libkoi byte scratch as the public domain model. CPU source `internal/crf` and `internal/decode` are context, not GPU truth.

The first typed proposal is deliberately below the opaque four-stage boundary.

```go
type StateID uint16
type MergeKey uint32
type ChunkExtent struct { ScoreFrameStart, Frames uint32 }
type BeamCandidate struct {
    Key MergeKey
    ScoreBits uint32
    State StateID
    Previous uint8
    Stay bool
}
type Arithmetic interface {
    // One timestep only. Go selects chunk/time and previous/next rows.
    GuideStep(GuideStepRequest) error
    ForwardStep(ForwardStepRequest) error
    PosteriorStep(PosteriorStepRequest) error
    // Elementwise arithmetic only. Go selects which equal-key pairs merge.
    LogAddPairs(LogAddRequest) error
    // Elementwise probability transform. Go owns neighbors and move groups.
    QualityTransform(QualityTransformRequest) error
}
```

These are proposed interfaces, not an implementation or established optimal split. Requests need typed tensor.Storage owners, exact declared float/half domains and dimensions, plus existing synchronous ownership/drain behavior. A timestep primitive must not run the time loop, choose beam survivors or hide traceback. LogAddPairs must not merge keys itself. Elementwise quality math must not silently choose grouping or moves. Vector scan reductions must preserve the inspected instruction/reduction order and architecture domain; a generic cuBLAS or CPU float64 replacement is not justified.

The alternative of four production wrappers around the existing host functions is rejected because beam decisions and full scans stay closed in Koi. A pure CPU translation is a useful exploratory comparator only until FTZ, approximate math and compiler contraction have discriminating evidence. Narrow GPU math operations provide a plausible way to keep those numerical details native while moving policy and scheduling into Go. This is a design inference, not a demonstrated parity result.

## Next differential oracle and remaining gates

Implement a verification-only producer that takes retained H01/S00 half score payloads and calls each of the four original functions once, retaining every stage. The producer must copy input scores first and retain the destructive posterior output separately. No model inference replay is needed.

Before execution, pin source/archive/member, score manifest and payload, scalar option bit patterns, exact descriptors, buffer capacities, prospective initialized masks, producer build/runtime objects and GPU architecture. Reject score corruption and altered dtype/stride/descriptor input before the native call. Preserve clocks/exits/raw logs and all failed attempts. The coordinator still owns runtime admission and host lock.

The first capture should cover fixed HAC C=1024/S=256 and fixed SUP C=4096/S=1024, default W=32, a genuine retained model handoff and synthetic controls. Retain aux, result words, scores, path, moves, sequence and Q after back-guide, beam, posterior and final calls. For guides use typed row bounds above; for unknown slots keep raw bytes plus prospective exclusions rather than a mismatch-derived mask.

Discriminating controls must include all-stay, no emissions, tied state seeds across initial cutoff, equal-score merge representatives, near-ties one half step apart, CRC collisions versus differing strings, cut threshold equality, beam-width boundaries, extreme/clamp-zero scores, short/final frames, mixed chunk boundaries, consecutive batches, probability/Q rounding boundaries, both move_pad settings and poisoned/error paths. Uninitialized result/history tails are not evidence. NaN/Inf/domain rejection policy must be explicitly admitted rather than guessed from sentinel behavior.

The byte partition and corrected compaction/terminal-selection trace are sufficient to begin an oracle and a pure Go beam record/CRC trace. Native validation of those rules, full CRC merge traversal and tie cases, posterior in-place packing, neighbor/group quality semantics, move_pad behavior and runtime-selected cubin remain unresolved acceptance gates. No native decoder fixture was produced in this investigation, and no numerical, basecalling or biological parity is claimed.
