# D03 full Go beam design

This is an implementation design grounded in the retained SM120 PTX, not a
completed port or native numerical acceptance. Go owns beam state and decisions.
GPU code supplies explicitly selected arithmetic. The actual Koi runtime image,
full staged beam fixture and collision scheduling remain unqualified.

The inspected checkout is `3c88e3442940f3b69f5a00fbe762c602b7527199` plus the
retained dirty source. Input hashes and fresh bounded controls are in
`artifacts/d03-full-beam-design-20261025-001/`. The original
`decoder_lib.sm_120.ptx` SHA-256 is
`7e46f59645025bca2cf7a519c9f0334b5435942ca9006d0dec5faa6d3e63b01a`;
`beam_search_step.ptx` is its original lines 417–2222, SHA-256
`27be81ea0dc242864e067967ec3e89a0b7aead3377db5bc0ac278e30be555051`.
Line numbers below refer to that original PTX. Earlier CRC, lane compaction,
terminal selection and successful traceback proofs remain in
`artifacts/parity-resume-review-20261024/d03-go-policy-review.md` and
`d03-compaction-sort-review.md`; this design reuses them.

## Runnable ownership and interface

Start with fixed N/T batches, W32 and the existing `GuidePlan` geometry:
S256/N multiple64 or S1024/N multiple32, T positive, checked extents. First
qualification uses the producer's actual scalar bits, initially clamp0, blank2
and cut100. Retained HAC/SUP model scores lie in [-5,5]. Reject nonfinite score,
guide and scalar inputs; do not treat a CUDA-success result containing poisoned
scratch as a successful beam. Packed/VCS descriptors and other beam widths are
separate work.

Use host bit arrays for the first implementation. An adapter downloads immutable
F16 scores and F32 guides once; the existing `BackGuides` operation still computes
guides with its Go reverse-time loop. Go gathers arithmetic rows and uploads only
their operands. This is a correctness implementation, with no throughput claim.
The two considered shapes are host Go records with batched arithmetic, and
device-resident records with Go scheduling and repeated decision downloads. The
former has fewer ownership seams and exposes every decision for replay. Choose
it first. A native one-frame beam routine would hide the work being ported; CPU
`math.Log/Exp` cannot replace the inspected GPU approximation.

The following signatures are the intended small internal API, using the existing
`BeamCandidate`, `BeamHistoryRecord`, `GuidePlan` and `TraceBeamHistory` types:

```go
type BeamInput struct {
    ScoresNTC []uint16 // F16 bits, [N,T,4*S], immutable
    GuidesTNS []uint32 // F32 bits, [T+1,N,S], immutable
}
type ChunkBeamHistory struct {
    Rows [][]BeamHistoryRecord // rows 0..T; each length is its initialized count
}
type BeamBatch struct {
    History []ChunkBeamHistory
    FinalScoreBits []uint32 // unguided score of terminal slot zero, per chunk
}
type BeamPlan struct { /* private GuidePlan, W=32, cut bits and checked extents */ }
func PlanBeam(guide GuidePlan, cut float32) (BeamPlan, error)
func BeamSearch(ctx context.Context, p BeamPlan, in BeamInput,
    numbers BeamNumbers) (BeamBatch, error)

type F32Pair struct { First, Second uint32 }
type ExpandRow struct {
    Previous uint32
    Transition [4]uint16 // Go gathered source half scores
    Guide [5]uint32     // emit0..3, then stay
}
type ExpandResult struct { Scores [5]uint32; OrderedMax uint32 }
type BeamNumbers interface {
    LogCuts(context.Context, []uint32) ([]uint32, error)
    Expand5(context.Context, []ExpandRow, uint32, uint32) ([]ExpandResult, error)
    LogAddPairs(context.Context, []F32Pair) ([]uint32, error) // stay, emission
    SubPairs(context.Context, []F32Pair) ([]uint32, error)    // first-second
    Midpoints(context.Context, []F32Pair) ([]uint32, error)
    MaxPairs(context.Context, []F32Pair) ([]uint32, error)
}
```

`BeamSearch` returns an empty `BeamBatch` on every error or cancellation, including
late failures. It owns all output rows and scratch; borrowed inputs never change.
One owner advances each chunk, and chunk results are published only after the
whole batch succeeds. Check cancellation before/after every arithmetic call and
at frame boundaries. A caller may adapt `History[c].Rows[1:]` to the existing
traceback function; its result is still the temporary full-state path before
quality overwrites path entries with base IDs.

## Exact frame algorithm

1. **Seed threshold and admission (613–1168).** Copy guide row zero to float
   scratch. The initial ping-pong merge emits up to W values per pair, retaining
   descending top-W prefixes. Its right pointer is `%r624`, left `%r626`;
   `leu.ftz(right,left)` selects left. The threshold address is
   `%r5-56*W-4 = B+4*(W-1)`, the Wth score. This differs from the terminal
   top-one invocation. Scan state IDs upward, admit `!(guide <FTZ threshold)`
   until W. Store score **+0**, raw initial CRC, state ID, predecessor0/stay0;
   store history row0. Never seed in score-rank order or seed scores with guides.
   For finite inputs a host stable top-W merge over FTZ comparison keys yields
   the same threshold prefix; preserve original score bits. NaN source threshold
   exits with result-score -1; the proposed input boundary rejects it explicitly.

2. **Expand (1235–1564).** At frame t, gather guide row t+1. For active slot i,
   state s, set `u=(4*s)&(S-1)`, `q=(4*s)>>log2(S)`. Emission a=0..3 uses next
   state `u+a` and score channel `4*u+q+4*a`. In admitted S256/S1024 domains all
   source signed-16 state/index conversions stay nonnegative and in range.
   Store emissions at **4*i+a** and stay at **4*B+i**, making a front of 5*B.
   Emission key extends two base bits; stay keeps the key/state. Predecessor=i;
   stay bytes are respectively0/1. Go computes all indices and record fields.
   Arithmetic is exactly:

   ```text
   limit = (clamp ==FTZ 0) ? FLT_MAX : clamp
   transition = minFTZ(limit, maxFTZ(-limit, halfToF32(raw)))
   emit[a] = addFTZ(nextGuide[u+a], addFTZ(previous, transition))
   stay    = addFTZ(addFTZ(blank, previous), nextGuide[s])
   laneMax = maxFTZ(...maxFTZ(maxFTZ(-FLT_MAX,emit0),emit1)...,emit3,stay)
   ```

3. **Merge (1566–1724).** Stay i scans j=0..B-1 at emission index
   `4*j+(signed16(stayState)%4)`; the admitted domain reduces the remainder to
   `stayState&3`. Match **CRC key only**, regardless of full string/state. Read
   both current scores anew at every match. If `stay <=FTZ emission`, emission
   keeps its representative, including finite equality; otherwise stay keeps its
   representative. Call `LogAddPairs(stay,emission)`, store the combined score in
   that winner, and store **0xff7fffff** in the losing score only. Neither key nor
   state/predecessor/stay metadata is cleared. Update that stay lane's existing
   laneMax with the combined score. **Continue scanning after a match**, including
   matches against already invalidated scores. A map keyed by CRC or an early
   break implements different behavior. Preserve each lane's ascending-j order.

4. **Construct cutoff (1729–1884).** Reduce laneMax through snapshot butterfly
   partners16,8,4,2,1 with ordered maxFTZ. Preserve lane maxima from expansion and
   merge, rather than silently recomputing a maximum from final front records.
   For positive cut, logCut is `mulFTZ(lg2.approx.ftz(cut),0x3f317218)`;
   nonpositive/unordered cut instead selects finite -FLT_MAX. The first admitted
   plan uses cut100. Initial cutoff is `subFTZ(max,logCut)`. Count every front
   slot with `score >=FTZ cutoff`, including finite invalidation sentinels.
   Count<=W bypasses refinement, even below the lower target. Otherwise use this
   source control, where `countAt` and every branch are Go operations:

   ```text
   lower=cutoff; upper=max; tooMany=true
   for step=1..9:
       savedLower = tooMany ? cutoff : lower
       savedUpper = tooMany ? upper : cutoff
       partner    = tooMany ? upper : lower
       cutoff = div.approx.ftz(addFTZ(cutoff,partner),2.0)
       count = countAt(cutoff)
       tooMany = count>32
       again = (tooMany || count<25) && step<9  // floor(8*W/10)=25
       lower=savedLower; upper=savedUpper
       if !again:
           if step==9: cutoff=savedUpper; count=countAt(cutoff)
           break
   ```

   The ninth-midpoint exit **always** falls back to the saved upper bound, even
   when the ninth count lies in [25,32]. Do not replace it with a generic search
   for exactly32 survivors. Nonpositive cut, 0<cut<1 and extinction behavior are
   not qualified successful-domain cases; the initial plan need not accept them.

5. **Compact and select terminal (1887–2113).** Reuse the proved inclusive,
   lane-major `CompactBeam`, capped at32; selected count is `min(count,W)`.
   Only t=T-1 runs terminal top-one selection (`r82=T-1`, `r643=t`). The existing
   `TerminalBeam` gives the correct head record but not the complete raw final
   history row: the source ping-pongs active A and expanded B, writing only one
   winner per pair. Untouched B slots retain expanded records; the final array
   need not be a permutation. Preserve both logical buffers and translate that
   already reviewed scratch mutation when producing all initialized final-row
   slots. A full stable sort would alter nonhead history. All first-nextCount
   B slots are initialized because nextCount<=5*previousCount.

6. **Subtract guides and write history (2114–2178).** For each retained slot,
   execute `subFTZ(candidateScore,guide[t+1,state])` and store this unguided score
   for the next frame. Store state16/predecessor8/stay8 in history row t+1 at
   slot k. Even terminal guide-zero subtraction must run, since FTZ/sign-zero
   behavior is observable. At completion result word+12 is active slot0's
   unguided score; result words0/4 are descriptor start/length. Other result words
   belong to later stages and must not be fabricated as initialized beam output.

Guides have Go layout `[T+1,N,S]`; source capture guides/history are chunk-major
`[N,T+1,S]` and `[N,T+1,W]`. Convert explicitly. History wire records are four
bytes; fronts are twelve bytes. Neither Go struct layout is a wire ABI. Never
compare capacity tails as initialized history or treat opaque terminal scratch
as a sorted beam.

## Arithmetic boundary, one backend revision

Implement the six methods together, behind one private checked operation enum
and one authenticated `DecoderBeamProgram`/numeric-dispatch backend extension.
Do not create a new exported C entry point and compiler grant for each primitive.
The linked cubin returned for custody must be the actual bytes passed to
`cuModuleLoadData`, following the existing guide-program ownership/lifetime model.
No Koi host function, hidden time loop, key matching, record construction, cutoff
count, compaction or history production belongs in this boundary.

| Primitive | Packed device operands and result | Required source instruction shape |
| --- | --- | --- |
| LogCuts | F32[K,1] -> F32[K,1] | Positive-cut LG2 approximate then multiply by ln2, 506–516 |
| Expand5 | prev F32[K,1], transitions F16[K,4], guides F32[K,5] -> F32[K,6] | emit0..3, stay, ordered lane max; half conversion, clamp and additions above |
| LogAddPairs | F32[K,2] ordered stay/emission -> F32[K,1] | abs(subFTZ), maxFTZ; difference>=17 gives correction+0, otherwise EX2 approximate and exact emitted log1p polynomial/FMA/bit operations, 1605–1706 |
| SubPairs | F32[K,2] -> F32[K,1] | `sub.ftz.f32`, operand order preserved |
| Midpoints | F32[K,2] -> F32[K,1] | `add.ftz` then `div.approx.ftz` by2; no CPU average substitution |
| MaxPairs | F32[K,2] -> F32[K,1] | ordered `max.ftz`, preserving source raw-bit behavior |

The log-add transcription must retain the `add.rz.ftz(z,1)` step, integer
exponent extraction, all FMA constants/rounding, the special-value tail and final
addition. A generic log1p implementation is insufficient. Keep exact source
opcode ranges as the implementation review checklist, not an algebraic rewrite.

For every ordinary operation, K is shared by operands, positive and bounded by
the checked batch workspace; zero rows skip dispatch. Views are rank-two,
unscaled, contiguous row-major NC with exactly the stated widths and dtypes.
All operands belong to the same live device and admitted stream. Output storage
is independently owned and byte-disjoint from all input intervals; read/read
aliasing is allowed. Validate overflow, extents, ownership, scalar bits and the
authenticated program before launch. Keep all work operands retained through
completion/drain, including launch/sync failure; follow existing poisoned-close,
module/linker retention and retry-owner rules. Returned extent must match K and
output bits must satisfy the admitted finite domain before Go uses them.

Host-bit-exact operations are integer state/index calculation, raw CRC, record
copy/packing, finite FTZ predicates, key equality, representative-role selection,
count/control loops, lane compaction and history/trace selection. Initially keep
even ordinary float arithmetic and max-result bits in the numeric boundary;
host float64 formulas do not establish SM120 approximate equivalence.

## Collision scheduling question, with a concrete future control

There is no barrier/atomic inside the merge loop. A sufficient condition for the
ordinary Go merge is that each mutable emission score has at most one stay-lane
owner. This is directly computable from the Go-selected pair graph before
arithmetic. With disjoint lane-owned score sets, lanes commute; preserve each
lane's j order and batch independent pairs. The weaker claim that retained
`(CRC,lastBase)` uniqueness is automatically preserved is not established.
Even unique `(CRC,lastBase)` parents can share CRC across different last bases;
extending both by the same base produces duplicate emission identities, and
equal-key/different-last-base stays can survive beside them. Invalidation only
changes scores, so it is not a metadata uniqueness filter.

`trace_controls.py` constructs an exact **beam-stage-only** witness from source
initialization: finite zero guides, +256/-1024 scores for17 frames, then +2
emit/stay, blank2/cut100. The sequences
`00000000000000000` and `10332131032311001` share CRC after17 emissions but have
different last bases. Row18 has four survivors, including duplicate
`(CRC,lastBase)`; frame19 has two stay lanes writing the same emission targets.
Those scores/guides do not prove HAC/SUP production reachability.

`bounded_candidate.py` then uses exact-half scores in [-5,5], selected target
edges+5 for17 frames, base0 continuation+2 afterward, all other transitions-5,
blank2/cut100/clamp0. It computes guides from the recovered recurrence using
**unqualified CPU float64/libm**, solely to propose a native control. The first
trial retains4..8 beams, never invokes cutoff refinement, and at frame19 finds
two stay owners for emission slots12/20. Row18's duplicate records have key
1966928604, lastbase0, states0/4 and score87 in that exploratory model. Full score
and exploratory guide arrays are retained in `bounded_candidate.result.json`.
This is stronger fixture preparation, not proof of the pinned GPU computation.

Do not invent a deterministic Go lane serialization for overlapping targets.
Implement the full ordinary Go algorithm now, with explicit detection of that
case; until qualified handling exists, return a typed collision-schedule error
with no partial batch. This is a temporary acceptance boundary, not a claim that
the caller cannot be ported or that collisions are invalid source inputs.

Prepare one source-specific extension for the same backend revision if required:
preselected warp-pair arithmetic takes the Go-built `[32,32]` match mask, B and
the numerical score/lane-max arrays. Go supplies pair bindings, ascending-j
traversal, tie-to-emission assignment roles and finite invalidation bits. The
primitive evaluates that selected conditional arithmetic with the source's
32-lane shared-memory schedule; it receives **no keys, states, records, guides,
cut parameters or history**. Conditional numeric stores may overlap only as
explicitly declared by that pair graph, unlike ordinary disjoint-row methods.
This is an option to resolve the schedule question, not an already established
ABI or accepted implementation. It must not rediscover matches or decide a
different merge policy. Native collision controls must determine whether its
store behavior can reproduce the actual selected original image; recompiling
similar PTX alone does not establish that. If original repeated results vary,
record that fact and narrow the comparison contract explicitly before closure.

## Verification handoff

Fresh commands, all read-only/nonblocking canonical-lock, two CPUs and 3 GiB:

```sh
uv run --no-project --offline python artifacts/d03-full-beam-design-20261025-001/initial_threshold.py
uv run --no-project --offline python artifacts/d03-full-beam-design-20261025-001/trace_controls.py
uv run --no-project --offline python artifacts/d03-full-beam-design-20261025-001/bounded_candidate.py
```

Observed: eight actual initial-threshold PTX-block cases pass for S256/S1024,
including FTZ ties; top-one threshold and score-ranked seed negatives differ.
The symbolic cutoff control detects the erroneous ninth-convergence shortcut.
The exact out-of-range witness refutes the general uniqueness assertion for its
stated beam inputs. The bounded-score exploratory fixture reaches the same kind
of target overlap, with the numerical limitations above. Logs, receipts, input
pins and negative observations live beside these scripts. No implementation,
compiler, loader, device, GPU or maintained parity command ran for this design.

Next implement the private Go engine and batched numeric marshalling together;
reuse accepted partial units. Retain per-frame active/front bits, selected pair
bindings, lane maxima, cutoff/count trajectory, retained indices and history
rows for differential diagnosis. Before D03 completion require pinned native
stage outputs, authenticated selected runtime image and numerical-program bytes,
rejections and all-stay/tie/near-tie/merge/collision/cut controls, genuine HAC/SUP
differential evidence, and failure/cancellation/immutable-input checks. Expansion,
merge, numerical pruning, subtraction and full history remain unimplemented at
this handoff; posterior and quality remain separate open port units.
