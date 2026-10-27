# D04 posterior and quality implementation frontier

This isolated worktree contains a source/host candidate. Go owns posterior
scheduling, traceback, quality neighbor selection, group ranges, output
placement and result-word construction. One closed numeric ABI dispatches
forward alpha, posterior rows, selected probabilities, selected group arithmetic
and selected summary arithmetic. No real backend, JIT, driver, GPU or model ran.
Canonical backend002 and the accepted D03 worktree remain unchanged.

Final host evidence is in `artifacts/d04-posterior-quality-20261025/`.
`attempt006-cgo0` passed all 17 required roots and 40 named test nodes.
`attempt005-cgo1` passed all 20 required roots and 43 named nodes against host-only
C projections, with three retained fake-compiler receipts. No named test skipped.
`attempt007-vet` passed the three scoped Go packages. The first CGO attempt's
Go-helper compilation failure, its exact source and raw diagnostics remain in
`attempt003-cgo1`; only the helper error-variable binding changed.

The 213 baseline paths have immutable content-addressed copies in `baseline/`.
The initial candidate is separately retained. The exact owned delta and all
native acceptance gaps are described by the artifact handoff. This document's
host-status prose was refreshed after those checks; compiled source and fixtures
were unchanged by that documentation edit.

## Inspected arithmetic

Use the pinned SM120 PTX identified in the D01 investigation. Whole-file line
references below are in canonical ignored `artifacts/decoder-static-20261025-001/decoder_lib.sm_120.ptx`.

Forward alpha starts at positive zero for every state. For state `s`, the five
terms are blank plus previous alpha[s], followed in k=0..3 order by previous
alpha[floor(s/4)+k*S/4] plus clamped half score[4*s+k]. The max, ordered
exponential sum, approximate LG2 and FMA log-normalizer match the reverse guide
arithmetic. Lines2440–2550 establish those operands and arithmetic. The new
forward header reuses the existing guide's arithmetic with changed gathers;
it needs independent operand/instruction review.

For t=0..T-1, compute alpha[t+1] from original score frame t BEFORE writing
posterior row t. Posterior row t uses alpha[t] plus backward guide[t], not the
new alpha. Swap alpha owners only after that row. A final posterior row uses
alpha[T] plus guide[T]. These are source shared-pointer swaps and barriers,
not an inference from final tensors (lines2380–2440,2800–2840,3570 onward).

The Go operation uses a distinct working parent. It reads forward frames from
immutable original scores. Before frame t, original in-place posterior stores
end at t*S while frame t starts at 4*t*S, so these reads preserve the source
operands. Working posterior prefixes and each next-alpha row start poisoned.
Alpha downloads and the final working parent are checked before publication.

Posteriors overwrite the FIRST (T+1)*S half elements of each chunk's original
score allocation. Chunk stride remains 4*T*S half elements. The remaining
tail is unchanged original scores. This is state posterior packing, not a
4*S transition posterior layout. Preserve complete parent bytes so D01 can
compare the destructive working-score allocation. Go must own time order and
row offsets; each native operation handles one chosen row/frame only.

The proposed posterior header computes alpha+beta with FTZ, stores a shared
row, and uses the first warp for the reduction. Each lane visits l,l+32,...
in ascending order. Max starts at positive zero, then uses XOR butterfly
16,8,4,2,1. Exponential sums also start at zero and use that butterfly order.
The normalizer is `fma(lg2.approx(sum), 0x3f317218, max)`. Each output is
`half_rn(ex2.approx((value-normalizer)*0x3fb8aa3b))` with FTZ. Do NOT substitute
exp/sum division; the inspected code recomputes exponentials from the
log-normalizer (lines2900–3550). Proposed loops replace source unrolling but
must preserve numerical instruction order for both S256 and S1024.

## Go quality policy and numeric boundary

`PlanQuality` selects ordered distinct neighbors and groups by moves; see its
copied documentation and canonical tests. It must be independently checked
against both compact and padded source branches before native qualification.
Go gathers only path-selected half posterior values and explicit ordered
neighbor counts, never native future intermediates. Numeric operations
convert the selected half values, sum neighbors then center, clamp [0,1], and
apply `ex2.approx(lg2.approx(p)*0x3ecccccd)` (lines4944–4958,5405–5420).

Go selects the exact group ranges. The selected-group numeric operation performs
sequential float32 additions from positive zero and the source approximate
division by group length. Error is 1-average. Q uses LG2 times0xbe9a209b,
times10, FMA(qscale,Q,qshift), min50/max1, plus33.5 and truncation toward zero.
Do not use CPU libm, generic round, a different reduction or float64.
Source per-group errors are accumulated in order for result mean error;
the source result words retain those raw bits. Negative terminal beam scores
select the source A/! fallback before reading history or posterior values. Padded mode
writes at group start but resets compact sequence count to zero. Do not
invent a corrected count or quietly treat padded output as compact output.

## Program and ownership

Checked Storage/Stream owners, byte-interval rejection and CUDAWork retain the
numeric operands through completion and failure. `nt_cuda_decoder_post_numbers`
is the single added mandatory symbol. Its private Go operation enum admits five
operations. Group jobs return separate F32 error and U8 quality outputs. Summary
jobs return raw F32 result bits and a U8 write marker, preserving legitimate
nonfinite padded means while rejecting a success without writes.

The guide program links guide, forward, posterior and quality PTX entries. The
complete retained cubin is the same byte array passed to `cuModuleLoadData`.
`DecoderPosteriorProgram(s)` checks a live same-owner stream.
`NewGPUQualityNumbers` requires the expected linked-program digest before
numeric dispatch. Initialization and close retain failed native owners.

Do not wire any of these changes into canonical before coordinator's backend002
H08/D02 wave. The D03 and SUP worktrees have independent pending ABI deltas;
future integration is an explicit allowlist merge, never a whole-worktree copy.

Acceptance needs meaningful host geometry/overlap/ownership/marshalling/error
controls, genuine compile/JIT under the shared runtime guard, and D01 retained
native working-score/path/move/sequence/quality comparisons plus discriminating
controls. No Go test or static instruction comparison closes native parity.
