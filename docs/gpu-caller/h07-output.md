# H07 fixed HAC output projection

Status is **the retained fixed N64/T32 factorised projection passed exact native
differentials, rejection controls and independent numerical review**.
The approved first slice is
the fixed factorised HAC branch at N64/T32, Cin1024/K128/Cout1024. VCS row 3 and
the two-LinearCRF branch remain H07 obligations after this slice.

## Source contract

Pinned Dorado is `8b8fc5d36a9c0baab262a743cb175b5878e38ca6`.
`FactorisedLinearLayer.cpp:30-106` reserves the temporary storage, narrows at time
row 2 for fixed chunks, zeros the down buffer, constructs three caches, calls
one `host_factorised_linear`, and exposes contiguous half NTC output.
`basecall/model/CRFModel.cpp:45-66` selects this branch only with supported Koi
FLSTM, size 1024, decomposition 128, and tanh times five.

The registered weight names are inverted relative to their semantic roles.
The authoritative archive pins remain in `internal/model/hac_weights.json`.

| Semantic role | Registered parameter | Archive | Source shape |
| --- | --- | --- | --- |
| Down | `factorised_linear.up.weight` | `11.linear.weight.tensor` | `[128,1024]` |
| Up | `factorised_linear.dn.weight` | `12.linear.weight.tensor` | `[1024,128]` |

Both have 131072 elements. The loader must retain actual tensor shape and
archive identity until role validation. Flat lengths cannot detect an exchange.
Archive deserialization uses the existing qualified loader.

Go owns half conversion, down quantization, all three source permutations,
working-memory spans, row-2 selection, down initialization, and fixed dispatch
arguments. It reuses `QuantizeFLSTMRows` and `permuteContiguous`.

| Cache | Source view | Permutation | Final storage |
| --- | --- | --- | --- |
| Down | `[8,2,2,2,2,1024]` | `[0,3,1,2,4,5]` | I8 `[128,1024]` |
| Down scale | `[8,2,2,2,2]` | `[0,3,1,2,4]` | Half `[128]` |
| Up | `[32,2,2,2,2,2,128]` | `[0,3,4,1,2,5,6]` | Half `[1024,128]` |

Down scales are half values, without reciprocal. Up is unquantized half.
Projection permutations differ from H04's FLSTM cache permutations.

## Memory and execution

The H05 result is the full forward-oriented I8 `[37,64,1024]` storage. The
kernel consumes its contiguous row-2 narrow `[32,64,1024]`. Fixed parity is 1,
scale is 5, and both bias and auxiliary output-layout pointers are null.

| Actual region | Byte interval in the H01 arena |
| --- | --- |
| Down scratch | `[0,524288)` |
| Scores and linear1 | `[524288,4718592)` |
| Gap | `[4718592,5357568)` |
| Complete H05 input | `[5357568,7782400)` |

The combined actual temp is 4718592 bytes. Reservation uses T35 from the
source's padded convolution traversal and reserves 5160960 temp bytes. Together
with its 2621440-byte current allocation, that gives the existing 7782400-byte
whole reservation. `HACOutputTempElements(n,t)` must support both geometries.
The first native harness qualification remains restricted to N64/T32. The typed
source boundary and execution plan accept positive N%64/T%32 geometry within
checked native integer limits. That source admission is not numerical
qualification of unobserved shapes. P03 already permits differing
reservation and use geometry within its checked budget.

The selected design uses one `HACOutput` object owning three uploaded caches,
an immutable plan, and invocation-local arena aliases. Run and Close serialize
cache lifetime. The caller owns exclusive use of each writable arena. All views
and native operand checks precede the first down-zero write. A private
validate/zero/kernel seam verifies this order without another execution engine.

The new typed native boundary retains six existing storage owners in CUDAWork
and calls one primitive. It validates exact dtypes, shapes, contiguous strides,
tags, device and stream ownership, pointer alignment, fixed dimensions,
parity/scale, signed conversion, and checked byte overlap. Down and output are
writable and must not overlap each other or any conflicting operand. Exact
adjacency is valid. Native return and launch status must succeed before scores
can be returned. Existing failed-drain retention remains authoritative.

## Native observation and qualification stages

The retained H01 capture already provides complete flstm6, linear1, scores, and
the post-projection backing containing all 524288 down-scratch bytes. Its
directory is:

```text
/home/jake/.local/share/opencode/worktree/e3db79/H01/artifacts/tensor/hacgpu-capture-008/stages
```

Three native cache observations, totaling 393472 bytes, were required.
The new verification-only oracle must instantiate the stock projection,
deserialize only archives 11 and 12, seed WorkingMemory with retained flstm6,
and call the unchanged projection method once. It must not run convolution,
recurrence, a full graph, or a fresh input signal. It publishes the three live
cache tensors only after down scratch and scores match H01 exactly. Candidate
payloads from any failed attempt remain retained.

Before native publication, the producer-free witness consumer must authenticate
the independently pinned H01 receipt, import genuine retained input and output,
and reject a corrupted copy. Before the producer launches, freeze and verify
the complete compiler, source/header, static/shared library, runtime, weight,
and input closure. Depfile and link evidence must cover all inputs, including
system headers and implicit compiler/linker files. A changed or missing input
invalidates the attempt; later hashes cannot repair an unrecorded build input.

After source readiness and host checks, separately admitted phases are the
cache-oracle build/capture, three-cache Go differential, and one-operation Go
projection differential with native rejection controls. All compare exact
bytes and retain complete candidate outputs. The projection differential also
compares down scratch and proves input/gap/other arena regions unchanged.

The new tensor backend build must include the pinned `0.conv.weight.tensor`
fixture for the maintained 13 CPU contracts. No extra standalone backend build
is needed. Full caller, VCS, two-LinearCRF, biological, and performance acceptance
remain separate from this initial numerical slice.

## Retained evidence

The accepted source cache capture is
`artifacts/tensor/h07-source-caches-002/capture.json`, SHA256
`39dbcd8f0e86c81ea2c78869363169475649d1e00fb2a7087777b856fde022dd`.
It calls the source projection once, matches all H01 down scratch, scores, and
remaining arena bytes, and publishes the three caches. Its five recorded
commands exit zero; 35 observed runtime objects are covered by the 44 objects
frozen before compilation. Device UUID bookends match. The bounded outer receipt
is `artifacts/h07-source-capture-20261025-002/receipt.json`.

The accepted direct source build is
`artifacts/tensor/h07-oracle-build-003/build.json`, SHA256
`f2e38cfc8adf6c03c514f59f7da18b0efa92908e48f6267314c892ebeade1482`.
It freezes 6333 inputs, compiles three source files, and retains 39 successful
metadata/compiler/link commands. Its immutable Go controller source snapshots
bind the executed producer binary independently of later consumer changes.

Failed build001 and source capture001 remain unaccepted. The first build failed
on an unrecorded transitive linker-script input. The first capture computed
the exact outputs but failed on an incomplete prospective runtime closure.
The fresh build003 prospectively included the complete metadata-derived closure
before the accepted capture002. No failed receipt was repaired with later pins.

The consumer includes a retained row-2 positive comparison and effective
wrong-row-1 and three cache corruption controls, all checked against the source
observations. Native parity 0 and scale 4 controls use otherwise valid operands;
both failed at validation with the complete arena unchanged. The consumer
authenticates its own complete Go source/embed/tool build manifest and executable
before any native loader call, and rechecks every source pin at completion.
Host preparation checks are retained in
`artifacts/h07-host-20261025-010/receipt.json`: 12 H07 tests in each CGO mode,
eight targeted model/tensor tests, and vet all passed with no skips.

The frozen consumer controller is
`artifacts/h07-host-20261025-011/controller-build.json`, SHA256
`992c8995eb9b7602b138c675e8f2e6e0797d392788f920838b1ce0b1e6855e2c`.
Its executable SHA256 is
`87314193e15694fdef579a4ff22fd18d09e66d2759a49a16e94e0eccacbe21c3`.
It retains 922 actual Go/Cgo/embed source pins and six tool hashes with no source
drift. Static consumer preflight queried the new backend and this executable,
recursively resolved their dependencies, and froze 46 runtime paths before any
genuine loader call. All 70 metadata commands succeeded. See
`artifacts/h07-consumer-preflight-20261025-001/preflight.json`.

The CPU cache differential passed in
`artifacts/tensor/h07-go-caches-001/consumer.json`, SHA256
`ff4d763e0dd701a44421bdff319967d18fa9954233ceea3c8e7857789b91abab`.
Production Go packing matched every byte of all three native caches. The row-2
positive comparison passed, and all four rehashed CPU control candidates were
rejected. Independent readback rehashed 1910 inputs, nine outputs, and 26 mapped
runtime objects. No CUDA device or projection was requested in this phase.

Projection consumer attempt001 failed at canonical lock admission before
creating its attempt directory or opening the native backend. It did not run
the native scalar controls or a projection. The failure is retained at
`artifacts/h07-projection-consumer-20261025-001/receipt.json`; it provides no
projection acceptance claim.

The fresh projection002 passed in
`artifacts/tensor/h07-go-projection-002/consumer.json`, SHA256
`9110ac5d2328673daeaf4f3fa4f4934a6949c24be299a9831c80aa8cefecf746`.
Exactly one production `HACOutput.Run` matched all 4194304 score bytes, 524288
down-scratch bytes, and the complete 7782400-byte H01 arena. The complete arena
hash is `899a00517e09d307a11fe93873bcb20d0c5fb6c065394c6071a843249bf0a342`.
Both native scalar controls rejected at validation stage 0 and retained complete
unchanged input arenas. All six controls passed their rejection checks. The
qualified device UUID matched before and after execution. Independent readback
rehashed 1910 inputs, 14 outputs, and 29 mapped runtime objects inside the frozen
46-path closure, with zero source drift. The outer scope exited zero without
timeout under 6 GiB memory, zero swap, and 200% CPU limits. Full readback is in
`artifacts/h07-projection-consumer-20261025-002/final-audit.json`.

This proof uses the current combined H06/H07 backend at
`artifacts/tensor/attempt-h06-h07-shared-build-20261025-001/libnanopore_tensor.so`,
SHA256 `f197566b0fda2be00a25e8541c0ad4fe98ed3aaafcb718b04c88ad38f3fb5b25`.
It qualifies only the retained N64/T32 fixed factorised projection. VCS row 3,
the two-LinearCRF branch, full caller, biological accuracy, and performance
remain outside this proof.

Independent review is `artifacts/parity-resume-review-20261024/h07-fixed-review.md`,
SHA256 `43018c71312d23fa986be8c2878da8b0e64e75bc7e048c4b9294d0eef97440ef`.
For the source capture, 34 mapped objects are in the 44-path prospective seed;
the additional mapped object is the newly built oracle executable, authenticated
separately by its build output hash.

A later selected-tool audit found a release-custody gap:
`artifacts/native-compiler-custody-audit-20261025/audit.md` shows that the actual
GCC15 link selected `liblto_plugin.so`, which build003 did not include in its
input census. The CGO1 controller also retained the GCC driver without a full
native-tool/header/cache trace. The observed byte comparisons remain valid for
their retained executables; complete compiler provenance is not claimed yet.
Repair must preserve the old receipts and use a separately closed build and
binary-identity attestation where possible, rather than rerunning the model to
repair metadata. Later metadata-only model API refactoring does not make the
retained Go consumer a new current-source execution.
