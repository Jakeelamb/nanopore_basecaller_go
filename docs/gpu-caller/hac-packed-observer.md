# Packed HAC source observer

Status: **native observer source only; Go host controls passed**. This unit prepares pinned
fixtures and generates copies of genuine Dorado source with observation calls.
It supplies no native executable, build/runtime admission, or GPU/biology proof.

The command source in `verification/hacgpu/packed/cmd` exposes exactly two modes:

- `generate --source-root <canonical-Dorado-root> --out <fresh-directory>`
- `prepare --request <canonical-request.json> --request-sha <independent-SHA256>
  --request-bytes <independent-size> --out <fresh-directory>`

No build/run mode, admission flag, lock, device query, library load, or child
process exists. The Go package and CLI source compile in the host tests; the
native observer has no executable or runtime admission.

`PackedHACRequest/v1` declares independent model/config/weight/dataset/CPU21 and
generated-source manifest pins, plus unchanged WorkingMemory, AuxiliaryData
header, CudaModelRunner and CudaCaller source pins. Every pin names canonical
regular bytes with an independent SHA256 and extent. The prepared manifest
retains the request, all authenticated inputs and output F16 pin. Native closure
is explicitly unavailable (`null`). Model/config/weight/dataset/provenance
manifests are custody inputs; their semantic authenticity is a later producer
review requirement. Pin matching alone does not establish their claims.

Preparation accepts only the first case: physical N64, configured/max chunk12288,
C_in1/C1024/K128/layers7, stride_in=stride_out=granularity6, and ordered rounded
lengths `[768,1536,12288]`. Declared development/validation membership, read IDs,
per-read calibration/mean/stdev/trim and exact parent/raw/normalized offsets are
required. Unknown calibration is rejected. HG005 labels are rejected; this text
check cannot authenticate dataset membership.

Request and generation JSON require every exact typed field once. Alternate
casing, duplicate aliases, missing declarations and ordinary nulls reject at
every nested object. Explicit zero fill/trim/offsets remain valid. Calibration
pointers retain unavailable values for validation to reject; `native_closure`
retains explicit null semantics. The preparer expands generation inputs before
one complete pin collection: identical-path identical-pin reuse is permitted,
conflicting pins and distinct-path hard links reject across the whole closure.
Canonical paths, per-file/aggregate byte bounds and final input rehash remain.

Input comes from explicit pinned, already normalized little-endian F16 read
fixtures. The code checks finite halves and preserves normalization provenance;
it does not decode POD5, read model tensors, or recompute normalization. Each
member's raw-length slice receives source repeat-prefix stride rounding, then
members concatenate without gaps. Capacity padding has a declared finite fill.
There are 14592 active input samples, 2432 scores and T_lstm2050. Assertions,
workers, required pinned CPU workspace, distinct observation fills and guards
are declared recipes, not executed controls.

Source upload order is OutLayout, ForwardEncoding, InLayout, BackwardEncoding.
Initialized words are `[131200,131200,2432,131264]`, capacities
`[131264,131200,2432,131264]`, and CPU workspace offsets
`[267340,2438,133638,136076]`. The initialized workspace prefix is 398540 words
within a 787584-word reservation. OutLayout's final 64 words remain unwritten.
These CPU21-bound declarations do not describe opaque kernel reads or mutations.

The generator gates five whole-file SHAs, requires unique method-scoped literal
anchors, records every insertion and reverses observed bytes exactly to original
bytes. It emits untouched copies alongside observed CRFModel, ConvStack,
FLSTMStack, FactorisedLinearLayer and AuxiliaryData, plus the embedded typed
`observer.h` and `generated.json`. It keeps original graph arithmetic, source
conditionals, seven-layer loop, encoded operands and projection row3 unchanged.

Hooks cover graph reservation/allocation/input/stacks; small convolution and
repair; CUTLASS padding/scatter/fast/fallback/repair; recurrent sentinel and
scratch writes, each encoded call and post-call orientation; projection row3,
down zeroing, gather/call/compact output; and Aux construction/preprocess/uploads.
Typed references expose full backing/current/views, caches and host/device Aux
operands. The header requires stable allocation identities, event/stream/device
metadata, immediate immutable snapshots, known source writes and separately
unknown read sets. There are no hook implementations or no-op definitions.
Fast convolution's post-call hook passes the actual selected
`use_f32_accum ? KOI_F32 : KOI_F16` operand in a typed `AccumulatorMode` record.
The event contract retains that value; other convolution phases explicitly pass
an unavailable accumulator mode rather than inferring one from tensor dtype.
Fallback entry passes the existing `try_fast_conv` as `has_status`: skipping the
fast call leaves its status unavailable; an attempted fast call retains its
actual `res`. The following fallback-result hook reports the actual returned
status. A value paired with `has_status=false` is inapplicable, not a native result.

Tests are authored for actual-source SHA/anchor uniqueness/reversal/corruption,
manifest tampering, pin changes, schema/order/split/shape/extents, finite input,
repeat-prefix packing, fresh paths, symlink/hard-link aliases and capacity spans.
Synthetic validator fixtures are labelled as such and provide no native evidence.
The focused Go suite passes 13 root tests and 56 subcases, including
actual-source SHA/anchor/reversal and rejection controls. The boundary
corrections cover exact JSON fields,
missing/null versus explicit-zero declarations, complete-closure hard links,
same-path pin reuse/conflicts and the selected accumulator operand/header/event
contract. Both independent same-model reviewers cleared the final correction.

The combined source branch passes `CGO_ENABLED=1 go test -p 2 -json ./...`:
900 root tests and 2,087 subcases, with 93 opt-ins skipped. Maintained parity
passes its CPU-package/native-algorithm scope. A separately built Go0 CLI has
806 pinned source/tool inputs. Its actual `generate` command emits all five
source units, each independently reversed exactly to the original. Existing
output and modified-source controls reject without changing the accepted files.
Raw build, command and failed-attempt receipts remain outside Git.

The `prepare` mode still needs a genuine packed-input fixture. The native
observer has no implementation, executable or runtime admission; these Go and
source-generation checks do not qualify a completed caller port or biology.

Next closed producer step: implement every header hook and a fresh-model,
fresh-Aux/workspace/KoiThreads, real guarded-stream harness calling genuine
`run_koi(input,&aux)` once. Close compile/link/archive-member selection and all
runtime dependencies, then bind phase gates, exact argv/output/resource contract
and independent target pins. Supply genuine development/validation fixtures and
authenticated model/CPU21 evidence. Retain positive and rejection attempts,
source snapshots, full backing bytes and upload/interval revisions. GPU scatter,
repair, encoded recurrence, row3 gather, scores and opaque Koi read/mutation
semantics remain unqualified. No build/run command is provided here.
