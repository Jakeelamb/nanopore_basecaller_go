# Prospective Go GPU acceptance protocol

The A00 plan is `provenance/gpu-acceptance-plan.json`. Its state is
`prospective-unarmed`. The checker validates that protocol, not caller acceptance.
No candidate experiment, native run, biological scorer, or timing study is armed.
The integrated host pass ran the seven checker test functions and the retained
pin command successfully. It did not read signal assets, run producers, or arm the
protocol. Native and candidate execution still require separate admission. Host
evidence is retained in `artifacts/gpu-plan/host-check-20261023T183053Z/`.

The public Go interface is `Decode(io.Reader) (Plan, error)`, `Check(Plan) error`,
and `VerifyPins(Plan) error` in `verification/gpuprotocol`. `Plan` contains pinned
retained manifests, two HG002 cohorts, endpoint rows, fixed policies, and deferred
gates. `Check` rejects endpoint or policy changes instead of accepting arbitrary
nonempty descriptions. It also requires the seven exact path/hash/size identities
frozen independently in the checker source. A supplied plan cannot authorize
replacement controls by recomputing their hashes. `Decode` requires exact field
spelling, every declared field and a nonnull value at each level. It rejects duplicate
keys, including keys expressed through equivalent JSON escapes, before typed decoding. `VerifyPins` reads only directly declared manifests and
membership files up to 8 MiB each. It never traverses asset pins or runs producers.
It establishes custody of retained manifests, not validity of all referenced data.

The retained stock plan binds parent UUIDs, original sample counts, POD5 assets,
model trees, executable pins, and requested inputs. Development is HG002 PAW70337
with 2,608 parents. Validation is HG002 PAW71238 with 2,027 parents. Membership
files pin byte order separately. Their hashes were recorded during source authoring.
Runtime ingestion remains unexecuted. The accuracy input manifest binds references,
GIAB v4.2.1 truth, normalized truth, masks, exclusions, and scorer configuration.
Its historical HG005 references are custody context only. They grant no selection,
execution, or scoring authority for HG005. The source build manifest pins unchanged
Dorado commit `8b8fc5d36a9c0baab262a743cb175b5878e38ca6` and its CUDA13 stack.
The official CUDA12 distribution remains a separate oracle.

Every endpoint in the JSON has a separate decision. Engineering, source tensor
bytes, official tensor bytes, canonical records, headers, raw FASTQ/BAM, accuracy
artifacts, biological accuracy, ALT retention, yield, native process timing, and
outer workflow timing cannot substitute for one another. Full allocation capture
includes scratch storage. Comparison includes source-declared initialized/live
ranges, guards and sentinels. Exclusions need a source justification before clocks
or capture. Observed mismatches cannot define a mask or tolerance.

The original source qualifier has eight equal numerical scores and six literal
alternate-alignment row-order failures. Those remain historical FAILs. Sorting the
tables or comparing row multisets explains a difference but cannot pass the frozen
literal artifact endpoint. The retained full comparison is pinned separately.

The biological endpoint retains the fixed reference-identical 100 bp and 500 bp
domains. It does not establish personal-haplotype or variant-calling accuracy.
ALT retention is separate. Any loss of a control-supported ALT observation fails
the prospective guard. Paired uncertainty keeps all children with their original
parent and reports zero-denominator resamples. Reference-only agreement cannot
replace ALT evidence. A02 must qualify the actual ALT observation and common-support
consumer on genuine data before any biological acceptance claim.

The complete parent universe and original signal samples supply yield denominators.
Missing parents and parents with zero eligible columns remain visible. Output
lineage uses `pi` where present and QNAME otherwise. Report child intervals,
represented/evaluable sets, emitted bases, clipping, unmapped output and gained/lost
sets. No additional absent parent or reduction of evaluable parents or eligible
query exposure passes the default guard.

Timing stays unarmed until A03 freezes a concrete arm/block/session schedule,
source and consumer commits, required test names, model/device precision dispatch,
batch policy, cache state, affinity, profile, cooldown, mapper and output policy.
A numerical speed budget requires registration before clocks. This plan invents
none. Fresh paired equal-work complete-process runs retain unsuccessful attempts
and pair correctness, biological results and yield with every timed output.
Historical stock clocks are descriptive only. The outer basecall/map/sort endpoint
needs its own qualified outer clock. Summed audit stages cannot replace it.

After explicit coordinator release, the maintained command is:

```sh
go run ./verification/gpuprotocol/cmd -plan provenance/gpu-acceptance-plan.json
go run ./verification/gpuprotocol/cmd -plan provenance/gpu-acceptance-plan.json -verify-retained
go test ./verification/gpuprotocol/...
go test ./...
```

The first command checks source policy. The second additionally verifies retained
manifest bytes, strict lowercase hexadecimal UUID syntax, membership counts,
within-cohort uniqueness and development/validation disjointness. Both print `PROTOCOL_VALID` on success and
leave all runtime gates pending. Invalid protocols or changed pins exit 2. Neither
command acquires locks, initializes a GPU, invokes native tools, or launches scorers.
The repository's maintained parity command remains separately required for runtime
qualification after admission.

Required local test names are `TestFrozenProtocol`, `TestRejectWeakenedProtocol`,
`TestRejectInvalidJSON`, `TestGenericPinReaderAndCorruption`,
`TestRejectReplacementWithRecomputedPin`, `TestStrictUUIDAndDisjointMemberships`,
and `TestRejectAmbiguousAndIncompleteJSONAtEveryLevel`. Generic pin-reader tests
use synthetic files and cannot authorize a production plan or establish genuine
control ingestion. Their positive cohorts are disjoint. Replacement tests require
production rejection even with valid recomputed hashes. Strict JSON controls mutate
objects and values at every protocol level. Genuine retained pin verification ran
on the declared manifests. An effective corruption rejection against retained
bytes, independent exact-delta review, complete-input runs, native parity,
biological/ALT scoring, and matched measurements remain deferred. No required test
may skip or pass with zero matches.
Every later executed check needs argv, UTC clocks, exit status and raw log/payload
hashes in an immutable attempt. Failed attempts remain retained.

This protocol preserves the original `research/prospective-experiment.md`,
`research/controlled-baseline-results.md`, and `research/source-qualification-results.md`
read-only. Candidate selection uses development only. A validation revision requires
a new study and confirmation set. No endpoint switches after outcomes are visible.
