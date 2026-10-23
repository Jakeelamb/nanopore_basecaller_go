Archived initial review. Its subread timestamp finding is resolved by the
[independent clock-fix review](independent-review-subread-clock.md). The original
verdict and raw evidence remain under `artifacts/_independent-child-interval-initial/`.

# Verdict

**FAIL for the documented subread domain. PASS for standalone interval trim and filter.**

The maintained checks pass for both units. Independent boundary probing finds one admitted subread input whose timestamp differs from the pinned native oracle. No implementation or acceptance check was changed. All review outputs and temporary source copies are under `/tmp/nanopore-go-child-interval-review`.

## Reviewed artifact

Working directory is `/home/jake/Projects/nanopore_basecaller_go`, branch `port/go-caller`. The initial HEAD was `4d3e546ce57f342e95f970b1b593c17c0e7d6e12`. Another owner advanced HEAD to `574410a1a97fe4f98780decc8545d1d193742079` during review. The reviewed leaf files were untracked working-tree additions, so a commit ID alone does not identify them. The exact artifact is the 50-file SHA256 freeze in [before.sha256](before.sha256), including owned paths, shared readtrim/decode dependencies, `go.mod`, and all native source pins. [after.sha256](after.sha256) matched it byte for byte at the freeze check. Later changes require another delta review.

The owned review scope is `internal/subread/**`, `verification/subread/**`, `docs/subread.md`, `internal/readtrim/interval*.go`, `verification/intervaltrim/**`, and `docs/interval-trim.md`. Existing `internal/readtrim/mux*` and `internal/decode/**` were inspected and pinned as dependencies. Caller composition, detection, splitting decisions, and default integration were excluded.

Native source HEAD is `8b8fc5d36a9c0baab262a743cb175b5878e38ca6`. Its working tree was clean. Source hashes match the declared oracle pins and retained copied native source files.

The core Go implementation pins are:

| File | SHA256 |
|---|---|
| `internal/subread/subread.go` | `526f359915b66554e8bc52c3037f2a34240fdc4f60d00cc123e89364ac22f8dc` |
| `internal/readtrim/interval.go` | `23c0df91f5081742fd270eaec28594d2333c2c5cfdbf42b69741cd7fb4cd6a6f` |

## Finding F1, admitted timestamps exceed the native conversion domain

Priority is P2. `internal/subread/subread.go:197` admits absolute timestamps through `253402300799999` milliseconds, and `child` at line 134 formats them with Go `time.UnixMilli`. `docs/subread.md:58` documents years 1970 through 9999. Pinned `dorado/utils/time_utils.cpp:26` instead calls `std::chrono::system_clock::from_time_t(ms / 1000)` and adds the remaining milliseconds at line 27 before converting to microseconds. This host's system clock uses signed nanoseconds. The native conversion overflows at an admitted millisecond boundary.

The minimal fixture has ID `clock-probe`, sequence `A`, quality `I`, moves `[1]`, signal bits `[0,1,2,3]`, stride 4, sample rate 1000, start sample 0, and supplied spacer `[0,0)`. That spacer materializes a child rather than taking root identity. Only acquisition start changes.

| Acquisition start milliseconds | Go result | Pinned native result | Exact represented output |
|---|---|---|---|
| `1700000000123` | `2023-11-14T22:13:20.123000+00:00` | Same | Yes |
| `9223372036854` | `2262-04-11T23:47:16.854000+00:00` | Same | Yes |
| `9223372036855` | `2262-04-11T23:47:16.855000+00:00` | `1677-09-21T00:12:43.145449+00:00` | No |
| `253402300799000` | `9999-12-31T23:59:59.000000+00:00` | `1816-03-30T05:56:07.066278+00:00` | No |

The exact command, run from `/tmp/nanopore-go-child-interval-review/clock-probe`, was:

```sh
go run . /tmp/nanopore-go-child-interval-review/subread/checks-20261023T043833Z-854148724/native-20261023T043835Z-269283629/build/subread-oracle
```

It exited 1 on four mismatches among eight probes. [clock-probe-boundary.stdout](clock-probe-boundary.stdout) retains every observed result. [clock-probe/main.go](clock-probe/main.go) is the evidence-only probe. Its imported implementation and oracle protocol files are exact copies of the frozen repository files, checked with `cmp`. [clock-probe.sha256](clock-probe.sha256) pins those copies, the probe, inputs, and native output. The probe does not change native source, production code, or acceptance checks.

A separate direct native replay exited 0 and reproduced `native-output.bin` exactly. Its command is in [clock-native-replay.command](clock-native-replay.command), with status and UTC clocks beside it. This proves that the mismatch came from a successfully executing native oracle, not a compiler error or malformed input. Native signed-overflow behavior is host-specific. The evidence does not recommend reproducing overflow in Go. The admitted domain must be reconciled with native representability before the documented source-parity claim can pass.

## Fresh applicable checks

The exact top-level commands were run from the repository root:

```sh
env DORADO_SOURCE=/home/jake/Projects/nanopore_basecaller_go/external/dorado-source SUBREAD_ARTIFACT_ROOT=/tmp/nanopore-go-child-interval-review/subread verification/subread/verify.sh --native

env DORADO_SOURCE=/home/jake/Projects/nanopore_basecaller_go/external/dorado-source INTERVALTRIM_ARTIFACT_ROOT=/tmp/nanopore-go-child-interval-review/intervaltrim INTERVALTRIM_CAPTURE_MANIFESTS=/home/jake/Projects/nanopore_basecaller_go/artifacts/caller/attempt-20261023T023928Z-Wr38Jl/capture/bb76c31a-36f0-48a0-abd2-107a10e1ade0/manifest.json:/home/jake/Projects/nanopore_basecaller_go/artifacts/caller/attempt-20261023T033758Z-xTMh1T/capture/bb76c31a-36f0-48a0-abd2-107a10e1ade0/manifest.json verification/intervaltrim/verify.sh --native
```

Both exited 0. These were fresh executions with `-count=1`, without new inference. Receipts include exact arguments, relevant environment, UTC clocks, exit status, and raw streams.

| Unit | UTC execution window | Ordinary | Race | CGO disabled | Vet | Behavior mutations | Native differential |
|---|---|---|---|---|---|---|---|
| Subread | `2026-10-23T04:38:33.227824883Z` through `04:38:40.084876893Z` | 0 | 0 | 0 | 0 | 0 | 0 |
| Interval trim | `2026-10-23T04:38:40.060776962Z` through `04:38:48.325464812Z` | 0 | 0 | 0 | 0 | 0 | 0 |

Subread evidence is [subread/checks-20261023T043833Z-854148724](subread/checks-20261023T043833Z-854148724). Its native [comparison.json](subread/checks-20261023T043833Z-854148724/native-20261023T043835Z-269283629/comparison.json) records 235 read cases and 21 UUID probes. These cases cover identity, existing parents, zero or multiple children, neighbors, inherited fields, resets, rounded current-era clocks, uint32 split-point wrap, sparse moves, and partial tails. The existing cases do not cover F1's timestamp boundary.

Interval evidence is [intervaltrim/checks-20261023T043840Z-1946232918](intervaltrim/checks-20261023T043840Z-1946232918). Its native [comparison.json](intervaltrim/checks-20261023T043840Z-1946232918/attempt-20261023T043847Z-3670375952/comparison.json) records exact bytes for 821 interval cases and 620 filter cases. The [retained comparison](intervaltrim/checks-20261023T043840Z-1946232918/attempt-20261023T043847Z-3670375952/retained-captures/comparison.json) adds two stored HAC and SUP trims. No other native computation was needed.

## Negative controls and semantic audit

Three isolated subread behavior mutations fail `TestMaterializedMetadata` assertions. They inherit the parent's mean-quality offset, drop inherited run ID, or floor the rounded child clock. Each raw mutation command exits 1. The maintained test succeeds only because those changes are detected. Logs are under the fresh subread `mutations-20261023T043835Z-2091477686` directory. Altered native UUID source is rejected by an actual `source hash mismatch`, and malformed magic, truncated payloads, and trailing protocol bytes produce controlled native exit 2.

Interval controls include one unchanged baseline that exits 0 and nine isolated mutations that exit 1 with real assertion failures. They cover leading offset, following zero moves, pretrim length, event count, cumulative sample count, quality equality, minimum length, offset equality fallback, and ignored offsets. The quality-equality mutation rejects an otherwise passing length-five read at Q40. The minimum-five mutation rejects a length-five read with quality filtering disabled. These failures demonstrate the checks can detect changed observable behavior. Every one of seven native source file alterations is rejected before compilation.

The native child oracle compiles seven unchanged source functions and full original UUID, OpenSSL crypto, and time units. I inspected its flat shared signal view, field transfer, default mean-quality initialization, supplied-spacer seam, and split finalization. The shim transfers each represented field and each signal bit. It does not duplicate the Go split algorithm. Actual children own copied moves, borrow signal storage, reset mean-quality start, events, read number and leading trim, derive IDs from the existing root parent, chain neighbors, and restore the original end reason only on the last child. Leading zero moves and final partial tails follow the native floor endpoints.

The interval oracle compiles seven unchanged source functions and the exact ReadFilterNode condition. Its tensor shim only slices opaque uint16 storage and clamps positive half-open endpoints. Full intervals retain identity. Interior empty intervals clear data without adding a leading offset. `[n,n)` on a nonempty read errors. Trimming retains zero moves after the last retained base until the next removed base. The implementation's leading and cumulative sample overflow guards were inspected. No interval or filter defect was found.

Filter comparisons use float32 accumulation and native libm operations. Equality passes, nonpositive finite thresholds disable quality filtering, zero quality has native clamped mean Q1 for nonempty reads, and offsets at or beyond length fall back to the full quality string. The API's fractional or negative thresholds are documented extensions beyond native CLI `size_t` input. CGO-disabled Filter intentionally returns `decode.ErrCGORequired`, even at minQ zero, because the API also computes mean-quality metadata. Interval slicing remains available in that build.

## Retained evidence integrity

I independently audited the supplied retained subread attempt `artifacts/subread/checks-20261023T043139Z-3407506787` and interval attempt `artifacts/intervaltrim/checks-20261023T043145Z-4273152231`, as well as the fresh attempts. All four recorded source-pin maps match the actual reviewed files. Eighty-two receipt stream hashes match their raw stdout, stderr, and stored stdin when present. The retained capture manifests and payloads match their declared hashes. Copied native source, extracted slices, drivers, compilers, executables, runtime libraries, and the subread oracle's recorded dependency headers match their manifest pins. Fresh differential outputs equal the retained differential outputs.

The rerunnable integrity command was:

```sh
bash /tmp/nanopore-go-child-interval-review/audit-evidence.sh
```

It exited 0. [integrity-v2.stdout](integrity-v2.stdout) and its associated clocks/status retain the audit. The first raw JSON `cmp` exited 1 because Go nil slices serialize as `null` while decoded native empty slices serialize as `[]` or an empty base64 string. The field protocol encodes both as length zero. The audit's second pass normalizes only those zero-length slice representations and matches every field. The initial script and failed receipt remain as `audit-evidence-v1.sh` and `integrity.*`.

## Unproven behavior and execution gaps

The acknowledged ceil-sized move-table child gap remains. Native final-child moves end at floor(signal length / stride), so a ceil-sized parent can produce a child with one fewer move than sequence bases. The standalone API preserves it. This review does not accept the caller's proposed floor-only composition rule because caller composition was excluded.

The two retained subread captures prove supplied-no-spacer identity only. Positive splitting evidence is synthetic. Neither unit proves real detector decisions, complete input processing, output FASTQ, full native container layout, ATen internals, client behavior, RNA, duplex or modified-base behavior. The interval oracle records source, driver, compiler and runtime pins but does not record every transitive standard-library header, unlike the subread oracle. Tests and sampled comparisons do not prove all valid inputs, and F1 demonstrates that limit.

No full-pipeline, GPU, inference, biological accuracy, throughput, or performance claim follows from this review. The full repository suite was not run because another owner was changing caller composition and it is outside these two frozen units.

No required tool is broken. One probe setup invocation failed before execution because it selected a working directory that had not yet been created. It was corrected by creating the directory first. No command ran in that failed invocation, so it has no process receipt. The initial six-case clock probe's input/output paths were overwritten by the expanded eight-case probe. Its original source and console receipt remain, and `clock-probe-initial-replay` reconstructs that deterministic six-case input and reproduces the same result under a separate retained receipt. The decisive eight-case boundary input and output remain intact and hash-pinned.

Poteto Mode's Prove It Works principle led to fresh native replay, receipt hash verification, negative controls, and the additional real-artifact clock probe. Delegation and comment-edit workflow steps were skipped because this assignment prohibits children and production edits.
