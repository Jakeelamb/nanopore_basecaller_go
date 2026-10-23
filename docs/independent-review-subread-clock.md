# Verdict

**PASS for F1's bounded subread clock-domain fix at the frozen artifact below.** No new defect was found in the six reviewed files. This closes the earlier timestamp-domain failure only. The earlier standalone leaf conclusions remain conditional on their original scope and limits. Caller composition and default integration were reviewed elsewhere.

## Artifact and scope

Repository is `/home/jake/Projects/nanopore_basecaller_go`, branch `port/go-caller`, observed HEAD `3b402f50ca0b9c28d6a484267d61df3aeb5b6034`. These leaf files are untracked additions, so HEAD alone does not identify them. [freeze-before.stdout](freeze-before.stdout) and [freeze-after.sha256](freeze-after.sha256) identify the exact reviewed bytes and match byte for byte. They include the six files, unchanged verification entrypoints, `go.mod`, and relevant native source units. [delta-pins.stdout](delta-pins.stdout) compares their hashes with the original review's authoritative `/tmp/nanopore-go-child-interval-review/before.sha256`.

Reviewed files are `internal/subread/subread.go`, `internal/subread/subread_test.go`, `verification/subread/fixtures.go`, `verification/subread/oracle.cpp`, `verification/subread/oracle_test.go`, and `docs/subread.md`. All nine subread files also match the fix worker's `owned-final.sha256` and preserved `final-source/` byte for byte. The core implementation hash is `d162401575831a1144cedf1b152f30b88859bd2cfbce5a4c1949a128bf6fe17f`.

The native source checkout remains clean at `8b8fc5d36a9c0baab262a743cb175b5878e38ca6`. The original time unit hash remains `5b4e928ece39f04d4c22dd8310c1a7f8c6878a7663e86bb46fec6fae9ed4f75d`. The seven extracted native functions and full original time unit match the original review's preserved native build. The changed oracle driver adds only the static clock checks and separate guarded command mode. Its binary-protocol path and original source conversion are unchanged. [code-delta.stdout](code-delta.stdout), [oracle-delta.stdout](oracle-delta.stdout), and [audit-v2.stdout](audit-v2.stdout) retain these checks. The two `diff` commands exit 1 because they show the expected changes.

## Requirement inspection and actual behavior

`subread.go:13` derives the last nonnegative whole millisecond from signed 64-bit nanoseconds as `9223372036854`. The unchanged native formatter first constructs a `system_clock` time point and only then converts it to microseconds. The exact last timestamp is `2262-04-11T23:47:16.854000+00:00`.

`subread.go:134` computes each actual child's sample start, applies the original `math.Round` expression, and checks its sum with acquisition start before integer addition and formatting. Acquisition is validated before this subtraction. The first comparison prevents an out-of-range float-to-uint64 conversion. The bound is exactly representable as float64. Existing sample-coordinate guards cover the start additions. Signal end does not constrain the converted start.

Both child call sites return `nil, err` on failure, including after an earlier child was constructed locally. No partial children escape. Empty sequence or moves bypass child creation. No-spacer roots preserve their clocks, and parented no-spacer reads retain native ID finalization. Full-spacer results emit no children. Acquisition and coordinate validation still apply to these paths, while their existing clock fields remain unconverted. The documented ceil-tail move behavior is preserved.

An independent evidence-only probe compared seven additional accepted cases against the newly built native executable. Every represented field matched. Cases include a child starting at the maximum while its signal ends later, a relative-only maximum, root and parented no-spacer reads with oversized relative offsets, empty-moves and empty-sequence bypasses, and an all-spacer zero-child result. Three Go-only rejection cases return nil and an error for a later overflowing child, invalid acquisition even on an empty result, and coordinate overflow on a no-spacer root. Invalid clocks were never sent through the native conversion. See [extra-probe.stdout](extra-probe.stdout), [probe/main.go](probe/main.go), and [probe/native-extra.receipt.json](probe/native-extra.receipt.json).

The shim's `--clock-boundary` mode rejects an unrepresentable sum before calling the original formatter. Its exit 2 is evidence about that shim guard. It is not evidence that unchanged Dorado reports an error on overflow. Accepted actual-child endpoints also run through the unchanged native `subread` function and full original time unit.

## Fresh checks and negative case

Run from the repository root, the exact maintained command was:

```sh
env DORADO_SOURCE=/home/jake/Projects/nanopore_basecaller_go/external/dorado-source SUBREAD_ARTIFACT_ROOT=/tmp/nanopore-go-subread-clock-review/fresh verification/subread/verify.sh --native
```

It ran from `2026-10-23T13:33:47.081615524Z` through `2026-10-23T13:33:55.121514029Z` and exited 0. Evidence is [fresh/checks-20261023T133347Z-1174948803](fresh/checks-20261023T133347Z-1174948803). All checks exited 0 with fresh `-count=1` tests.

| Check | Actual command, with relevant environment | Result |
|---|---|---|
| Ordinary | `go test -count=1 -v ./internal/subread ./verification/subread/...` | PASS, including eleven clock scenarios and bypass validation |
| Race | `CGO_ENABLED=1 go test -race -count=1 ./internal/subread ./verification/subread/...` | PASS |
| No cgo | `CGO_ENABLED=0 go test -count=1 ./internal/subread ./verification/subread/...` | PASS |
| Vet | `go vet ./internal/subread ./verification/subread/...` | PASS |
| Mutations | `SUBREAD_MUTATIONS=1 go test -count=1 -v ./verification/subread -run '^TestBehaviorMutations$'` | PASS because three isolated behavior mutations produced assertion failures and exit 1 |
| Native | `DORADO_SOURCE=/home/jake/Projects/nanopore_basecaller_go/external/dorado-source go test -count=1 -v ./verification/subread -run '^TestPinnedNativeMaterialization$'` | PASS, 238 exact read cases, 21 UUID cases, ten clock probes |

Each check's `.receipt.json` contains exact argv, relevant environment, UTC clocks, exit, and raw-stream hashes. The final native directory is `fresh/checks-20261023T133347Z-1174948803/native-20261023T133349Z-3309419223`. Its native payload hash is `39b82304322191d4826d21a6df2a10ec9b77cf865dab7311554f2f5b8090e2c9`, matching the worker's final payload.

To prove the new regressions detect the old defect, I copied the exact preserved pre-fix implementation and exact final tests into `negative/`. I changed neither copied file. The old implementation hash matches the original review's `526f359915b66554e8bc52c3037f2a34240fdc4f60d00cc123e89364ac22f8dc`. From that isolated directory I ran:

```sh
go test -count=1 -v ./internal/subread -run '^(TestNativeClockDomain|TestUnconvertedClockBypass)$'
```

It exited 1 from `2026-10-23T13:36:30.448234369Z` through `2026-10-23T13:36:30.663767837Z`. Six clock scenarios fail with actual improperly accepted outputs, including the later-child case, and the bypass test detects invalid acquisition. This is a real negative case, not a compilation or setup failure. See [negative-old-source.stdout](negative-old-source.stdout) and [isolated-source.sha256](isolated-source.sha256).

From `probe/` the exact additional command was:

```sh
go run . /tmp/nanopore-go-subread-clock-review/fresh/checks-20261023T133347Z-1174948803/native-20261023T133349Z-3309419223/build/subread-oracle
```

It exited 0 from `2026-10-23T13:36:30.452329122Z` through `2026-10-23T13:36:30.649625087Z`.

## Evidence integrity

The rerunnable audit command is:

```sh
bash /tmp/nanopore-go-subread-clock-review/audit-v2.sh
```

It exited 0. It independently verified 64 retained and fresh command receipts against raw stdin, stdout, and stderr; all six check exits and clocks; final source freezes; full native source, copied units, slices, driver, compiler, executable, runtime libraries, and recorded dependency headers; retained capture pins; exact read fields after only null/empty slice normalization; and all ten clock outcomes. The seven native functions and time unit also match the original review. The extra probe's receipt streams and exact source copies were verified by `final-checks.sh`, which exited 0 and confirmed the scoped before/after freeze and clean native checkout.

[commands.tsv](commands.tsv) records every retained outer command, working directory, clock, and exit. Each outer receipt has a `.raw.sha256`. Native and nested check stream hashes reside in their JSON receipts. [external-evidence.sha256](external-evidence.sha256) pins the supplied review, original freeze, fix handoff, and red receipts used here. Selected raw stdout pins are:

| Evidence | SHA256 |
|---|---|
| `maintained-native.stdout` | `c9ccdfec296531f1576060c887cf610c0138f2d7197b51880084479f176ae2bd` |
| `negative-old-source.stdout` | `e3d4122694ca79da610e0e4043eeb59258f4b027c420d40784b9a1a862d7a1cf` |
| `extra-probe.stdout` | `c9df27679490f3a4e9a9929086a61002e0bc2e3e4cfe6579082d53c42beba9df` |
| `audit-v2.stdout` | `41d2ac47156fc80608701445625de4482d0347dcb29070bba9a5814bda3cc889` |

## Limits and observation failures

A complete initial six-file source snapshot is unavailable. Initial implementation bytes and the original native driver are preserved, enabling exact deltas for those two files. Initial bytes for `subread_test.go`, `fixtures.go`, `oracle_test.go`, and `docs/subread.md` are missing. Their original hashes remain in the first review freeze. Their final contents were inspected, but a complete six-file line-by-line delta is unproven. The red test snapshot is a later pre-fix regression artifact, not the original test file.

No required execution tool is broken. The first audit script exited 1 because it queried nonexistent receipt fields such as `started_utc` instead of the actual `started`. Its failed receipt and script remain as `audit.*` and `audit-v1.sh`. `audit-v2.sh` uses the retained schema and passes without replaying any successful producer. One exploratory `cat` used the wrong receipt filename separator and failed; the actual `.receipt.json` files exist and were audited.

This is Linux signed-nanosecond clock and standalone supplied-spacer evidence. It is sampled execution plus source inspection, not an exhaustive proof over all coordinates and rates. The acknowledged ceil-sized move-table quirk remains. Positive splits are synthetic, and the two retained HAC/SUP captures prove no-spacer identity only. Other clock platforms, pore detection, caller composition, full input processing, FASTQ, full ATen or native container behavior, inference, GPU execution, biological accuracy, and performance remain unproven by this review. No full repository or neural-network suite was repeated.

Only assigned evidence and temporary copies under `/tmp/nanopore-go-subread-clock-review` were written. No implementation or acceptance check was edited, and no child agent, commit, download, GPU action, inference, or original-workspace write occurred. Poteto Mode's Prove It Works principle led to fresh native execution, the preserved-old-source negative case, and direct boundary comparisons. Delegation and comment-edit steps were omitted under the explicit bounded assignment.
