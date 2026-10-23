# Post-call unit review

Verdict PASS for the two stated component contracts, with one minor documentation finding. No implementation defect found. This is a same-family independent source and artifact review.

The reviewed workspace is `/home/jake/Projects/nanopore_basecaller_go` at base `2ea835007b332b09eb4cc4ae239aa87d9ec30ed8`. Both units are uncommitted additions. Their exact reviewed bytes are pinned in `replay-source-manifest.sha256` and `readtrim-source-manifest.sha256` beside this report. Other work was present and remained outside review.

## Qualified result

Replay qualifies exact float CPU source decoder and stitch parity on the retained Go CPU13 HAC and SUP tensors. All four chunks have 2,048 blocks. The two HAC chunks have 256 states, and the two SUP chunks have 1,024 states. The independent audit checked all 31,462,400 score, backward-guide, and posterior float32 bits against the actual native input protocols. Forward arrays and all other declared payloads were hash-checked. Native chunk sequence, quality, moves, state arrays, final-score bits, and quality-data bits match the Go outputs. Native stitch input contains the actual native decoded outputs. Both final native outputs match the Go and retained sequence, quality, and move bytes. The HAC endpoint has 1,653 bases, and the SUP endpoint has 1,650 bases. Each endpoint has 3,307 move bytes.

The retained replay evidence is `artifacts/replay-moves/attempt-20261023T035505Z-niop51/`. All six process receipts match their actual input, stdout, and stderr hashes. All six exits are zero, with ordered UTC clocks. Source hashes, retained extracted decoder source, compiler executable, available oracle executables, and linked libc, libm, libgcc, libstdc++, and loader hashes match. The Dorado checkout HEAD is `8b8fc5d36a9c0baab262a743cb175b5878e38ca6`.

Readtrim qualifies the standalone MUX transform against four complete verbatim functions from that Dorado commit. Source indexing, float32 0.3 threshold, equality handling, minimum trim, zero-move boundary, signal prefix, events reset, and pre-trim metadata agree with the source. The read and signal shims expose the accessed types and the explicit slice contract. Go borrows the input slices, limits trimmed capacity, preserves no-op metadata, and performs no writes. Invalid input checks run before no-op returns.

The retained readtrim evidence is `artifacts/readtrim/checks-20261023T035855Z-3329615571/`. The audit checked its final source hashes, all six check receipts and log hashes, four full source pins, four verbatim source slices, driver, executable, compiler, runtime dependencies, and actual binary native and Go outputs. The 892 ordinary cases include 458 positive trims. The three large float32 boundary cases contain one positive trim and two no-ops. The two genuine HAC and SUP captures are non-mux identities.

## Rerun evidence

Exact commands are in `commands.txt`. All commands ran in the reviewed workspace and wrote evidence only under this report directory.

| Check | Observed result | Evidence |
| --- | --- | --- |
| Replay ordinary tests | Exit 0. Native and genuine controls explicitly skip in this mode. | `replay-checks/ordinary.txt` |
| Replay race tests, vet, shell syntax | Each exits 0. | `replay-checks/race.txt`, `vet.txt`, `shell-syntax.txt`, and adjacent `.exit` files |
| Seven copied-capture rejection controls and two source mutations | Exit 0. Each altered input is rejected with its intended diagnostic. | `replay-checks/rejections.txt` |
| Retained native decoder with the short input protocol | Exit 2, stderr `truncated float bits`, empty stdout. | `short-protocol/` |
| Readtrim ordinary, race, no-cgo, vet, shell syntax | Each exits 0. Native and mutation tests explicitly skip in ordinary mode. | `readtrim-checks/` logs and `.exit` files |
| Twelve isolated behavior mutations | Exit 0 for the rejection suite. Every changed implementation exits 1 through a behavior assertion. | `readtrim-checks/mutations.txt`, `readtrim-checks/mutations-20261023T040040Z-2923946572/` |
| Readtrim native source differential rerun | Exit 0. 892 ordinary cases, three float32 cases, two retained identities, and four source alterations checked. | `readtrim-checks/native.txt`, `readtrim-native/attempt-20261023T040321Z-2559133328/` |
| Independent retained protocol and binary artifact audit | Exit 0. Direct file values match, including native outputs and summary hashes. | `artifact-audit.json`, `artifact-audit-v2.stdout` |
| CLI malformed JSON, type, base64, integer overflow, byte-count overflow, absolute path, and trailing JSON | All seven cases exit 1 with the intended diagnostic and empty stdout. These cases fail before omitted large tensor payloads are accessed. | `json-rejections/summary.json`, per-case receipts and logs |

The Q12 mutation fails `TestMuxBehavior/mid12-tie-earlier-minimum`, returning end 91 instead of 89. The float64 threshold mutation fails `TestBinary32ExcessiveBoundary`, trimming length 3,495,263 when no trim is expected. These observed failures show the checks detect the relevant mistakes.

## Finding

P3, `docs/readtrim.md:116`. The statement that native decoder parity is a separate outstanding contract is stale when both reviewed units are delivered together. Replay now supplies exact source CPU beam and stitch move evidence on these retained Go CPU13 tensors. Official CUDA12 binary moves remain unproven because the stock endpoint is FASTQ only. The adjacent native test message at `internal/readtrim/native_test.go:124` is accurate for readtrim alone, but can be read too broadly in a combined run. A crosslink and explicit source-versus-official-binary wording would remove the ambiguity. No source or documentation changes were made during this review.

## Evidence limits

The result does not qualify official binary moves, GPU execution, default splitting, adapter or primer trimming, positive biological MUX tails, general ATen behavior, caller integration, biological accuracy, or performance. No model, POD5, stock basecaller, download, GPU job, or inference ran during review. The full replay native producer was not repeated. Its retained protocols and native outputs were checked directly, and its short-input rejection was rerun.

Capture hashes authenticate local consistency, not historical producer provenance. Shared libm semantics are part of the numerical contract. Compiler executable and runtime pins do not pin every compiler frontend, assembler, system header, or NVTX header. Thus the receipts support the observed local build, not a complete reproducible toolchain closure.

The replay receipt pins `/tmp/go-build1100024484/b131/replaymoves.test`, but that temporary executable no longer exists. Its recorded hash cannot be rehashed now. The reviewed source still matches its recorded replay source hashes, and current checks were rerun. The missing historical executable limits complete reconstruction of the original process. The source identity inventory also includes in-progress readtrim files from that time, so it is not a final combined revision manifest.

Large native count guards were inspected, not exercised with multi-gigabyte allocations. Prefix-value parity does not prove storage aliasing parity with ATen. The no-write and borrow contract is separately checked by Go tests and race tests.

Two reviewer evidence attempts failed before completing. `artifact-audit.stderr` records an incorrect uppercase key in the reviewer script. Its corrected rerun passes and preserves the original script as `audit_artifacts-attempt1.py`. `json-rejections/run.stderr` records that hardlinks cannot cross from the workspace filesystem to `/tmp`. The corrected script copies only the small prerequisite payloads. Its rerun passes. Neither failure changed implementation or acceptance checks. No other broken tools or missing required component receipts were observed.

The Prove It Works principle changed the review from receipt-summary inspection to direct protocol floats, native bytes, and fresh rejection runs. No child agents were spawned. Implementation and acceptance checks remained unchanged.
