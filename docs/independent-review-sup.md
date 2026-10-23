# Independent review verdict

PASS for tensor ABI 3, the Go-owned SUP graph, assigned documentation, and the bounded default SUP/HAC retained caller endpoints. No unresolved implementation finding in the assigned scope.

The reviewed checkout is HEAD `086f2361b7b4900ddcf0ceb1cdbfad1cf7a9c1e2`, with uncommitted work over HAC freeze `7a0f193`. Exact final bytes for 61 scoped source, test, verification, and documentation files are in `source-freeze.json` and `review-manifest.json`. This verdict attaches to those hashes, not to a claim that HEAD alone contains the implementation. No implementation, maintained acceptance check, or original stock workspace was changed by this reviewer.

## Independent execution

All commands ran in `/home/jake/Projects/nanopore_basecaller_go`. Exact environment overrides, UTC clocks, exit status and log hashes are in the listed JSON receipts.

| Exact command | Observed result | Evidence |
| --- | --- | --- |
| `go test -count=1 -v ./internal/tensor ./internal/model ./verification/model` | Exit 0. Ordinary tests pass; native gates explicitly skip without configured inputs. | `checks.json`, `ordinary.log` |
| `go test -count=1 -v ./internal/tensor` | Exit 0 with ABI3 library, genuine HAC weight, and immutable ABI2 rejection library configured. Includes primitive bit comparisons, copied masks, negative shape/value/overflow/foreign/closed inputs, and actual ABI2 rejection. | `native-tensor.json`, `native-tensor.log` |
| `CGO_ENABLED=0 go test -count=1 ./internal/tensor ./internal/model ./verification/model` | Exit 0. Unsupported-native path passes. | `nocgo.json`, `nocgo.log` |
| `go test -count=1 -race ./internal/tensor ./internal/model ./verification/model` | Exit 0. Tensor native gate configured; model native producer gates unset. | `race.json`, `race.log` |
| `go vet ./internal/tensor ./internal/model ./verification/tensor/... ./verification/model/...` | Exit 0. | `vet.json`, `vet.log` |
| `artifacts/tensor/checks-008-sup-abi3/abi-probe artifacts/tensor/attempt-005-sup-abi3/libnanopore_tensor.so` | Exit 0. Native SDPA exceptions keep output handle unpublished, respect capacities 0, 1, and 31, and reject null error buffers. | `abi-probe.json`, `abi-probe.log` |
| `go test -count=1 -v -overlay /tmp/nanopore-go-sup-review/overlay.json -run '^TestIndependentSUPPublicConsumer$' ./internal/model` | Exit 0. Public `Open` consumes 139 pinned weights. One `Forward` matches all 524,288 output F32 values bitwise at shape `[1,128,4096]`. Input/output survive model close. | `public-consumer-receipt.json`, `public-consumer-result.json`, `public-consumer.log` |
| `go test -count=1 -race -v -run '^TestRetainedHACModelScores$' ./internal/model` | Exit 0 with retained HAC input/model/backend configured. All 16,384 output F32 values match. Cancellation, reuse and lifetimes pass. | `retained-hac-model-race.json`, `.log` |
| `go test -count=1 -v ./internal/caller` | Exit 0 with producer gates unset. SUP admission boundary controls and existing capture tamper tests pass. | `caller-unit.json`, `.log` |
| `go test -count=1 -race -v -run '^TestCapturedNativeEndpoint$' ./internal/caller` | Exit 0 separately for SUP and HAC retained manifests. Raw FASTQ exact. SUP has 1,650 bases; HAC has 1,653. | `caller-sup-race.json`, `.log`; `caller-hac-race.json`, `.log` |
| `go build -o /tmp/nanopore-go-sup-review/basecaller ./cmd/basecaller` | Exit 0. | `evidence-build.json` |
| `strace -f -e trace=execve,openat -o /tmp/nanopore-go-sup-review/{sup,hac}-replay.strace /tmp/nanopore-go-sup-review/basecaller replay --manifest <retained manifest>` | Both exit 0. Each trace has one execve and no tensor weight, tensor backend, or POD5 library opens. Full literal paths are in `{sup,hac}-replay.json`. | `review-manifest.json`, trace files |
| `cmp /tmp/nanopore-go-sup-review/sup-replay.stdout artifacts/native-cpu-control/stock-sup-dev1-001/stdout` | Exit 0. Raw SHA256 `dbb28c99d9c29d9808bc27c56f96a1057cfdf0b5fe197ac940e898d380b994c0`. | `caller-raw-checks.json` |
| `cmp /tmp/nanopore-go-sup-review/hac-replay.stdout artifacts/native-cpu-control/stock-hac-dev1-001/stdout` | Exit 0. Raw SHA256 `53ec2ca57acff8ce0041576a6bade2724a623110421867827032990d8df860cd`. | `caller-raw-checks.json` |
| `git diff --check` | Exit 0 on final scoped docs and source. | Tool receipt in review transcript. |

The overlay adds only an independent evidence test in `/tmp`. It does not replace production source or maintained tests. `consumer_test.go` and `overlay.json` are hashed in `public-consumer-receipt.json`. It is separate from the maintained qualified parity gate.

## Negative controls

- The independent public score comparator detects exactly one differing F32 bit pattern after flipping one copied expected score bit.
- A naive whole masked attention call differs from the retained original 12-split result at 355 of 2,080 F32 positions on the T260 component fixture. This directly demonstrates that the long fixture detects the split-boundary error.
- Changed config and first-weight bytes fail public `Open`. Unaligned input, canceled context, and closed model calls reject.
- A copied regular SUP capture manifest with first chunk `States=256` fails with `captured chunk shape or coordinates differ`. The genuine SUP capture has 1,024 states, two 2,048-block chunks, and offsets 0 and 7,560. See `reject-sup-hac-states-final.json`.
- A valid four-line FASTQ with channel 2479 changed to 12479 exits 1 in `go run ./verification/caller/cmd`. UUID, sequence, and quality remain exact while raw header and FASTQ comparisons fail. See `caller-raw-checks.json` and `changed-header-final.stdout`.

## Source and artifact assessment

`uv run --no-project python /tmp/nanopore-go-sup-review/audit_retained.py --sup-parity artifacts/model/sup/attempt-20261023T033405.423939301Z/parity.json` passed. It verified 21 current tensor sources, six check inputs, nine check logs, 4,576 tensor build inputs, ten runtime libraries, five build logs, the ABI3 output and original LibTorch archive, 30 parity-declared current sources, six native absolute inputs, 329 synthetic payloads, all 139 genuine SUP archives, and 176 original stock capture files. It independently checked all ten selected original source files and verbatim extracted slice text/hashes. The original Dorado checkout is unchanged at `8b8fc5d36a9c0baab262a743cb175b5878e38ca6`.

The immutable ABI3 library SHA256 is `9d215db7abc78150807769677602c917f645a913829e418b10fa02147f61984f`. Retained corrected SUP evidence reports 12 fixtures, 287 stages, and 15,228,786 exact F32 values. This reviewer did not regenerate those native captures. The independent public consumer and negative controls check the current artifact beyond the worker report.

Inspection confirms the private typed graph dispatch, unchanged HAC graph/manifest, source RoPE double power followed by F32 cast and reciprocal, Q/K halves and unchanged V, default SDPA with copied bool masks and literal asymmetric split bounds, RMS epsilon/order, two DeepNorm residuals, SiLU gate ordering, upsample interleaving, and weight-side scale-once without tanh. `load_tx_model_weights` consumes the explicit 139 files and omits the 18 deepnorm-alpha archives. Production exports no whole-model native wrapper.

The shipped C++ functional linear header matches both its declared hash and original ZIP member. For rank-three biased input it performs matmul then separate bias addition. Independent comparison of retained probe bytes found fused `at::linear` differs at 2,294 of 32,768 positions, first at element 31; C++ module, functional, and explicit matmul/add match all original projection bits. See `linear-wrapper-review.json`. The corrected SUP-only `linearNTCBias` follows that order; HAC and generic tensor Linear remain unchanged.

The default caller admission uses fixed CPU chunks, SUP sizes divisible by 768 and overlaps divisible by 12. Native source selects the fixed path through the default `variable_chunk_sizes=false`; the selected genuine read has `signal_positive`. Replay derives CRF states from pinned output channels, admitting SUP 1,024 and preserving HAC 256. The producer's initial compare failure after successful inference is preserved, with the subsequent state-count fix tested against retained output. No inference was restarted.

The two documentation findings are resolved in the final files. Explicit zero overlap is labeled a Go extension without native endpoint parity. The stale claim that SUP graph execution remained unqualified was replaced with the demonstrated synthetic scope and linked separate caller evidence.

## Limits and failed observations

This is CPU13 development/source-function evidence plus one frozen HG002 read endpoint. It does not establish original module ABI, stock CUDA12.8 GPU behavior, native move parity, complete-input processing, larger cohorts, biological accuracy, or performance. Mux-change tail trimming remains outside the demonstrated caller conditions. Explicit zero-overlap native parity is unqualified.

Go race checks do not instrument C++, prove absence of leaks, or make cancellation interrupt an active synchronous ATen operation. Ownership is supported by scoped-source inspection and cancellation/reuse/lifetime behavior. The native oracle manifest does not pin every transitive C++ header. Its selected source/executable/compiler/library inputs, tensor header dependency closure, original archive, and required functional-linear header were independently checked.

No execution tool was broken. Reviewer observation errors are preserved and corrected. An inline hash assertion initially omitted `287`; a tensor member-map lookup failed because the new nn header was absent; a nullable source-slice list required handling. The first replay runner combined stderr with stdout, so initial `cmp` attempts included `replayed reads=1` before the FASTQ and failed. The retained bytes were split using that exact status line and checked again without rerunning replay. The initial symlink-based tamper attempt failed at file-type validation; the final regular-file tamper reached and failed the intended state check. `additional-checks.json` retains those initial nonpassing observations and must be read with `caller-raw-checks.json` and `reject-sup-hac-states-final.json`.

Applied Prove It Works by inspecting actual pinned functions, validating retained byte identity, and executing a separate public consumer. No child agents or cross-model panel were used in this verifier task.
