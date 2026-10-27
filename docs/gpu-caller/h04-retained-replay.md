# H04 retained cache replay

The host command authenticates the retained H04 native capture against an
independently pinned historical producer receipt. It reads all 98 tensor payloads,
including all 49 final cache tensors. Each observed native digest must match both
digests and the equality flag in its historical comparison record. Geometry,
direction, quantizer contract, shapes, dtypes, byte counts, input identity, and
per-layer domain records must agree.

The original successful producer receipt has SHA256
`15e88aaca5fb2de6f00c9485b42634528bd66d7518115e18b4c8917b8e7b4eff`.
The replay creates a fresh receipt and leaves the producer evidence unchanged.
Its consumer executable, Go build information, invocation, runtime limits, and
current source and test file identities are separate from historical provenance.
Retain the outer build and invocation receipts with the replay receipt.
The source inventory covers the standalone H04 command's dependency packages,
their tests, native ABI headers, and both embedded weight manifests. A sibling
H05 verification edit does not change this inventory. A packer edit does.

Run from the repository root with a fresh output directory:

```sh
GOMAXPROCS=2 GOMEMLIMIT=3GiB CGO_ENABLED=0 GOPROXY=off GOSUMDB=off \
  go run ./verification/hacgpu/h04/cmd/replay \
  -receipt /home/jake/.local/share/opencode/worktree/e3db79/H04/artifacts/tensor/h04-cache-001/receipt.json \
  -receipt-sha256 15e88aaca5fb2de6f00c9485b42634528bd66d7518115e18b4c8917b8e7b4eff \
  -out artifacts/tensor/h04-retained-replay-NEW
```

A successful replay reports `retained_witnesses_authenticated`, 98 native
payloads, 49 final caches, and nine changed-copy controls. The controls alter a
genuine copied payload, capture shape, source revision, quantizer, direction,
historical Go digest, oracle build manifest, oracle source, and current consumer
source. Each control must change the bytes and fail validation. Unchanged control
payloads are read through symlinks to retained native files. The changed payload
is a separate regular file. The controls never write through those symlinks.

The same command has a separate `-verify-consumer` mode. It requires a Go binary
built with cgo, an accepted `-backend`, and explicit `-runtime-admitted`
coordination. The existing native CPU loader reads the 42 pinned float32 weight
archives. Go runs production `PackFLSTMStack` and `QuantizeFLSTMLayer` and
compares all 98 outputs with the retained native capture. It saves the observed
Go payloads and all successful mutant cache payloads. Seven arithmetic mutants
must differ or be rejected in every layer. An additional changed-copy control
rejects a corrupt observed Go payload.

This mode writes `consumer-receipt.json` beside the witness receipt. It pins the
current consumer, source and tests, accepted backend and declared build inputs,
and the shared libraries mapped while the CPU loader is open. It rehashes all
saved Go and retained native payloads before success. This mode creates no CUDA
device and calls neither model forward nor the historical oracle. Its fresh Go
arithmetic evidence is separate from the historical witness authentication.

The default witness mode authenticates retained witnesses. It does not rerun quantization,
packing, the seven arithmetic mutants, a native producer, or GPU computation.
The historical producer omitted Go source and executable pins, complete oracle
compilation and link inputs, authenticated child runtime maps, and device
bookend agreement. The replay preserves these gaps. A current source pin cannot
prove which historical Go implementation produced a recorded digest.
The `-verify-consumer` mode provides new arithmetic evidence for the current Go
revision and still leaves the historical provenance gaps open.

The public quantizer and packing boundaries reject overflowing geometry before
allocation. The regression `QuantizeFLSTMRows(nil, 1<<62, 4)` previously panicked
because the product wrapped to zero. It now returns `ErrGPUWeights`. Tests also
reject overflowing packer, view, and permutation shapes. These host checks do
not qualify general CUDA half division, H03, H05, end-to-end calling, or
biological accuracy.

The first recorded regression failure is
`artifacts/tensor/h04-overflow-before-20261025-002/receipt.json`.
The focused passing correction is
`artifacts/tensor/h04-overflow-after-20261025-001/receipt.json`.
The initial witness revision checks are recorded under
`artifacts/tensor/h04-host-tests-20261025-002/`,
`artifacts/tensor/h04-replay-build-20261025-002/`,
`artifacts/tensor/h04-replay-invocation-20261025-002/`, and
`artifacts/tensor/h04-retained-replay-20261025-002/`.
The unreceipted first regression attempt and the earlier passing replay are
retained as separate attempts. Current consumer mode host checks are under
`artifacts/tensor/h04-consumer-host-tests-20261025-005/`.

The current Go arithmetic passed in
`artifacts/tensor/h04-current-consumer-20261025-004/consumer-receipt.json`.
It records 98 exact fresh comparisons, 49 final caches, 294 mutant comparisons,
392 saved Go payloads, 93 consumer source and test pins, and 25 mapped runtime
pins. The reduction-axis mutant rejects all seven layers. Each of the other six
mutants differs in all seven layers. The observed Go corruption control rejects
a changed copy. Build and invocation receipts are under
`artifacts/tensor/h04-consumer-build-20261025-004/` and
`artifacts/tensor/h04-consumer-invocation-20261025-004/`.

Consumer attempt 001 failed before native loading because a shared backend's
declared Go input had changed after its build. Attempt 002 completed the exact
arithmetic comparisons but failed its final source check when the old inventory
included an unrelated H05 edit. Both failed receipts and their payloads remain.
Attempt 003 passed at its recorded source state. A later H05 observation callback changed one declared model-package input, so attempt 004 rebuilt the consumer and repeated the bounded CPU comparison against the frozen current source. Its final audit is `artifacts/tensor/h04-final-audit-20261025-004/summary.json`. The corrected dependency inventory excludes unrelated sibling verification packages. The canonical nonblocking host lock covers CPU loading and all final
observations. Normal host tests use a temporary held lock and never take the
real host lock.
