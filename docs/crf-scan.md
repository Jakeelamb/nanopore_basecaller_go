# CPU CRF scans

`internal/crf` gives Go ownership of Dorado's forward and backward guide loops,
transition index construction, reverse-time scheduling, and posterior scheduling.
Native calls perform individual tensor operations through tensor ABI 2. There
is no native production CRF scan or whole-decoder wrapper.

## Interface and storage

```go
type Result struct {
	Blocks, States int
	Scores, Forward, Backward, Posts []float32
}
func Compute(context.Context, *tensor.Engine, *tensor.Tensor, float32) (Result, error)
func (Result) DecodeInput() decode.Input
```

`Compute` accepts CPU float32 scores in `[T,1,4*States]` order and a finite blank
score. It currently supports exactly one batch. `States` is an exact integer
power of four in `[4,16384]`. `T` is positive. Scores and all three guide result
arrays together are bounded to `2^27` float32 elements before allocation.

The returned `Scores` array is contiguous `[T,4*States]`. `Forward`, `Backward`,
and `Posts` are contiguous `[T+1,States]`. All returned arrays own their storage.
The input tensor is borrowed and remains unchanged. `DecodeInput` borrows the
result's scores, backward guides, and posterior arrays. Those slices must remain
unchanged while `decode.Decode` uses them.

Nonfinite values and unsafe accumulation ranges are rejected before scanning.
The absolute score and blank bound is
`MaxFloat32 / (64 * (T+1) * (T+1))`. This keeps accumulated guides within the
existing CPU beam decoder's stricter input range. Rejected inputs have no native
parity claim.

A canceled operation returns an empty result. Cancellation is checked before
work, inside both time loops, and around posterior construction. A synchronous
native operation finishes before the next cancellation check. Every temporary
float tensor and index tensor closes on success, error, or cancellation. Each
scan step releases its intermediate tensors before advancing.

## Source schedule

The source is Dorado commit
`8b8fc5d36a9c0baab262a743cb175b5878e38ca6`. The verified
`dorado/basecall/decode/CPUDecoder.cpp` SHA256 is
`b8e147e771108134860a9eb291b12ec1b9c08b7e8f684ade6620844372249235`.

For each destination state `s` and base `b`, the Go forward index is
`idx[s,b] = s/4 + b*(States/4)`. Each step adds transition scores to its four
predecessor guides. It concatenates the stay transition first, then the four
step transitions, and reduces that last dimension with ATen `LogSumExp`.
Forward row zero is all zero.

The backward score permutation comes from ATen's ascending, unstable
`ArgSort(idx.flatten())`. Go copies that int64 permutation once and shifts its
values right by two to construct the backward predecessor indices. It preserves
the native ordering of ties. It does not substitute a Go sort. The loop visits
times in reverse and writes each result into its original time row. Backward
row `T` is all zero.

Go stores completed rows in owned arrays and retains only the current guide
row during a scan. The original source's unfilled `-1E38` rows are never read by
this schedule. Posterior values come from `Softmax(Forward + Backward, -1)` with
the same shapes and operation order as the original source.

## Oracle and acceptance

`verification/crf` verifies the entire source file before extracting these
unmodified statements.

- Lines 16 through 37 contain `scan`.
- Lines 42 through 91 contain the complete forward and backward functions.
- Lines 126 through 129 contain the forward, backward, and posterior calls.
- `torch_utils.cpp` lines 15 through 34 provide the original CPU initialization
  and deterministic settings. Its full-file SHA256 is
  `01a711350bea009e9c0b5d0a3578327c10ac65fa773c1cec1e1cafafd8b19482`.

The adapter supplies binary inputs, dependency includes, and raw output capture.
It applies the extracted initialization and an inference-mode guard. It uses
the same qualified CPU13 libraries as the production tensor backend. The build
receipt checks the backend binary and selected libraries, verifies used header
pins, and retains generated source, executable, input, and output hashes. Raw
commands retain exit status, UTC clocks, logs, and log hashes.

The final accepted gate is
`artifacts/crf/check-002/commands.json`. The detailed proof is
`artifacts/crf/check-002/native-attempts/attempt-20261023T023747.033251747Z/parity.json`.
Its exact comparison covers 66 fixtures, 264 tensor stages, and 1,857,280 float32
values, with zero differing bits. Fixtures cover state counts 4, 16, 256, and
1024; time lengths 1, 2, 9, and 32; ties; varied blank scores; score ranges up to
2000 in magnitude; and the supported maximum of 16384 states.

One fixture reuses the model unit's captured stock-weight HAC scores from a
short synthetic signal. It does not repeat model inference. The captured score
bytes have SHA256
`36229b983d66c9db8db986ea968f4fc679f81e6da8cf9bff5f75178128c35657`.
The source NTC shape `[1,16,1024]` has the same bytes as TNC `[16,1,1024]` because
`N=1`. `verification/crf/model-score-fixture.json` pins the scores and the model
unit's provenance and parity receipts. The guide output is accepted by the
existing Go beam decoder. This handoff check does not independently qualify a
whole stock caller.

The same gate passes ordinary tests, the cgo-disabled build, native rejection
and cancellation controls, race checks for lifecycle behavior, and `go vet`.
Controls reject modified original source, invalid dimensions and state counts,
multiple batches, NaN, infinity, unsafe values, foreign tensors, closed tensors,
and closed engines. A deterministic mid-scan cancellation leaves no partial
result and permits a subsequent successful computation on the engine.

`artifacts/crf/check-001` preserves an actual failed oracle build. The adapter
initially lacked the Torch namespace and parallel declarations needed by the
extracted initialization. Adding the dependency headers corrected that failure.
The extracted source statements remained unchanged.

## Reproduction commands

These commands run from the Go repository root. Paths to the original checkout
and existing model score capture are read-only. The oracle writes only beneath
the new repository's ignored artifact directory. Choose a fresh output name.

```sh
go test ./internal/crf ./verification/crf/...

go run ./verification/crf/check \
	-backend artifacts/tensor/attempt-004/libnanopore_tensor.so \
	-source /home/jake/Projects/nanopore_basecaller/software/dorado-src \
	-out artifacts/crf/check-next
```

The direct native gate is also available.

```sh
NANOPORE_TENSOR_BACKEND="$PWD/artifacts/tensor/attempt-004/libnanopore_tensor.so" \
DORADO_SOURCE=/home/jake/Projects/nanopore_basecaller/software/dorado-src \
CRF_ARTIFACT_DIR="$PWD/artifacts/crf/direct" \
	go test -count=1 -v ./internal/crf
```

Ordinary tests explicitly skip native parity when those variables are absent.
The maintained check command requires them and records separate contracts.
It performs one native oracle execution for the fixture set. Its race command
runs validation and cancellation checks without repeating the oracle producer.

This unit qualifies the CPU13 float32 scan algorithm over its supported input
contract. It does not qualify stock CUDA12.8 CPU compatibility, GPU execution,
batched scans, full read processing, biological accuracy, or performance.
