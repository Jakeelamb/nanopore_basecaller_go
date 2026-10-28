# Tensor backend

Go owns tensor handles and caller decisions. The native library performs tensor
operations and reads Torch weight archives. It contains no model graph, recurrent
loop, CRF scan, decoder schedule, or read pipeline.

The CPU path uses float32 computation from ONT LibTorch
`2.9.0.0`, distributed in the CUDA 13.0 archive. Historical ABI4 CUDA half
acceptance is recorded below on one GPU and pinned runtime.
This does not qualify official stock CUDA 12.8 parity, full basecalling, biological
accuracy, or performance.
The original experiment, stock controls, dependencies, weights, and build remain
read-only.

`torch_utils.cpp` initializes one CPU thread, disables deterministic filling of
uninitialized memory, and enables deterministic algorithms. The backend matches
those pinned process settings once. The common CPU boundary then calls public
`at::init_num_threads()` once per entering OS thread, before tensor work. Go can
enter CGo on a thread other than the opener. Process-only initialization left
foreign threads able to create excess OpenMP workers during direct Tanh calls.
Each call also enters inference mode on its current OS thread, matching
`ModelRunner::call_chunks`. No environment workaround, permanent Go thread lock,
ABI change, or arithmetic change is required. The independent fixture producer
applies the same process settings.

## Ownership and types

`tensor.Open(path)` loads a separately built shared library and requires ABI 5.
The Go package needs Linux, cgo, the C compiler, and `libdl`. Building the package
does not require LibTorch headers or link flags. A disabled cgo build returns
`ErrUnsupported`. Ordinary tests explicitly skip the native gate when
`NANOPORE_TENSOR_BACKEND` is absent.

`Engine.NewFloat32(shape, data)` copies a float32 tensor into native storage.
`Engine.NewInt64(shape, data)` copies a separate `IndexTensor` into native storage.
An `IndexTensor` cannot be passed to float arithmetic. Both types expose `Shape`,
`Data`, and `Close`. Shapes and data are independent copies. Scalars have rank
zero and one element. Empty dimensions are supported. The maximum rank is 32.

Every result owns a new handle. Views retain ATen storage after their inputs
close. `Tensor.Close`, `IndexTensor.Close`, and `Engine.Close` release handles
explicitly and are idempotent. Closing an engine releases every remaining tensor.
Go garbage collection does not release native handles. Callers must close them.

One shared engine state serializes its calls and closure, including copied engine
values. Different engines reject one another's tensors. A native registry checks
engine IDs and tensor IDs independently of Go checks. IDs are not reused during
the library's lifetime. A native mutex protects those registries and synchronous
operations across engines. This is a correctness implementation, with no
parallelism or throughput claim.

The C boundary retains no Go pointer. Inputs are copied before a call returns.
Native exceptions return an error through the caller's bounded buffer, with no
thread-local error retrieval on a subsequent Go call.

## Operations

The float32 API includes the following methods.

- `MatMul`, `Add`, `Sub`, `Mul`, `Square`, `AddScalar`, and `MulScalar` provide arithmetic.
- `Rsqrt`, `Reciprocal`, `Sin`, and `Cos` provide float32 elementwise functions.
- `Clamp`, `Tanh`, `Sigmoid`, and `SiLU` provide elementwise activations.
- `Transpose`, `Permute`, `Contiguous`, `Flip`, `Reshape`, `Narrow`, `Concat`,
  `Stack`, and `Unsqueeze` provide tensor layouts.
- `Conv1d` accepts stride, padding, dilation, and groups. `Linear` accepts a
  weight. Both accept a nil bias.
- `Zeros` allocates float32 state. `Mean` and `LogSumExp` reduce one dimension and accept
  `keepdim`. `Softmax` operates along one dimension.
- `IndexSelect` accepts a float tensor, a dimension, and an `IndexTensor`.
  `ArgSort` accepts an `IndexTensor` and preserves ATen's ascending, unstable
  default, including its tie order.
- `ScaledDotProductAttention(q, k, v, maskShape, mask)` calls ATen with zero
  dropout, causal false, default scale, and grouped-query attention false.
  Both mask arguments nil omit the mask. Otherwise, the shape describes copied
  0/1 bytes, where 1 allows attention. ATen validates broadcast compatibility.
  The mask is not a general tensor handle and cannot enter float arithmetic.

`LoadWeight(path)` matches Dorado's `torch_utils/tensor_utils.cpp` vector archive
reader. It requires exactly one float32 tensor per file and maps storage to CPU.
It rejects int64, empty, and multiple-tensor archives. The native archive parser
is a dependency boundary, not a newly implemented file format.

ABI 1, ABI 2, ABI 3, and ABI 4 attempts remain immutable evidence. Current Go code deliberately rejects
those old libraries. There is one production ABI and no compatibility loader.

## CUDA half owners

ABI4 adds distinct `Device`, `Stream`, and `HalfBuffer` types. Their copied Go
wrappers share closure state under the owning engine mutex. Two devices opened at
the same ordinal remain different logical owners. Streams and buffers must belong
to the same engine and logical device owner.

`Engine.NewCUDADevice(ordinal)` opens an owner. `Device.NewStream()` creates an
owned nonblocking stream. `Stream.NewHalf(shape, bits)` copies raw half bits.
`HalfBuffer.View(shape, strides, offsetElements)` retains storage independently of
its parent handle. `HalfBuffer.Layout()` returns copied shape and stride slices.
`Stream.HalfBits(buffer)` returns packed logical half bits without a numeric cast.
The rank is 1 through 8. Dimensions are positive. View strides are nonnegative,
and checked reachable intervals cannot expand beyond their parent interval.

`Stream.GEMMHalf(a, b, out)` accepts rank-two row-leading matrices with final
stride 1. It requires A `[M,K]`, B `[K,N]`, and output `[M,N]`. Row strides and
M, N, K must fit signed `int`. A and B may alias. The output cannot overlap an
input's conservative storage interval. Disjoint views of one allocation are valid.

The production call matches the selected source endpoint in
`cuda_utils.cpp:377-394`. It reverses A and B for row-major interpretation and
calls `cublasGemmEx` with half alpha and beta, `CUDA_R_16F` input, output, and
compute types, and `CUBLAS_GEMM_DEFAULT_TENSOR_OP`. It does not select Dorado's
Torch or sm61 branch. Go owns graph decisions. This unit adds no model wrapper,
Koi adapter, asynchronous completion API, decoder, or neural inference endpoint.

Each CUDA operation sets its device and stream on the calling native thread.
A successful return includes stream synchronization. Native storage and host
staging remain owned through that synchronization. Failed execution poisons the
device owner. A failed drain retains pending storage. Failed closure preserves the
engine and library. Close may be retried. The native failure test injects drain
and close failures into the same drain-and-release helpers. A separate fake ABI4
library verifies that Go does not unload a library after failed engine closure.
These tests do not establish hardware-failure recovery.

`HalfBuffer.Close`, `Stream.Close`, `Device.Close`, and `Engine.Close` are explicit.
Device closure releases its streams and buffers after drains. Stream closure
leaves completed buffers valid for another stream of the same owner. Parent
closure leaves its views valid. No finalizer releases native resources.

CUDA failures use `DTypeError`, `DeviceError`, `StreamError`, `LayoutError`,
`CUDAError`, and `CUDAUnsupportedError`. Closed handles wrap `ErrClosed`.
An ABI5 CPU-only build wraps `ErrCUDAUnavailable` without loading CUDA.
`ErrUnsupported` retains its platform and no-cgo meaning. CPU signatures and
operation IDs are unchanged. The single ABI5 loader also requires the symbols
introduced in ABI4.

## Rebuild and verify

Run from the isolated Go checkout's repository root. Dependencies, source, and old libraries remain
read-only. Every output directory must be new and below `artifacts/tensor`.
Use Linux, Go 1.24 or newer with cgo, GCC, a C++20-capable `g++`, Git, and
`ldd`. The listed source, archive, extracted Torch tree, and HAC weight must already
exist locally and match the pins below. The builder emits ABI5 with CUDA disabled
when `-cuda-family none` is selected.

The builder, loader, and CPU and CUDA verifier metadata guards now require ABI5.
CUDA replay rejects historical ABI4 receipts. The first ABI5 CPU backend passed
all ten maintained checker commands before the thread fix. Fresh HAC and SUP CLI
runs and all ten maintained checker commands also pass with the fixed backend.
The native checker runs the corrected foreign-thread regression. See
[current CPU acceptance](cpu-abi5-acceptance.md) for the retained evidence.
Build a fresh output from current source and check that same output below.

```sh
TORCH_ROOT=/home/jake/Projects/nanopore_basecaller/software/source-build-qualification-20261023/dependencies/torch-2.9.0.0-ont-CUDA-13.0-linux-x64-cxx11-abi/libtorch
TORCH_ARCHIVE=/home/jake/Projects/nanopore_basecaller/software/source-build-qualification-20261023/archives/torch-2.9.0.0-ont-CUDA-13.0-linux-x64-cxx11-abi.zip
DORADO_SOURCE=/home/jake/Projects/nanopore_basecaller_go/external/dorado-source
HAC_WEIGHT=/home/jake/Projects/nanopore_basecaller/models/dna_r10.4.1_e8.2_400bps_hac@v6.0.0/0.conv.weight.tensor

go run ./verification/tensor/build -torch "$TORCH_ROOT" -archive "$TORCH_ARCHIVE" \
  -dorado-source "$DORADO_SOURCE" -weight "$HAC_WEIGHT" -cuda-family none \
  -out artifacts/tensor/abi5-cpu

CPU_TENSOR_LIBRARY="$PWD/artifacts/tensor/abi5-cpu/libnanopore_tensor.so"
```

Check that new CPU build with the same weight. The backend initializes each
entering OS thread itself. No `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, or
`OMP_THREAD_LIMIT` workaround is needed.

```sh
go run ./verification/tensor/check \
  -backend "$CPU_TENSOR_LIBRARY" -weight "$HAC_WEIGHT" \
  -out artifacts/tensor/check-abi5-cpu

go run ./verification/tensor/check -pure-only \
  -out artifacts/tensor/check-abi5-source
```

The CPU check records ordinary, no-cgo, native, direct C, race, and vet results.
For the optional historical ABI3 rejection control, add
`-old-backend /home/jake/Projects/nanopore_basecaller_go/artifacts/tensor/attempt-005-sup-abi3/libnanopore_tensor.so`
to the native check. That control runs in a separate process. The old library is
never a current caller backend.
The pure-only check does not load a native backend. Ordinary and no-cgo tests
cover all Go packages. These are source-readiness checks.

The builder verifies Dorado commit `8b8fc5d36a9c0baab262a743cb175b5878e38ca6`,
the source file pins, and ONT Torch archive SHA256
`8680441dbe3b0990fa7a99a70319affe25bc51f0cd6e105c3e07d522d7355929`.
Extracted Torch compile dependencies and linked libraries must match archive
members. Manifests retain exact source hashes, compiler identity, depfiles,
command arguments, clocks, exit status, raw stdout and stderr hashes, runtime
link closure, and output hashes.

Every substantial native build or native check opens the existing stock
`/home/jake/Projects/nanopore_basecaller/results/stock/host.lock` read-only and
uses `LOCK_EX|LOCK_NB`. It never creates or truncates that lock. A held lock
records `admission_blocked` with no compiler, producer, or consumer command.
The executable exits 75. `go run` reports `exit status 75` and itself exits 1.
The lock remains held until all admitted child processes have exited.

## CUDA build and runtime gates

The CUDA build compiles and links the backend and independent source oracle.
It creates raw input fixtures. It does not capture outputs, probe a device, or
initialize CUDA. The ONT distribution is labeled CUDA 13.0. Toolkit headers and runtime must be
major family 13. Their exact versions and bytes are separate pins. The installed
toolkit headers declare 13040. This unit does not relabel them as 13.0.

```sh
go run ./verification/tensor/build -torch "$TORCH_ROOT" -archive "$TORCH_ARCHIVE" \
  -dorado-source "$DORADO_SOURCE" -weight "$HAC_WEIGHT" \
  -cuda-family 13.0 -cuda-root /opt/cuda \
  -out artifacts/tensor/abi5-cuda
```

Runtime capture and replay require a coordinator release and the same lock.
The explicit `-runtime-admitted` flag records that release in the invocation.
Without that flag, the command retains `runtime_held` and launches no producer or
consumer. The CPU check refuses to load a CUDA-enabled build.

```sh
go run ./verification/tensor/cuda -runtime-admitted \
  -backend "$PWD/artifacts/tensor/abi5-cuda/libnanopore_tensor.so" \
  -out artifacts/tensor/capture-abi5-cuda

go run ./verification/tensor/cuda -runtime-admitted \
  -backend "$PWD/artifacts/tensor/abi5-cuda/libnanopore_tensor.so" \
  -capture "$PWD/artifacts/tensor/capture-abi5-cuda/manifest.json" \
  -weight "$HAC_WEIGHT" -out artifacts/tensor/replay-abi5-cuda
```

The runtime command retains fresh GPU state and compute-process bookends while
holding the lock. It rejects a competing compute process. It pins loaded Torch
and CUDA libraries, device UUID, driver, runtime, cuBLAS, visible-device mapping,
stream, pointer mode, math mode, atomics mode, and relevant environment values.
Capture stores this producer identity. Replay rejects changed GPU or driver
bookends, ordinal, visible-device mapping, or workspace environment before
consumer execution. Unset and empty environment values remain distinct. Runtime
library bytes remain pinned. Every consumer GEMM reports its cuBLAS modes, which
must match the source. Native diagnostics also compare actual driver, runtime, cuBLAS version, ordinal,
and UUID with the source identity.
Producer process mappings are printed after each batch. Prelaunch link closure
and actual post-batch mappings are checked. In-process pre-operation mapping
checks remain absent. Lazy dependencies are checked after mapping. This does not
guarantee that every lazy library can be verified before its first enqueue.

The independent oracle includes the exact extracted source function. The source
whole-file SHA256 is
`327df4b1f23552c51a57a515aa04aad8fe1fcc242fa94619026bc48763f675c1`.
The extracted block and its hash are saved. A logging shim only writes source
errors to stderr. Source process-exit failure behavior is preserved.

The corpus has 37 cases. One copies all 65,536 half payloads. Four GEMM cases
cover contiguous, rectangular, padded, and disjoint shared-storage layouts.
The other 32 cases use seed `0x48414c46`, M=N=16, K in 32, 64, 128, 256, and
eight mixed-magnitude finite inputs per K. The contiguous GEMM inputs include positive and negative finite values, positive
zero, and negative zero. The files save actual little-endian
half bytes, layouts, allocation sizes, offsets, and hashes. The parser rejects
wrong dtype, endian, metadata, spans, lengths, file kinds, and byte hashes before
consumer device allocation.

Two fresh producer processes each execute the complete positive corpus. One
separate producer process executes the 32 deliberate float32-compute mutants.
The mutant uses valid float32 alpha and beta with `CUBLAS_COMPUTE_32F`.
Exact repeatability, full parent guard preservation, and numerical mutant
separation have independent results. No difference among the 32 mutant cases
remains a discrimination gap. A CUDA failure or source-pin rejection does not
count as numerical separation.

Replay validates retained capture pins and builds one native Go test executable.
It records current consumer source hashes, Git head, executable hash, and stable
before-and-after identity separately from producer pins. It forces cgo on.
Required JSON test names, subcase counts, and absence of skips are checked
independently of process exit status. A no-tests exit zero cannot qualify replay.
A fresh process first checks CPU contracts against the CUDA-enabled library.
Another process checks CUDA ownership. One consumer process compares all saved cases. It compares every complete parent half bit, including gaps and
guards. Replay
does not rerun the source producer. A changed consumer requires a new comparison,
while source-identical captures remain reusable.

## Local evidence

The evidence below is historical ABI3 and ABI4 acceptance at its recorded source
and runtime pins. It does not qualify the current ABI5 build, CPU caller, or GPU
runtime. Current CPU evidence is recorded separately in
[ABI5 CPU acceptance](cpu-abi5-acceptance.md), including its passing post-fix checker.

The ABI3 library and its sibling manifest remain unchanged in the canonical Go
checkout. Its earlier acceptance remains evidence for ABI3 only. ABI4 rejects it.

`artifacts/tensor/attempt-abi4-cpu-001/manifest.json` retains the first blocked
admission. No native command ran. `attempt-abi4-cpu-002` built the CPU-only ABI4
library. `check-abi4-cpu-001/manifest.json` passed every listed CPU contract,
including real ABI3 rejection, all required CUDA loader symbols, native dtype and
owner rejection, bounded C error buffers, injected ownership failures, and the
Go close-failure library-retention control.

`artifacts/tensor/check-abi4-source-001/manifest.json` retains passed full-Go
ordinary and no-cgo tests, focused race tests, and vet. CUDA attempt 001 rejected
an overstrict exact-minor toolkit guard before CUDA compilation. The guard now
accepts major family 13 and records exact toolkit version 13040 separately.
CUDA attempt 002 was blocked at admission before any command. Both attempts are
retained. The host's other authorized native work restored the operational hold.

CUDA attempt 003 compiled and linked both the production backend and independent
source oracle. Its archive check then compared ZIP symlink text with followed
target bytes and failed. The repaired matcher verifies exact symlink text,
confined target paths, cycle rejection, and target bytes. Attempt 003 stays failed.
Fresh complete `attempt-abi4-cuda-004` passed all eight commands and pins 4,817
inputs and 4,576 archive members. It records every binary and depfile immediately
after compilation. Its backend SHA256 is
`d70b612b442e8449b2357d12284705bd82b30cdfa7316358a29ad1c800c9e3d7`.

`capture-abi4-cuda-001/manifest.json` has SHA256
`3c358a57509c94cf67a470b4a4e237e086eee2b34fd18c9cc6ad4cd4ce72ccf9`.
Two fresh producer batches repeated every complete parent bit for all 37 cases.
Every guard was preserved. The fixed float32 mutant differed in all 32 declared
precision cases, totaling 3,616 different half values. No search was expanded.

`replay-abi4-cuda-001/manifest.json` has SHA256
`15ca5c4a0cda12af18a430cf7763efa2bf78590636aa501892b9106d53c9beae`.
It reused the retained source outputs. CPU contracts against the CUDA-enabled
library passed 13 root tests and 66 subtests. CUDA ownership passed three root
tests. Source replay passed its root test and all 37 subcases. No required test
skipped. The standalone transfer check preserved all 65,536 half payloads.
The pinned Go test executable SHA256 is
`fb9c8839745e53b0a1b5133515a9cc07689151f1955d3c7d221a83782a315347`.
Its 32 current consumer source pins remained stable before and after execution.

The source and consumer used RTX 5070 Laptop GPU UUID
`807d7ae76a99c79512c223fc801d38e7`, driver `615.78.08`, runtime API version
`13040`, and cuBLAS version `130800`. Pointer, math, and atomics modes were zero.
Device and process bookends matched and showed no foreign compute applications.
These are actual runtime identities, distinct from the Torch archive's CUDA 13.0
label and toolkit header version `13040`.

`artifacts/tensor/acceptance-abi4-001/case-matrix.tsv` lists every case's shapes,
strides, offsets, allocation sizes, exact comparison, guards, and mutant
mismatches. The source and Go comparison covered 334,365 complete parent half
values per repeat. Shared parent allocations appear once per serialized role in
that count. `native-test-execution.json` lists exact successful test names and
zero skipped tests. The sibling acceptance manifest pins all retained receipts
and raw command outputs, including failures and blocked admissions.

`check-abi4-source-003/manifest.json` passed full ordinary and no-cgo Go tests,
focused race tests, and vet. Subsequent source changes only update this document.

The concurrency control exercises `HalfBits` racing copied-wrapper closure.
A successful GEMM followed by operand and stream closure is checked separately.
GEMM racing closure is not a separate executed case. Injected drain and close
failures exercise shared native helpers with dummy owners and Go library
retention. They do not inject a real CUDA hardware failure. Producer mappings
are observed after each batch, with prelaunch link and post-map hash checks.

This historical ABI4 unit qualifies synchronous half transfer, retained views, the selected
source half GEMM, and the listed controls on this device and pinned runtime.
It does not qualify model dispatch, Koi, full inference, basecalling, biological
accuracy, stock CUDA12 parity, or performance.
