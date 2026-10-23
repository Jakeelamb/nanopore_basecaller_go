# CPU tensor backend

Go owns tensor handles and caller decisions. The native library performs tensor
operations and reads Torch weight archives. It contains no model graph, recurrent
loop, CRF scan, decoder schedule, or read pipeline.

This development backend uses CPU float32 computation from ONT LibTorch
`2.9.0.0`, distributed in the CUDA 13.0 archive. It does not qualify the official
stock CUDA 12.8 runtime, GPU execution, full basecalling, accuracy, or performance.
The original experiment, stock controls, dependencies, weights, and build remain
read-only.

`torch_utils.cpp` initializes one CPU thread, disables deterministic filling of
uninitialized memory, and enables deterministic algorithms. The backend matches
those pinned source settings once when its runtime opens. Every C call enters
inference mode on its current OS thread, matching `ModelRunner::call_chunks`.
The independent fixture producer applies the same initialization.

## Ownership and types

`tensor.Open(path)` loads a separately built shared library and requires ABI 2.
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

- `MatMul`, `Add`, `Mul`, `AddScalar`, and `MulScalar` provide arithmetic.
- `Clamp`, `Tanh`, `Sigmoid`, and `SiLU` provide elementwise activations.
- `Transpose`, `Permute`, `Contiguous`, `Flip`, `Reshape`, `Narrow`, `Concat`,
  `Stack`, and `Unsqueeze` provide tensor layouts.
- `Conv1d` accepts stride, padding, dilation, and groups. `Linear` accepts a
  weight. Both accept a nil bias.
- `Zeros` allocates float32 state. `LogSumExp` reduces one dimension and accepts
  `keepdim`. `Softmax` operates along one dimension.
- `IndexSelect` accepts a float tensor, a dimension, and an `IndexTensor`.
  `ArgSort` accepts an `IndexTensor` and preserves ATen's ascending, unstable
  default, including its tie order.

`LoadWeight(path)` matches Dorado's `torch_utils/tensor_utils.cpp` vector archive
reader. It requires exactly one float32 tensor per file and maps storage to CPU.
It rejects int64, empty, and multiple-tensor archives. The native archive parser
is a dependency boundary, not a newly implemented file format.

ABI 1 attempts remain immutable evidence. Current Go code deliberately rejects
those old libraries. There is one production ABI and no compatibility loader.

## Rebuild and verify

From the Go repository root, use read-only paths to the original dependencies.
These commands write only beneath the new repository's ignored `artifacts` path.
Choose fresh output names for every attempt.

```sh
TORCH_ROOT=/home/jake/Projects/nanopore_basecaller/software/source-build-qualification-20261023/dependencies/torch-2.9.0.0-ont-CUDA-13.0-linux-x64-cxx11-abi/libtorch
DORADO_SOURCE=/home/jake/Projects/nanopore_basecaller/software/dorado-src
TORCH_ARCHIVE=/home/jake/Projects/nanopore_basecaller/software/source-build-qualification-20261023/archives/torch-2.9.0.0-ont-CUDA-13.0-linux-x64-cxx11-abi.zip
HAC_WEIGHT=/home/jake/Projects/nanopore_basecaller/models/dna_r10.4.1_e8.2_400bps_hac@v6.0.0/0.conv.weight.tensor

go run ./verification/tensor/build -torch "$TORCH_ROOT" -archive "$TORCH_ARCHIVE" \
  -dorado-source "$DORADO_SOURCE" -weight "$HAC_WEIGHT" -out artifacts/tensor/attempt-004

go run ./verification/tensor/check \
  -backend artifacts/tensor/attempt-004/libnanopore_tensor.so \
  -old-backend artifacts/tensor/attempt-001/libnanopore_tensor.so \
  -weight "$HAC_WEIGHT" -out artifacts/tensor/checks-005
```

The builder verifies the full archive SHA256
`8680441dbe3b0990fa7a99a70319affe25bc51f0cd6e105c3e07d522d7355929`.
It matches the extracted compile dependencies and selected CPU libraries against
archive member bytes. Its JSON manifest records compiler identity, command
arguments, exit status, UTC clocks, raw log hashes, source hashes, output hash,
and resolved runtime library hashes. No download occurs. Failed attempts retain
manifests and logs.

The native test also hashes the actual loaded backend, `libtorch_cpu.so`, and
`libc10.so` found in `/proc/self/maps`, then checks their build manifest pins.

The check command records the current Git head and exact source hashes. It runs
ordinary tests, the cgo-disabled path, real native tests, and the race detector.
The real gate exercises allocation, copying, arithmetic, views, convolution,
linear operations, reductions, gather ordering, native argsort ties, weight
loading, and closure. Its rejection controls include malformed shapes, buffers,
dtypes, operation arity, dimensions, stale IDs, foreign owners, missing libraries,
missing symbols, and the actual retained ABI 1 library.

The native fixture producer writes independent ATen outputs for stable
`LogSumExp`, `Softmax`, `IndexSelect`, and `ArgSort` cases. Tests compare float32
bits and int64 values exactly. A supplied real HAC convolution weight is compared
against every bit of an independent native snapshot. These checks qualify the
operation and serialization contracts. Source-linked model algorithm parity
belongs to the model and decoder verification units.

For the smaller acceptance commands, use the following forms.

```sh
go test ./internal/tensor
NANOPORE_TENSOR_BACKEND="$PWD/artifacts/tensor/attempt-004/libnanopore_tensor.so" \
  go test -count=1 -v ./internal/tensor
```

Half precision, CUDA tensors, streams, asynchronous kernels, and GPU allocation
remain unimplemented. The runtime library pins describe the fresh test process;
embedding another LibTorch distribution in that process needs separate runtime
qualification.

## Local acceptance

The final library is retained at
`artifacts/tensor/attempt-004/libnanopore_tensor.so`, with SHA256
`ef23a4ee0ec9f4ca499b3106aaf83ffe597277bde83e27892e352bc8c4282837`.
Its build manifest matches 4,553 distribution members against the verified
archive. `artifacts/tensor/checks-005/manifest.json` retains the final ordinary,
cgo-disabled, native, and race checks. `go vet ./internal/tensor
./verification/tensor/...` also passes.

`artifacts/tensor/rejected-archive-001/manifest.json` retains an expected failed
builder invocation with a wrong archive hash. The command exits 1 before
compilation. Earlier successful attempts and checks remain preserved and are
superseded by the final initialization contract and test sources.
