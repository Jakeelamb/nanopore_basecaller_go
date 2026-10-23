# CUDA ABI4 review

An independent native Sol-family verifier passed the bounded half-transfer,
ownership, and source-GEMM unit. The 44 final source files are pinned in
[`cuda-abi4-reviewed-source.sha256`](../provenance/cuda-abi4-reviewed-source.sha256).
Integration into `port/go-caller` preserved every reviewed file hash.

The full verdict is retained locally at
`artifacts/tensor/independent-runtime-review/verdict.md`, SHA256
`bcff73a8f405633790862a1ee68b65ec30fde4ccdbeda11a0ab56bb2508d176e`.
Its exact commands, UTC clocks, raw outputs, and file pins are retained beside it.
The final implementation handoff is
`artifacts/tensor/handoff-abi4-final-001/manifest.json`, SHA256
`f73f1f8cc57a7b64b72d3b270855520af6e1a100a2c3d35cde8111bb36b1ee9b`.

The reviewer independently checked native build and archive pins, all source and
consumer identities, raw command receipts, actual test JSON, runtime diagnostics,
and complete parent bytes. Two source processes repeated all 37 cases exactly.
Go replay matched 334,365 serialized parent half values, including guards and
shared-storage roles. All 65,536 raw half patterns transferred exactly. All 32
valid float32-compute mutants differed numerically, with 3,616 changed values.
A deliberate padded-output guard corruption failed the independent audit.

The CUDA-enabled library also passed 13 CPU contract roots and 66 subtests in a
fresh process. CUDA ownership passed three roots; source replay passed all 37
subcases. No required test skipped. The reviewer rehashed all 65 retained command
receipts, including failed and blocked attempts. No inference or GPU computation
was repeated by the reviewer.

Earlier source reviews found real verification gaps. The fixes require named
native tests to execute, pin the current consumer independently, compare actual
GPU/runtime identities, include zero GEMM operands, and run CPU regression on
the CUDA-enabled artifact. Their failed controls and subsequent source PASS are
retained under `artifacts/tensor/independent-source001/` and
`artifacts/tensor/independent-source003/`.

See [the tensor contract](tensor-backend.md) for maintained build and replay
commands. Acceptance uses the original nonblocking host lock and the observed
RTX 5070 device/runtime. It does not establish other GPUs, stock CUDA12 parity,
full model graphs, Koi kernels, basecalling, biological accuracy, or performance.
Failure injection used shared helpers with dummy owners. A separate
GEMM-versus-close race and real device failure were not exercised. Producer maps
were collected after each batch. The historical dynamic-interpreter hash was
not pinned, so this is not a complete-host reproduction claim.
