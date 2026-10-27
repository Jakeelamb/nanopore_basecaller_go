# Fixed HAC recurrence consumer

H05 runs the Go-owned seven-layer recurrence over retained H01 conv2 bytes and
verified H04 packed caches. It never invokes a model producer. The native ABI
performs one fixed C1024 K128 layer per call and keeps encoding null.

The production shape requires N positive and divisible by 64, T positive, an
I8 CUTLASS [T+5,N,1024] allocation, and an invocation-owned U8 temp. The byte
sizes are 512*(T+1)*N for down, 4*(T+1)*N for scales, and 2*N for workspace.
Go initializes three sentinel rows and zeros combined scratch once. The seven
source pairs are (-1,1), (+1,0), (-1,1), (+1,0), (-1,1), (+1,0), (-1,1).
The fixed boundary rejects other direction/parity pairs before enqueue.

Run the input-only check without acquiring a host lock or loading CUDA.

```sh
GOMAXPROCS=2 GOMEMLIMIT=3GiB go run ./verification/hacgpu/h05/cmd \
  -inspect \
  -capture /home/jake/.local/share/opencode/worktree/e3db79/H01/artifacts/tensor/hacgpu-capture-008/stages \
  -caches /home/jake/.local/share/opencode/worktree/e3db79/H04/artifacts/tensor/h04-cache-001/native
```

Native execution needs a coordinator release and a backend whose build manifest
pins the current hac_flstm.inc. Supply that backend and a fresh attempt path.

```sh
GOMAXPROCS=2 GOMEMLIMIT=3GiB OMP_NUM_THREADS=2 \
LD_LIBRARY_PATH= LD_PRELOAD= DORADO_MATMUL_FP16_MODE=CUBLAS \
go run ./verification/hacgpu/h05/cmd \
  -capture /home/jake/.local/share/opencode/worktree/e3db79/H01/artifacts/tensor/hacgpu-capture-008/stages \
  -caches /home/jake/.local/share/opencode/worktree/e3db79/H04/artifacts/tensor/h04-cache-001/native \
  -backend "$CURRENT_H05_BACKEND" \
  -out artifacts/tensor/h05-recurrence-002 -runtime-admitted
```

The harness acquires the canonical host lock. The observer runs after each
synchronous production layer. It saves full inout and full backing bytes before
checking the source-declared output rows and all initialized scratch bytes.
For this retained H01 case, WorkingMemory puts native temp at byte zero opposite
conv2 at the back. The candidate uses its own checked temp placement. The
comparison relocates these byte intervals without converting or masking scratch.
A scratch mismatch blocks acceptance and retains both source and candidate pins.

Use an external process-group timeout and retain argv, environment, clocks,
exit status and raw stream hashes. Run positive comparisons first. Only after
positive acceptance and a separate native release, add `-controls` to run
wrong-pair/parity-range rejection probes and a valid-pair cache-order mutation.
Invalid direction/parity kernel mutations are not executed. The first attempt
predating this guard timed out without stage observations, so its failing stage
and cause remain unknown.

Host acceptance uses `GOMAXPROCS=2 GOMEMLIMIT=3GiB go test ./...`. The recording
checks establish source order, exact spans, one scratch zero, cache snapshots,
shape/overflow rejection and fail/cancel stopping. Host tests and this isolated
recurrence consumer do not establish whole-model inference, GPU performance,
basecalling accuracy, or biological acceptance. Variable chunk encoding cannot
be passed through the fixed primitive.

## Retained acceptance on 2026-10-25 UTC

The final bounded consumer passed on the retained N64, chunk192, T32 case.
Every production layer matched H01 on its prospectively declared output rows.
All seven complete initialized scratch intervals, each 1,089,920 bytes, matched
its retained native backing without exclusions. Native wrong-pair and parity2
rejections and the valid-pair cache-order control separated.

- Final receipt is `artifacts/tensor/h05-controls-006/receipt.json`.
- Raw streams, argv, environment, clocks and exit0 are pinned in
  `artifacts/h05-host-20261024/native-wrapper-006/receipt.json`.
- The retained final executable is `artifacts/tensor/h05-controls-006/consumer`,
  SHA256 `53c053fa519ef1fbf766834a072fcc0a0b1af0e935c7716fc2fbdf27a39238a6`.
- The shared backend is `artifacts/tensor/attempt-h05-shared-build-20261024-002/libnanopore_tensor.so`,
  SHA256 `93f0137ff94d850f012db1fce93bde5c855749ed45c48949d82a2d3df3ad2b80`.

`native-wrapper-001` timed out at 180 seconds without stage observations or a
retained executable. Its failing operation and cause remain unknown.
`native-wrapper-002` stopped at a consumer compile error before native loading.
`native-wrapper-003` passed positive output/scratch parity.
`native-wrapper-004` passed positive parity and wrong-pair rejection, then failed
the consumer's expected error type for parity2. The native rejection itself was
retained. The corrected typed-error observation has a host regression check and
passed in `native-wrapper-005`. These attempts remain separate.

`native-wrapper-006` adds retained mutant observation without changing model or
kernel code. Its `control-cache-order-inout.bin` has SHA256
`cd8a0747d3e5323e193ff5723f147b5ecd3907ba26ee98e9a7bd86e2848d2690`.
The embedded control stage compares 2,228,224 bytes after excluding source rows
`[[1,1],[35,2]]`. Its compared digest is
`b48e6fce81290aba680a56f9af32e82b359fee69c74dd8d7b085936821a15a5f`,
while the native compared digest is
`dbea207a043217d1d6e449a0674a734424b4e2ab68c35af4bb72f423eb4f6d75`.
The observer re-reads and verifies the raw payload and comparison record.
Targeted host tests reject payload and comparison-digest tampering. Stdout and
persisted receipt are byte-identical, including the single final clock.
