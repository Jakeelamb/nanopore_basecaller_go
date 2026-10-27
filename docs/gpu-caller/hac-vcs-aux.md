# HAC VCS CPU auxiliary packing

The candidate `Aux.PackLSTM()` computes the four initialized upload arrays in Go. It matches the pinned original CPU producer across the frozen 21-case matrix: 84 runs and 16,150,916 signed int32 words. This is a bounded CPU packing result. Caller wiring, GPU math, end-to-end basecalling, biological validity, and performance remain separate work.

## Production interface

`LSTMPacking` owns independent `InLayout`, `OutLayout`, `ForwardEncoding`, and `BackwardEncoding` slices. `OutLayoutCapacity` describes the larger destination allocation. The initialized OutLayout span has `N*L` words; its destination has `N*(L+1)`. Go returns the initialized span, while verification checks that the final N destination words remain untouched.

The implementation performs ordered subbatch partitioning, stable descending chain sorting, greedy chain merging, direct or interleaved placement, recurrence encoding, sentinel removal, and inverse layout construction. Root identifiers retain their raw upper 16 bits, including negative int32 encodings. Invalid profile, workspace, coordinate, and root identifier inputs reject before packing.

The method contains no stock executable, C wrapper, or native host packing call. The native GPU recurrence kernels remain outside this unit. This patch does not change tensor ABI, model, or caller files, and does not wire the arrays into the caller's H2D transport.

## Pinned CPU oracle

The source pin is Dorado `8b8fc5d36a9c0baab262a743cb175b5878e38ca6`. Original Koi archive SHA256 is `17cf3083c5db538f1e275b0a41ac19560b5986c143634e8a25540b9735f5805f`. The genuine object members are:

| Member | SHA256 |
| --- | --- |
| `utils_lstm.cpp.o` | `7598a9c75f9b4d3f3936e3871a31155a8e4b63c417017b7f5b2132db96051243` |
| `utils_lstm.cu.o` | `ce998c4411202c7c1c1d6240bad5677dfb8c1d69118355bf9affef6170045117` |
| `thread_pool.cpp.o` | `89cdf4ea0968f38e557787203b8a8b180dcac74c6df077bf89bf18d837b13f2d` |

The oracle calls the original public `host_lstm_preprocess` and genuine public thread-pool APIs. It does not guess the private helper's hidden aggregate-return ABI. The official `koi_thread_pool.h` supplies C linkage for the public header. A regression gate rejects the genuinely miscompiled old oracle object and requires the corrected object's three unmangled public symbols.

Only registration bookkeeping and H2D transport are replaced in verification. Each source span is captured before transport. Device queries, configuration, kernel lookup, and launch abort. The original CPU algorithm and pool computation remain intact. Static ELF inspection and actual process maps admit only the oracle, five independently pinned CPU DSOs, and the independently pinned read-only, non-executable loader cache. No CUDA, Torch, or driver DSO is admitted.

The CPU oracle has SHA256 `61d6321c64bdefe7afe76166c4f9d1a9af949134322dec051e8b8ae94e06a426`. Its successful build receipt has SHA256 `d3c202e924ac01f23ee6dabef036dc5fa8c34c2696dd27fe0568327dba9d69c4`. The separate corrected Go comparer has SHA256 `dfaa28f18ba25975267f8d7a2cbcd39dffa74081ff153fe58e7c8fce36b4a7aa`.

## Frozen comparison

`fixtures.json` was fixed before native payload output, retaining all original 20 cases and adding the source-admitted signed identifier case. Its SHA256 is `d4602d3fd2ac2e45515a949087b82fe50fe5a0a92e3dcffc66b82979d0c9e906`. Prepared input SHA256 is `5532c32a6b6ff13238e563a0ce4ec873a7018300f70abf5848195210c77dcf4d`.

The first geometry is `N=64`, `L=2050`, subbatch 32, and sizes `[128,256,2048]`. Result-order lengths are `2432`, `131200`, `131200`, and `131264`. Native copy order is OutLayout, ForwardEncoding, InLayout, BackwardEncoding. The remaining cases cover dense batches, boundaries around 32 and 64 roots, ties, permutations, odd and even lengths, tiny chunks, multiple ordered subbatches, and signed root identifiers. Each case runs with one and two pool workers and two workspace/destination fills.

The added signed case has `N=32832`, `T_out=1`, `L=4`, and 32832 one-step chunks. Their 196992 raw samples fit source batch capacity exactly. Sentinel accounting uses 98496 words. Native parts fit 42 three-word chains into `32*L=128` words, so 782 parts fit the 1026 available subbatches. Each part has fewer than 64 roots, the cumulative root end is below 65535, and the recovered 787968-word workspace requirement equals source capacity. Root IDs 32768 through 32831 set the sign bit. These bounds were recorded before native payload output.

Initialized regions were declared before observing output. With `A=S+2*num_chunks`, `B=N*L`, and `D=N*(L+1)`, source offsets in copy order are `[2*A+B+D, A, A+B, 2*A+B]`. The initialized workspace prefix is `2*A+2*B+D`. Unused backward inverse storage and remaining capacity must retain their fills. No mask, normalization, or expected tail value was inferred from native outcomes.

Every run compares all four signed upload spans, actual source-workspace agreement, exact copy order/count/kind/stream, successful statuses, source input nonmutation, guards, initialized workspace independence from fills, and untouched workspace/destination regions. Both worker counts and fills must match the same Go result. The signed case preserves 128 negative words in each forward and backward encoding per run, including forward `INT32_MIN`.

The complete matrix made 336 source copies. Original constructor and destructor counts were one fat binary registration, six function registrations, 24 variable registrations, one registration end, and one unregister. Error-string and unexpected device-call counts were zero. A changed Go word rejected with `InLayout[0] Go=67 native=66`. A wrong independent input pin rejected before any checker or oracle child. Focused map tests also reject corrupted cache bytes, executable/writable cache mappings, unknown executable mappings, and missing required ELF mappings.

## Evidence and limits

| Evidence | SHA256 |
| --- | --- |
| Full matrix runtime receipt | `a7fab91e7b4e2d18a30d81409bea18cb2fe25b056669fd5ec3a006208ff169e5` |
| Full matrix capture manifest | `68a5ad557b7dd8f8e0ae0653e6882e56ae8ee37c169dfc0a2730c0681de44305` |
| Full matrix cleanup | `f565a756a8097089d8cd65bdfd7e3a2e808050808956aa561c303e9f1dcdc85c` |
| Corrected retained-first observation | `70fb53e4a5370400a314ee12dc542eac78669e42c07881cbb4dead35bb6f760d` |
| Corrected retained-first cleanup | `07d8443fdf9589efe6b1b3d9c326cd22f986e478fc0ae69fcc95af82e54dd488` |

The matrix capture contains 1431 files under ignored `artifacts/hac-vcs-aux-observer/matrix-001/data`. All 1794 bound input pins and the original 71 first-capture files matched afterward. Cleanup confirms all 51 observed process/thread births gone, all five command/supervisor births observed, three child groups gone, and the cgroup absent. Resource limits were two CPUs, 3 GiB memory, zero swap, and 64 tasks with read-only nonblocking stock-lock admission. This was validation, not a throughput benchmark.

Failed Python setup, linker attempts, the original first observation failure, and old producer bytes remain retained. The original CPU first producer succeeded, but its original reader rejected `/etc/ld.so.cache` before numerical comparison. That failed receipt remains unchanged. A separate consumer worktree corrected only observation, using the cache pin already present in the independent compiler baseline. It qualified the saved first output without native replay. The later all-selection process validated the expanded frozen matrix.

## Reproduction and path bindings

The maintained saved-evidence command is a Go-only comparison. With coordinator resource admission, its exact successful matrix invocation is:

```sh
cd /home/jake/Projects/nanopore_basecaller_go_worktrees/hac-vcs-aux-observer-20261025
./artifacts/hac-vcs-aux-observer/host-001/hacvcsaux-observer \
  -unit compare \
  -evidence "$PWD/artifacts/hac-vcs-aux-observer/matrix-001/data/evidence.json" \
  -evidence-sha 68a5ad557b7dd8f8e0ae0653e6882e56ae8ee37c169dfc0a2730c0681de44305
```

The narrow native build used `verification/hacvcsaux/recompile.py` with `--compile-admitted`. Its immutable `recompile-plan.json` has SHA256 `5bc51df352a5938eed3059c956f40a7b5a13afc1bd252d6f38956e54ea062f08`. It compiles only the corrected oracle TU, retains the pinned transport object, checks the old and new public-symbol gate, links the three original objects and explicit CPU runtime, then performs static ELF inspection. It does not execute the oracle. Existing output directories are never overwritten; a new admitted build needs a recorded fresh output/unit suffix while retaining every input pin.

These verification controllers are task-specific and contain absolute bindings. They are not relocatable discovery tools. The producer root is `/home/jake/Projects/nanopore_basecaller_go_worktrees/hac-vcs-aux-20261025`; the corrected consumer root is the directory in the command above. Historical producer Go sources and all 1540 native-build inputs remain at their original paths. The consumer's native source copies must byte-match the producer's recorded native source pins. The independent compiler baseline is `/home/jake/Projects/nanopore_basecaller_go/artifacts/tensor/backend-closed-plan-005/plan.json`, SHA256 `b2d33b287e9135d85fc3ac6191de33c7aa9e70eda5a029f3a02b89a460b76d7a`. Native objects, GCC15, Koi headers, and libraries remain in the pinned stock qualification tree. Importing source into another checkout requires deliberately rebinding and repinning the verification paths; source import alone cannot reuse a stale successful receipt.

Focused production/fixture tests passed 47 named cases; observer controls passed ten named cases, plus vet and sealed CLI checks. These do not imply that the entire repository test suite or maintained CPU-core parity command ran in these worktrees. Those remain coordinator integration gates, alongside independent review and caller wiring. No merge or commit is included.
