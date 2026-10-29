# Bounded ABI5 CUDA compiler evidence

The source-current backend compiled successfully and remains unexecuted.
This is bounded executed compiler proof. Independent native compiler-unit review
is required. Runtime admission, CPU13, and GPU qualification remain open.

## Source and artifact pins

`W` is `/home/jake/Projects/nanopore_basecaller_go_worktrees/abi5-closed-backend-20261028`.
`B` is `/home/jake/Projects/nanopore_basecaller_go/artifacts/pstack-herdr-polish-20261028`.
The source is branch `feature/abi5-closed-backend-20261028`, commit
`c10a08925660180181fc0469d34d7e8d6ccc0de9`. Paths below expand these two roots.
All digests are SHA256.

| Artifact | Path | Digest |
| --- | --- | --- |
| Current Go0 producer | `W/artifacts/tensor/abi5-closed-controller-001/controller.json` | `6d47fe42585efd2c7614260b075f5af616955a2100e489e8613abad649f3168e` |
| Current closed executable | `W/artifacts/tensor/abi5-closed-controller-001/closed` | `7226114e880cd7269b484c6b34ada4180c1196f29391a5da68ab8e1a2d113037` |
| Current maintained executable | `W/artifacts/tensor/abi5-closed-controller-001/maintained` | `a514291777f78613290eae044c6fe398cfbd1e9153c723b513df2ee2933826ce` |
| Declarative recipe | `B/abi5-closed-recipe-host-001/stdout` | `175abad5c66a8981613ea4ffc3b3cc660bb0c8aed8ff78f0ed7e16b32e6cfdf1` |
| Current plan | `W/artifacts/tensor/abi5-closed-plan-001/plan.json` | `eb2adbe65460e0401071d610dd2c14866392048da3b02df7c1e67bb090e9b913` |
| Current build | `W/artifacts/tensor/abi5-closed-build-001/build.json` | `b26d8ec45d6ceba3e515c5f26380a8e1087cf871d9103ab4bb8c37d136216b3d` |
| Unexecuted library | `W/artifacts/tensor/abi5-closed-build-001/libnanopore_tensor.so` | `bceb29a210ba094b5237a79a107739700b34b121e0da3d49df23b95890210328` |
| ABI5 header | `W/internal/tensor/abi.h` | `ee5fef3ca2b3052d80a3502bde8250327a439ec6c8ef8642c62d9a526017153b` |
| Backend source | `W/verification/tensor/backend.cpp` | `79d320d5c751cee5b6142e8f586900c772d861525e943cc2a90befad8e308f91` |
| Common thread initialization | `W/verification/tensor/runtime.h` | `402ba8331aa31fba265ec91397524ef8c87cfea2d11256362031a7d15d6b0136` |

The fresh producer has 954 inputs and immutable snapshots, and five Go0 commands.
The old qualified `backend-closed-controller-005/closed`, identified historically
as `eca5...`, served only as the bootstrap. It is not the source-current producer.
The recipe retains the earlier declarative options. The fresh plan binds 2,027
inputs, 23 compiler queries, one backend unit, 23 prospective runtime ELFs, and
eight initial archive members.

The compiler uses GCC15, ONT Torch2.9, and Koi0.7.5 with asset family `13.0`.
The actual CUDA header contract is `ToolkitVersion=13040`, CUDA13.4.
`plan.json` records the exact distribution paths in `Recipe.Options`, the selected
compiler ELF domains, path bindings, runtime inventory, and these asset pins.

| Selected asset in `plan.json` | Digest |
| --- | --- |
| `Recipe.Options.Archive`, `torch-2.9.0.0-ont-CUDA-13.0-linux-x64-cxx11-abi.zip` | `8680441dbe3b0990fa7a99a70319affe25bc51f0cd6e105c3e07d522d7355929` |
| `Recipe.Options.Koi` plus `/include/koi.h` | `8b4115c2e92ebf1acf1db2aa76bdc9b3a22768cb595b92363f88e44ff92d3b88` |
| `Recipe.Options.Koi` plus `/lib/libkoi.a` | `17cf3083c5db538f1e275b0a41ac19560b5986c143634e8a25540b9735f5805f` |

## Retained commands and bounds

Each directory below contains `receipt.json` with exact argv, UTC clocks, exit
status, and hashes of raw stdout, stderr, and cleanup observations. The receipt
argv is the rerun specification. A rerun requires fresh output directories,
fresh unit names, updated dependent path/hash pairs, and controller admission.
These references authorize no execution.

| Stage | Exact command source under `B` | Entrypoint and ownership |
| --- | --- | --- |
| Go0 controller | `abi5-closed-controller-host-001/receipt.json` | Bootstrap `closed -unit controller -host-admitted -go-root /usr/lib/go -out NEW`; controller acquires the host lock |
| Recipe | `abi5-closed-recipe-host-001/receipt.json` | Fresh `maintained -plan-only` with all recorded asset flags; declarative, no compiler or host-lock acquisition |
| Plan | `abi5-closed-plan-host-001/receipt.json` | Fresh `closed -unit plan -profiles backend` with every independent path/hash pair; metadata-only, no compiler or host-lock acquisition |
| Compile | `abi5-closed-compile-host-001/receipt.json` | Fresh `closed -unit compile -compiler-admitted -plan PLAN -plan-sha SHA -out NEW`; compiler acquires the host lock |

All four outer launches use CPUQuota200%, MemoryMax3G, swap0, Tasks64,
KillMode=control-group, and TimeoutStopSec5. Controller, recipe, and plan have
180-second outer bounds. Compilation has a 600-second outer bound and
180-second bounds per compiler program. The controller and compiler acquire the
existing `/home/jake/Projects/nanopore_basecaller/results/stock/host.lock`
through `runner.Acquire`, using O_RDONLY and LOCK_EX|LOCK_NB.
The root controller owns admission and all native execution.

The compile receipt records exit0 and unchanged source. Its retained cleanup
reports inactive/dead, MainPID0, and an empty ControlGroup. The build records
cpu.max `200000 100000`, memory.max `3221225472`, memory.swap.max0, and pids.max64.
All 26 commands exited0. The 7,086-input build retains one object, two depfiles,
and one link map. Dependency sets from `-M` and `-MD` match. Status is
`built_outputs_unexecuted` and `RuntimeHeld=true`. Compile clocks support no
speed or performance claim.

## Static authentication and rejection controls

The historical Go0 helper `8ce3aa0e7a5a13d69a59b30f2d7579a298a3b56f578ab057f4223ab5a7fde653`
passes typed `closedbuild.ReadArtifact` authentication in
`B/abi5-closed-artifact-positive-host-001/`. Root verified its 970 immutable
Go-source snapshots. This is a metadata helper, not a current runtime consumer.
Its wrapped stdout digest is `4fcd6518092eeb11e314b2741c4146f4df7dc3be36eaed2c5e1c294ae1df606f`.

The source-current Go0 metadata CLI also passes in
`B/h08-current-backend-wire-before-001/`. Its executable is
`/home/jake/Projects/nanopore_basecaller_go_worktrees/cgo-build-proof-20261028/artifacts/tensor/h08-current-metadata-reader-before-001/h08-host`,
digest `0ad544ab4c9cc223486a287379834c29ae1b4481366914bcfe3e41b388a25403`,
with 1,090 pins. Its raw artifact stdout digest is
`135d19cd827e921302d71917ccf4df1cd7573f7162a64e4d8444769de1d3c389`.
The raw payload equals the historical helper's `.Backend` payload.

| Control under `B` | Observed result |
| --- | --- |
| `abi5-closed-unadmitted-host-001/` | Exit75, `admission_blocked`, before attempt creation or native work |
| `abi5-artifact-omitted-snapshot-host-001/` | Exit1, `incomplete historical producer custody` |
| `abi5-artifact-source-binding-host-001/` | Exit1, `historical snapshot/source binding differs` for `verification/tensor/closedbuild/controller.go` |

The two artifact controls select genuinely rehashed controller/plan/build
families in `B/abi5-artifact-{omitted-snapshot,source-binding}-001/` and retain
the original library. These are adversarial metadata fixtures, not producer
executions. Their exact mutant pins and stderr hashes are in their receipts.
The source-binding rejection occurs before snapshot-byte verification.

The pre-correction CLI emitted the raw artifact while the compiler planner
expected `nanopore-h08-backend-proof/v1`. The genuine before regression fails
at that planner gate in `B/h08-envelope-genuine-before-001/`. The corrected
Go0 CLI `74068fbd14ea3498a3fed8bfe49ab863a16f2a2f7335df9e32b4c7fe53062449`
passes the same producer/planner test and a changed-schema rejection control in
`B/h08-envelope-genuine-after-001/`. Its source correction has separate review.
The new producer digest does not invalidate these native artifact pins or
require native recompilation. Retained stdout was not rewritten.

## Remaining qualification

ABI5 describes the pinned header's static compile intent. No library version or
runtime ABI query, output execution, driver/GPU call, model call, or genuine
CPU13 run occurred. CPU-only `b11d22...` acceptance does not transfer to this
library. A Torch CUDA13 candidate does not establish stock CUDA12.8 parity.
H08, graph, CPU13, GPU caller, biology, and timing remain open. The production
GPU registry remains empty. Historical failed005 stays failed, and old `52ae...`
stays unexecuted. All failed attempts and raw receipts remain retained.
Weights, archives, signals, native libraries, and raw receipts stay outside Git.
Stock source, stock experiments, and HG005 remain unchanged.
