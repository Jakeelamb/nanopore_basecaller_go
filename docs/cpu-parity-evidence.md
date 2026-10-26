# CPU port qualification

The implemented units are caller computations. They are not a complete
basecaller. Native model execution, guide and posterior generation, POD5 loading,
splitting, scheduler integration, and final caller output remain open.

The source is Dorado commit `8b8fc5d36a9c0baab262a743cb175b5878e38ca6`.
[The manifest](../provenance/dorado-source.json) pins source files and NVTX.
The new repository has its own source checkout under ignored
`external/dorado-source/`. The original experiment workspace is a read-only
control for this work and may advance independently.

| Unit | Native differential endpoint | Observed result |
| --- | --- | --- |
| Three chunk planners | Exact coordinates and native rejection cases | 4,995 cases pass |
| Standardised DNA pA scaling | Half bits and float32 metadata across all ADC inputs in six cases | 393,216 outputs match exactly |
| Stitching | Sequence, quality bytes and moves | 726 cases pass |
| Float32 CPU beam search | Sequence, quality, moves, states, final score and quality-data bits | 1,465 cases match exactly |

Scaling uses verbatim source formulas with scalar float32 arithmetic and compiler
binary16 conversion. It is not the full Torch/SIMD scaler. Stitching compiles the
untouched algorithm with a minimal read-container shim. Beam search compiles an
audited verbatim source slice without the Torch tensor adapter. The decoder uses
five native scalar libm functions while Go owns beam decisions and aggregation.
These qualifications do not establish CPU or GPU whole-caller parity.

The retained development BAM comparator controls pass with 32 HAC alignment
records and 33 SUP alignment records. Comparing those different streams rejects
them with exit one. This is a comparator rejection control, not an accuracy
comparison. Raw byte hashes and header hashes remain separate from canonical
record equality.

## Reproduction and retained attempts

```sh
./verification/setup-source.sh
go test ./...
go vet ./...
./verification/verify.sh --source "$PWD/external/dorado-source" --refresh
./verification/verify.sh --source "$PWD/external/dorado-source"
```

The mandatory verifier rejects skipped or missing native gates. It records source,
Go code, compiler, effective child targets, native numerical-library identities,
UTC clocks, exits, raw logs, and native fixture hashes. It rechecks inputs after
execution. Reuse requires a completed successful receipt and unchanged payloads.
It refuses unpinned compiler and dynamic-library overrides.

Local combined evidence is under `artifacts/parity/`. Attempt
`2db3948c8e8d8ce3-1353016698` was rejected because code changed during tests.
It remains failed evidence. Later successful attempts have separate directories.
Syscall execution traces are under `artifacts/traces/`. Traced clocks are not
performance measurements. Native inference and GPU jobs were not repeated.

## Corrections made during independent review

The first decoder sample accepted six intermediate float differences. Independent
review constructed a calibration that turned one difference into a wrong final
quality byte. The failing fixture and original attempts remain retained. The
float32 libm fix and threshold sweeps now require exact float bits. See
[the root-cause trace](decode.md).

Review also reproduced premature cache acceptance of a skipped native run and a
child-target identity collision. Successful receipts now follow native-gate
validation, cache lookup revalidates those gates, and actual child targets are
keyed. Backend extraction publishes its receipt and payload atomically after an
interruption test exposed an unrecoverable publication window.

Review is independent within the Sol model family. No cross-family review ran.
Synthetic fixture parity, CPU CI, prerequisite downloads, and compilation remain
separate from complete input processing, GPU behavior, biological accuracy, and
performance qualification. No speedup or model improvement is claimed.

## Portable tests and qualified host audits

The CPU CI workflow uses Go 1.24.x. The CGO0 testmain fixture declares the
module's minimum Go version, 1.24.0. Portable CPU13 tests use a hermetic source
tree and synthetic generator template with independently specified expected bytes.
They check exact generation, test-copy selection, and external test imports.
Changed whitespace or test entries reject at both selection and compiler
boundaries. Missing or foreign external files and imports also reject. These
fixtures do not audit the running Go generator. The production H08 compiler plan
keeps its Go1.27.3 requirement and exact generator-source and output checks.

Portable SUP compiler tests use inert ELF metadata fixtures to check exact DSO,
interpreter, and loader-cache inputs, lookup answers, and omitted or changed
inputs. The transitive closure fixture checks all three declared objects and
rejects an unknown mapping, changed pin, linker-script seed, or missing dependency.
HAC map fixtures check exact cache bytes and reject writable or executable cache
mappings, unexpected mappings, and missing required objects. No fixture compiler
or loader executes. A missing qualified GCC layout also rejects before command
logs or a compiler query can start.

Four host audits require `NANOPORE_QUALIFIED_HOST_TESTS=1`:

- `cgobuild.TestQualifiedGo1273GeneratedTestmainMatchesRealGo0Generator` compares
  source-derived bytes and package selection against a real CGO0 Go1.27.3
  generator in a private module. It retains the exact byte comparison and all
  selection and compiler rejection controls. An enabled audit fails on another
  Go version, missing qualified layout, or different generated bytes.
- `hacvcsaux.TestIndependentLoaderCacheIsExactSystemCache` checks the immutable
  compiler baseline and exact live `/etc/ld.so.cache` pin.
- `supgpu.TestQualifiedCompilerRuntimeStaticMetadata` checks the qualified GCC15
  tool, plugin, interpreter, and DSO family.
- `supgpu.TestCompilerRuntimeGatePrecedesFirstQuery` removes the compiler-only
  libisl input and requires rejection before any command logs appear.

Without that variable, those four tests report skips with a qualification
message. When enabled, missing files, changed pins, and runtime violations fail.
Portable subprocess controls require the enabled generator audit to fail on
Go1.24 version metadata and a Go1.27.3 tree with missing tools, before launch.
The opt-in changes only tests. Production compiler and runtime admission still
require their original qualified paths, identities, and checks.

```sh
GOMAXPROCS=2 GOMEMLIMIT=3GiB NANOPORE_QUALIFIED_HOST_TESTS=1 \
	go test -p=2 -count=1 -v \
	-run '^(TestQualifiedGo1273GeneratedTestmainMatchesRealGo0Generator|TestIndependentLoaderCacheIsExactSystemCache|TestQualifiedCompilerRuntimeStaticMetadata|TestCompilerRuntimeGatePrecedesFirstQuery)$' \
	./verification/hacgpu/h08/cgobuild ./verification/hacvcsaux ./verification/supgpu
```

These audits establish no native compilation, GPU execution, full
caller parity, or biological qualification. The full repository suite and
`verification/verify.sh` remain separate integration checks.

## Process disappearance during cleanup

CI run `40418303031` at `f8d9eb8bc211acd773a2c5e8e04f6c95dd6fbbfe`
failed when a descendant exited between listing `/proc` and reading its `stat`.
The read returned `ESRCH`, which the observer treated as a failure.
`observeProcessGroup` now omits both `ENOENT` and `ESRCH` reads. Its private
`scanProcessGroup` helper accepts the listed entries and stat reader so tests can
reproduce disappearance without scheduling a real process exit. The regression
retains a surviving member and excludes a different process group. Permission,
I/O, unknown read errors, and malformed stat records still fail.

```sh
env -u DORADO_SOURCE GOMAXPROCS=2 GOMEMLIMIT=3GiB GOPROXY=off GOSUMDB=off \
	go test -p=2 -count=1 -v -run '^TestProcessGroupScan' ./verification/supgpu
env -u DORADO_SOURCE GOMAXPROCS=2 GOMEMLIMIT=3GiB GOPROXY=off GOSUMDB=off \
	go test -p=2 -count=3 -v -run '^TestLoaderCleansDescendantsAfterLeaderExit$' ./verification/supgpu
```

The fake-peer cleanup test still requires SIGKILL evidence, a final empty
observation, rejection of surviving descendants, and preservation of unrelated
processes. Cleanup timeouts and runtime admission are unchanged. Earlier source
hashes are historical after this guard edit. Native and model acceptance remain
open. These focused CPU tests establish no GPU or native execution qualification.
