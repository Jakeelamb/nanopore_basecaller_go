# Current H08 Cgo compiler plan

This package prepares and executes the fixed H08 CGO1 compiler boundary in
separately admitted phases. `plan` performs only static preparation. `queries`
and `build` require explicit native admission, the existing read-only nonblocking
host lock, effective 2 CPU/3 GiB/swap0/64-task limits and a sealed CGO0 interposer.
The current H08 `Consume` hold is unchanged.

The source/host phase produces a prospective plan, source/tool/resource pins and
immutable snapshots. It is not a CGO1 build receipt. The concrete CC/tool
interposer and phased native execution path now exist and are exercised with a
sealed Go fake compiler only. Independent source review and a separate native
grant still precede real CC use. Plan placeholders describe the driver-generated
sealed invocation; they are never passed to a child. No historical six-tool
CGO1 proof is reused.

## Shape and ownership

`Options` selects the current checkout, Go1.27.3, a private work root, immutable
closedbuild Plan005 compiler data, and the accepted real typed backend002 proof.
`Prepare` creates `Plan`. `ValidateSpecification` rejects a missing plugin,
substituted CC/CXX or omitted C link resource before a child could be created.
`GeneratedSet` binds a Cgo producer receipt, required output roles, live bytes,
path bindings and immutable snapshots. `CheckGeneratedLive` is the prospective
pre-child check; it does not replace native compiler custody.

`Controller` authenticates the actual forced CGO0 interposer build, before/after
Go source lists, complete source/tool snapshot binding and executable. `Session`
binds one live supervisor identity/cgroup to the immutable plan and controller.
`Lease` binds one selected Go cgo/link invocation before it starts. The lease is
inherited by that invocation's descendants, and is not a mutable current-package
registry. `operation` and `Receipt` retain before-child manifests, immutable
inputs, native and Go-tool argv/clocks/raw streams/exits, process identities,
generated roles, output identities and the whole-build rejection marker.

The fixed H08 target currently has only `internal/tensor` and `runtime/cgo` as
Cgo packages. The selector applies Go's actual build constraints with Cgo enabled
using filesystem-only `go/build.ImportDir(..., IgnoreVendor)`. It admits the
fixed module and GOROOT, rejects new Cgo packages, CXX/Fortran/SWIG/pkg-config,
and rejects directory/glob embeddings. Explicit nested regular files such as
`native/audit.c` must resolve within their canonical package directory. It selects the Go/Cgo/C/H/S/
syso/embed/module transitive source set. A future admitted actual CGO1 package
list must match this prospective selection; this plan does not invent evidence
that cmd/go has already selected or compiled it.
That comparison excludes compiler builtins `C` and `unsafe`; the latter's
documentation-only Go file is not compiled. This selector records 140 source
packages, including the two Cgo packages, rather than claiming an unnormalized
cmd/go package-list count.

Compiler closure comes from immutable closedbuild Plan005 data and is rebuilt
with `Compiler.Check`/`nativeproof`. The selected `gcc-15` is the named
`gcc-driver` tool with its own retained and rederived ELF domain; its interpreter,
DSOs, loader cache, GCC LTO plugin and BFD plugin family are explicit. `g++-15`
is pinned separately, even though this fixed consumer contains no CXX TUs.

The 25 named resource pins include both PIE and non-PIE CRT variants, GCC
libraries, libc scripts/providers, exact empty `libpthread.a`/`libdl.a` and the
loader. Existing002 resource pins must match. New non-PIE prospective choices are
source-current candidates, not claimed query observations. Twenty-three resource
queries plus `--version` and `-### -x c -c -` form the explicit first native
metadata phase; every expected file is declared before those queries.

The Cgo build does not link `libnanopore_tensor.so`, Torch or CUDA. It links the
current `loader.c` shim and C runtime resources. It pins backend002's real typed
proof and selected output as a historical producer anchor, without traversing
its unconsumed NN/runtime closure or requiring old Go originals to be current.
The future genuine H08 consumer must still run `closedbuild.ReadArtifact` and
its current runtime policy before loading that backend. Native backend rebuilds
and source NN oracle reruns are absent from this unit.

## Why an interposer

A plain `CGO_ENABLED=1 go build -x` followed by hashing misses inputs used before
the first visible TU compile. Go1.27.3 source defines the relevant calls:

| Source | Native calls |
| --- | --- |
| `/usr/lib/go/src/cmd/go/internal/work/buildid.go:206` | CC `-### -x c -c -` identity probe |
| `/usr/lib/go/src/cmd/go/internal/work/exec.go:2470` | CC flag compile/link probes, including expected failures |
| `/usr/lib/go/src/cmd/cgo/gcc.go:1805` | macro, diagnostic classification and DWARF/type compiles |
| `/usr/lib/go/src/cmd/cgo/util.go:20` | transient `cgo-gcc-input-*.c` is written and removed around CC |
| `/usr/lib/go/src/cmd/go/internal/work/exec.go:3011` | final Cgo generated C/H/Go roles |
| `/usr/lib/go/src/cmd/go/internal/work/exec.go:3268` | `_cgo_main.c`, dynimport link and generated import Go |
| `/usr/lib/go/src/cmd/link/internal/ld/lib.go:1434` | external linking, flag probes and optional response file |

Two options were considered. A dedicated fixed-H08 CC/tool interposer can pin
derived files before native calls while preserving Go's real Cgo computation.
A post-build `-x`/directory audit cannot recover transient inputs or prove what
was declared before compiler invocation. The former is recommended. A manual
replacement of cmd/go's package graph/packing/link orchestration would be a
larger independent build system and is unnecessary.

The interposer must support these finite roles, not arbitrary commands:

1. Run the 25 reviewed metadata queries under the complete current compiler and
   resource closure. Authenticate canonical query answers before translation.
2. Intercept the Go `cgo` tool invocation to bind its package, source/flags and
   output directory before generation. Intercept every CC request, including
   temporary C files, snapshot generated input bytes before the child, then run
   same-option `-M` before actual `-MD` where available. Macro/DWARF probes require
   success; diagnostic classification deliberately may return nonzero. Expected
   native statuses and stderr are retained per role, never described as all-zero.
3. On Cgo-tool return, snapshot the required generated C/H/Go set before cmd/go
   resumes. Handle dynimport as a separate cgo-tool call. Compile actual original
   C/S and generated TUs, require dep equality, freeze objects before each link,
   and close trace/map against predeclared CRT/script/archive/provider pins.
4. Build the current executable with a fresh cache, `-a -p=1`, explicit closed
   environment, external linking and pinned Go tools. `GOMAXPROCS=1` also makes
   cmd/cgo's source-level parallel queue serial. Its flags and generated inputs
   remain derived by the pinned Go/Cgo tools. The actual sequence/count is not
   claimed before that admitted build.

The implementation also performs an actual admitted CGO1 `go list -deps -json`
through the closed CC environment and compares all selected source members
before building. Go tools are selected by pinned path and hash. Unknown tool or
native flags, foreign C roots, unplanned include paths, outside/private-parent
output escapes and foreign external linkers reject before their child. Selected
metadata support probes use empty stdin; any nonempty replacement rejects.

Go support/link probes that compile C and link it use a retained explicit probe
object. Its compile has `-M`/`-MD` evidence, and its bytes are declared before the
separate link. This avoids treating a deleted GCC-internal temporary object as
an externally pinned linker input. The original Go request and transformed
native argv are both recorded; compiler/linker flags retain their roles.

`cmd/link`'s `linkerFlagSupported` uses only `extld[0]`, dropping a command prefix.
The exact immutable session is therefore also inherited in the closed
environment. A bare helper invocation from that selected Go linker enters the
same CC gate. A CXX request is forbidden for this fixed Cgo source set.

Unknown requests must fail the entire build even when Go tolerates a failed
feature probe. A broker rejection cannot be mistaken for an unsupported compiler
flag and cached as a valid result. Every actual native call needs its own exact
argv, input-before-child pins, raw clocks/exits/logs and the 180-second process
group limit. The existing outer canonical lock/cgroup supervisor remains a
separate ownership/resource boundary; these profiles do not provide runtime
first-frame, target audit-DSO, lazy-loading or GPU supervision.

## Plan-only command

Build this CLI with CGO disabled after coordinator host admission, then run it
inside the maintained bounded host supervisor:

```sh
CGO_PLAN -host-admitted \
  -compiler-plan artifacts/tensor/backend-closed-plan-005/plan.json \
  -compiler-plan-sha b2d33b287e9135d85fc3ac6191de33c7aa9e70eda5a029f3a02b89a460b76d7a \
  -backend-proof artifacts/backend-artifact-host-20261025-004/genuine-backend002.stdout \
  -backend-proof-sha fcf013464f48129fede814f3f57d895ec90913805d503586e143795c8d1af972 \
  -out artifacts/tensor/NEW_H08_CGO_PLAN
```

The independent input pins must use canonical absolute paths; the relative paths
above show the repository locations and must be expanded by the supervisor.
The default plan mode writes `plan.json`, `snapshots.json`
and a receipt identifying its actual executable, argv, clocks and effective
limits. Actual compiler queries, Cgo generation/build, target execution, real
CPU13/genuine-weight fixtures, shared audited runtime guard integration and
driver/PTX/GPU/HAC graph acceptance are separate gates.

## Source/host observation

`artifacts/h08-cgo-plan-host-20261025-001` records four top-level tests (nine
named passes, no named skips), vet, a fresh CGO0 planner build, actual pure-static
preparation, and admission75 rejection before input reads/output creation. The
effective caps are CPU200%, memory3 GiB, swap0 and64 tasks; each child has a
180-second process-group cap inside a600-second control-group supervisor. The
canonical read-only nonblocking lock is held by the supervisor for Go commands
and by the plan CLI for preparation. The completed cgroup is absent.

The planner executable is
`31a54980fb975015612adffa40de132352a0bf0adea591a433077476f26d7382`.
The actual plan is `artifacts/tensor/h08-cgo-plan-001/plan.json`, SHA256
`fab1314d0aef33a28f4cadd4e0878841819ddc9f4c86bf322b0e5b351da485dd`.
It pins1,065 inputs and snapshots them, including37 compiler inputs and the
three-file GCC C-driver ELF domain. Its nine Go executable tools are pinned,
with Go configuration/include files also declared. The current host tool/source
selection has1,056 inputs, immutable copies, actual Go argv/raw streams/exits,
and unchanged source membership/bytes before and after preparation.

Effective host controls reject omitted LTO/BFD plugin, changed CC path, missing
pthread/DL/CRT resources, missing generated header, changed generated C bytes,
unknown generated role/package and a generated symlink escaping the private
work root. They exercise structural/file guards; no real CC interposer was run.
No C/CXX query, Cgo generation, CGO1 Go build, output load or NN execution ran.

## Concrete interposer source/host evidence

The corrected implementation is frozen in
`artifacts/h08-cgo-interposer-host-20261025-003`:11 top-level tests,21 named
passes, no named skips, vet, a forced current CGO0 interposer build, static
preparation, genuine controller read and effective rejection of a rehashed
controller with an omitted snapshot. Unadmitted `queries` and `build` each exit75
before opening the supplied missing plan or creating an output directory.

The interposer is
`88cf20a75f9d8808f353eea04aa02f1675f1a0a5159d8f6d6e3fa7e911e454aa`;
its controller custody JSON is
`3724a481dc9fe99a775557f874722fd1142f8947b1c797090e6dacd3abffa36c`.
The current controller has998 selected Go/tool inputs with complete immutable
snapshots. The full host selection has1,062 inputs and unchanged source
membership/bytes. Plan004 is
`bc59a640f212c9e0e354375bb0f0e8f49d0f7c9f889c6196845b4356e29552e5`.

Fake compiler execution uses a sealed CGO0 Go helper, never a C compiler. Tests
exercise real before-child manifests/immutable snapshots, `-M`/`-MD` equality,
an intentional diagnostic exit1, unsupported-flag exit1, empty-stdin checks,
source changes between preflight and actual compile, omitted plugins/CRT,
substituted CC, foreign source roots, changed generated bytes, whole-build
poison after a rejected probe, incomplete final proof and child group cleanup.
Go-tool output controls include an absent private output with existing parent
(accepted), absent parent, a symlinked escaping parent, foreign output and
foreign external linker (rejected before any child).

Host001 retains an unused-import compile failure. Host002 retains the earlier
passing implementation; the path/extld correction has its own fresh003 freeze.
No old plan, controller or source capture was rewritten. Actual compiler/Cgo
profiles remain unexercised until independent review and a separate grant. An
unexpected real Go request will fail closed and preserve its exact evidence;
it must not be silently widened or automatically retried.

After review, the separately granted metadata phase uses this form:

```sh
INTERPOSER -mode queries -native-admitted \
  -plan PLAN004 -plan-sha bc59a640f212c9e0e354375bb0f0e8f49d0f7c9f889c6196845b4356e29552e5 \
  -controller CONTROLLER003 -controller-sha 3724a481dc9fe99a775557f874722fd1142f8947b1c797090e6dacd3abffa36c \
  -out artifacts/tensor/FRESH_NATIVE_METADATA
```

`-mode build` additionally requires the independently pinned successful
`-queries`/`-queries-sha` receipt. Its linked CGO1 executable remains unexecuted.
The real Cgo build, native artifact review, shared runtime guard/current runtime
policy, CPU13/genuine-weight fixtures and GPU/HAC phases remain separate gates.

## Independent-review corrections

The corrected tool freeze is
`artifacts/h08-cgo-interposer-host-20261025-006`:20 top-level tests,50 named
passes, no named skips, vet, a forced CGO0 interposer build, genuine controller
read, rehashed omitted-snapshot rejection and query/build admission75 before
input reads or output creation. Its executable is
`00148698da29038a2bedd1241cc45d7d8eaa2bee870435f876772dd7eadb9de8`;
controller JSON is
`921f4fae90a8eb80df73a53e76852aebf506f57da04bfbb01d5ba7f248a1302e`.
It has999 controller inputs and1,064 host inputs, with immutable snapshots and
unchanged selected source bytes. The cgroup finished inactive/dead and is absent.

Final output symlinks reject, including those with a private parent. Existing
regular private outputs remain valid. Transient C and link-object paths must be
private both lexically and after canonical resolution. Original native requests
are reclassified before every child and again after snapshot/receipt writes.
Tests cover escaping final symlinks, derived-file/directory aliases and an output
replacement before spawn; a private directory alias is accepted. Private0700
work directories and trusted selected tools provide the writer boundary. These
checks do not claim an atomic filesystem sandbox against a hostile concurrent
process with the same user identity.

The fixed Go pack `r` profile now checks its positional archive output. Config
inputs retain path bindings; `-importcfg` closes referenced private archives,
`-embedcfg` closes all `Files` targets against selected inputs, and unobserved
inline `-importcfg=` is rejected. Output/config/embed/pack controls stop before
a fake compiler could fork.

The receipt reader binds selected command argv/environment, complete before-child
input sets, immutable input bytes, native role classification, transformed probe
steps, reparsed `-M`/`-MD` sets, link trace/map inputs, exact `-o` identity and
successful Cgo generated-file producers. Intermediate output snapshots retain
bytes before Go packing/build-ID updates can mutate archives. Effective controls
reject rehashed wrong commands, foreign environments, omitted plugins, changed
resource answers/dependencies/link maps, unrelated generated producers and
modified final outputs.

The new plan sets the metadata-only Go linker flag `-buildid=`. Pinned Go1.27.3
`src/cmd/go/internal/work/gc.go:644` appends user linker flags after its default
build ID. `src/cmd/go/internal/work/buildid.go:704` returns without rewriting
when the default temporary ID is absent, explicitly recognizing this override.
This lets the reader require identical final external-link, Go-link and copied
consumer bytes. Existing plans are unchanged.

Host004 retains the failed positive query fixture that included an unrelated
generated-source record. Host005 retains18 top-level/42 named passing tests,
vet and a fresh tool build; current target preparation then rejected the new
`native/audit.c` embed. The explicit nested-file correction is covered in006.
Target plan preparation is deferred until the separate H08 guard/CPU13 source
integration freezes. The retained006 tool can prepare that plan without a Go
build replay. No real compiler query, Cgo1 build, loader or GPU ran.

`ReadBuild(Pin) (Receipt, error)` remains the H08-command compiler-artifact reader API;
`Receipt.Output` is a selected executable pin and `RuntimeHeld` stays true.
The accepted006 target was fixed to the ordinary H08 command. The explicit CPU13
profile below extends it without a generic target override. A future D02 command
still needs a separate source-bound profile and reviewed generated/native request
set. Changing a target string is insufficient. Compiler proof supplies no CPU13
or runtime authorization.

## Fixed CPU13 test-binary profile

`Options.Role` and `Plan.Role` select only `H08Command` (`h08-command`) or
`CPU13Tests` (`cpu13-tensor-tests`). An omitted preparation role selects the
ordinary command; every prepared plan and execution receipt carries an explicit
role. `ReadBuild`/`ReadH08Build` accept only the ordinary command and
`ReadCPU13Build` accepts only the maintained test binary. A role mismatch rejects
before opening its plan. An accepted compiler artifact still carries
`RuntimeHeld=true`; it supplies no fixture or runtime authorization.

The CPU13 command is fixed to `go test -c ... ./internal/tensor`, output
`tensor.test`. Its single add-only Go overlay maps
`internal/tensor/h08_phase_main_test.go` to the pinned
`verification/hacgpu/h08/cpu13/testmain.go.txt`. The destination must not exist.
The overlay imports H08's external `tensor_test` TestMain wrapper; it copies no
maintained test bodies. The build does not execute a test binary or choose the
genuine/fake runtime selector. Those selectors remain the existing eleven
audited genuine tests and two separately qualified fake-loader tests.

Static source selection includes the actual CGO-enabled tensor test files and
their imports, the wrapper imports, and Go's four testmain dependencies. Go's
test-copy descriptions are admitted only for selected packages that depend on
tensor, plus the exact external wrapper and generated main. Actual admitted
`go list -deps -test -json -overlay=...` selection must match those files and
variants. Ordinary H08 source selection admits none of these synthetic packages.

The plan pins Go1.27.3's test loader, package descriptions, test driver and list
generator. It expands the exact pinned `testmainTmpl` using the maintained test
function declarations. Added benchmarks, fuzz targets, examples, external tests
or another TestMain require a separately reviewed profile change. The generated
`_testmain.go` must match those bytes before its Go compiler child; the operation
snapshots retain them and the final reader requires that compiler operation.
The add-only overlay JSON is also snapshotted before the first actual Go command
and included in every native/tool before-child input set.

Tests compare this derivation with a real CGO0 `go list -test` generator in a
private fake module. This exercises Go's test variants and entry-point bytes
without a C compiler, Cgo generation, test execution or backend loading. Separate
controls reject wrong roles, extra/changed overlay mappings, replacement of an
existing source, foreign package argv, changed/missing generated bytes and a
foreign external TestMain. Current ordinary and test plans remain held until
the profile/API source review and target freeze are accepted.

The concrete profile/tool source handoff is
`artifacts/h08-cgo-profile-host-20261025-003`. Typed per-package checks cover37
required roots and72 named passes:23 retained cgobuild roots and12 retained H08
roots from001, the corrected generator root from003, and the fresh fixture helper.
The001 package failure remains failed; the retained projection explicitly
excludes only its failed generator subtree and aggregate failure after checking
that no other failure/skip exists. All other source inputs and the test-function
prefix remain unchanged. Vet, the forced current CGO0 tool build, controller read
and admission/omission controls passed. Its executable is
`8d2a48a9e1324f2af411155591475da1c612bfbe7712ed231439c0a94b8d0e19`;
controller JSON is
`7cbaa950ced3576f6a43d49892a5643e5c77fe54573d463c503fb4cede6d8e65`.
The freeze has1,019 controller inputs and1,144 host inputs; the completed cgroup
is absent.

Host001's failed helper removed its temporary child diagnostics; that observation
gap remains explicit. Host002 retained a successful Go0 generator and exact
generated bytes/variant comparison, then failed a unit argument using the cache
filename instead of `_testmain.go`. Host003 corrected only that argument/control
and repeated that one root, with persistent helper raw logs and actual/expected
generated source. It did not repeat the other passing roots. No target plan,
real CC query, Cgo1 build, backend load or GPU phase ran.

## Metadata identity stdin correction

The first real H08 metadata attempt retained25 successful selected GCC queries,
then its reader failed on the exact planned `-### -x c -c -` identity request.
The producer handles that metadata request before stdin capture; the reader
incorrectly applied the compile-stdin requirement first. Its receipt remains
failed at `artifacts/tensor/h08-cgo-command-queries-001/receipt.json`; it is not
current build authority. CPU13 queries were not started.

The persistent correction exempts only requests reclassified as exact planned
metadata, rejects any retained metadata stdin, and preserves the nonmetadata
stdin requirement. Narrow fake-Go controls check those three cases and reject
a nonexact metadata spelling. A diagnostic-only in-memory view validates the
retained25 operations with the corrected reader; the original failed status and
bytes remain unchanged and the public reader still rejects it.

`artifacts/h08-cgo-stdin-fix-host-20261025-002` records the passing narrow control,
diagnostic, vet, fresh CGO0 interposer/controller and both plan-only preparations.
Host001 preserves a negative fixture stopped by an earlier path guard;002 changes
that fixture to reach the intended metadata-stdin guard. The fresh interposer is
`bd9d777844189f82bbf5721eaef2f8d4f8f79ea95ffa6fc4c9272bcebf97b248`;
controller JSON is
`3b84e3d86b69ed510fd0040bc8b8d669a507e11459bb7556c1c5b1f43acfd6c8`.
Both new plan002 input sets change only `evidence.go`;37 compiler pins,25 resources
and25 prospective queries remain unchanged. Since the target Go source changed,
each new plan needs separately granted metadata queries and CGO1 compilation.
No metadata query replay or native target work occurred in this correction.

## Fixed Go orchestration budget

The first current CGO1 build passed actual source selection and retained86
accepted operations, then the top-level Go build hit the shared180-second tool
watchdog while rebuilding dependencies. Its failed receipt remains at
`artifacts/tensor/h08-cgo-command-build-001/receipt.json`; it has no output.
CPU13 compilation was not started.

Only the fixed top-level `Execute` Go build or `go test -c` command now receives
a1,800-second orchestration budget, within a separately admitted2,100-second
outer cgroup. Its helper derives argv/environment from the typed fixed role,
plan and session and rejects another phase or target. Go list and every
individual Go tool, C compiler and linker keep the180-second watchdog. There is
no CLI budget override. All before/after pins and immutable snapshots remain.
The focused host controls inspect actual context deadlines without waiting and
reject foreign orchestration callers before output files can be opened.

The new source requires a fresh sealed CGO0 tool/controller, both current plans
and separately granted metadata/build phases. Historical query and failed build
receipts are unchanged; no retry is implied by the budget correction.

## Go invocation recipe corrections

Build002 passed source selection and retained231 accepted operations, then failed
at two previously unexercised Go requests. Its raw receipt and all233 process
records remain under `artifacts/tensor/h08-cgo-command-build-002`; no output was
produced and CPU13 was not started. The cohesive source audit is
`artifacts/h08-cgo-recipe-audit-20261025-001/audit-final.md`.

Go1.27's `gccToolID` prepends `BuildToolexec` to its compiler identity request.
The tool interposer admits only the exact build-phase nested request
`[sessionDriver,-###,-x,c,-c,-]`, through the existing CC metadata gate. A pinned
wrapper record retains the full original argv, selected driver and source-defined
environment values before the compiler child. The reader authenticates those
same bindings. Other controller/tool paths, languages and arguments remain
rejected. The driver is not a general selected Go tool.

Cgo translation uses each package's exact absolute source paths, matching
Go's `mkAbsFiles` for runtime/cgo and tensor. Both packages retain their fixed
flags and private generated-file ownership. The source-derived environment
helper is shared with the reader: translation clears CGO_LDFLAGS, dynimport keeps
its source-defined value, and the trimpath linker clears GOROOT. Incoming values
are checked; the wrapper does not inherit arbitrary ambient overrides or silently
repair unexpected ones.

An immutable session selection record binds the actual successful Go-list command,
raw stdout/stderr and the fixed role's DefaultGODEBUG. It is written before the
Go build starts and snapshotted before the Go linker. Only its exact inline
`-X=runtime.godebugDefault=...` assignment is admitted. CPU13 additionally permits
the exact separate `-X testing.testBinary=1` pair. Missing, duplicate, foreign or
wrong-role assignments reject before the linker; the final reader checks the
same producer and environment proof. There is no generic linker assignment API.

The initial same-option `-M` remains dependency-discovery metadata, matching the
maintained closed builder. Known source/object/tool inputs are pinned before it;
newly discovered headers are pinned and snapshotted before actual compilation,
then exact M/MD equality and afterward pin checks are required. No retroactive
header-before-discovery claim or whole-system inventory is introduced.

These are source corrections with focused sealed-Go-fake controls. A fresh CGO0
tool/controller and plan004 preparations need their own host evidence; genuine
queries and another CGO1 attempt still require separate grants. Historical query
and failed-build artifacts stay unchanged. A plain `go -n` is not used as a
no-compiler boundary because its compiler/tool identity probes can execute.

## Full compiler-prefix support probes

Build003 passed the wrapped compiler identity, both Cgo translations/dynimports
and native C operations, but a Go support trial reached the TU seed guard. Go's
`ccompile` passes the complete `compilerCmd` prefix into `gccSupportsFlag` for
three trials: `-fdebug-prefix-map=a=b`, `-ffile-prefix-map=a=b` and
`-frandom-seed=1`. The existing bare-CC recipe did not cover that form. The
rejected request and the deliberate stop of the owned Go-build process group
remain in `artifacts/h08-cgo-build-host-20261025-006`. It produced no final output;
the partial native observations do not qualify the failed build.

The full-prefix family is bound to an existing captured runtime/cgo or tensor
translation context. It requires the exact source-directory include and
Linux/amd64 PIC/m64/pthread prefix; optional compiler-supported flags retain Go's
source order. A work prefix map must name that context's object directory and
`/tmp/go-build`. The tail admits only the three trials above, fixed `-O2 -g -c
-x c -`, empty stdin and `/dev/null` output. The producer and generated type-file
pins become before-child inputs, and the reader derives the same context.

This adds no general compiler flag list. Native TU action seeds still require20
characters; literal seed1 remains rejected for a TU. Source-grounded fake controls
cover all three trials for both contexts and reject foreign directories/maps,
unknown/reordered prefixes, other trials, changed stdin/output and absent Cgo

## Retained compiled proof

The retained reader authenticates saved build005 as data through the common
`validateReceiptBody` used for successful compiled receipts. Its private profile,
`saved005-final-reader-v1`, admits only the selected original producer tuple,
handoff, outer command, cleanup and failure stderr. It checks frozen producer
sources, command clocks, unchanged input manifests, all operation files and
snapshots, and historical process records. It does not change the failed receipt.
The strict raw receipt APIs still reject that receipt.

The common validator checks these corrected relationships:

- Exact query answers bind individual resources. The selected prior query receipt
  supplies 16 noncanonical answers, seven special spellings and nine others.
  Each answer must match its prospective Query and Resource, the selected command,
  authenticated producer and before-child snapshots. Trace and map tokens are
  checked before cleaning. No directory alias or current filesystem resolution
  supplies an implicit binding.
- A split probe binds its successful compile to the separate link through the
  explicit probe object, dependency evidence, exact transformed argv and snapshots.
  Compiler flags retain their compile role. Linker flags retain their link role.
  An expected unsupported probe remains an expected exit, with its diagnostics.
- Exactly one native `go-link` and one `go-tool-link` share the selected lease,
  command context, output target and authenticated snapshots. The fixed Linux/amd64
  Go1.27.3 linker profile derives the 32-byte Go FIPS HMAC update. The entire final
  file must match after that update, with every other byte unchanged.

Both Go-tool input validation and the final-output profile accept the selected
external driver as its bare absolute path or exactly `strconv.Quote(driver.Path)`.
Foreign drivers, ambiguous prefixes and other quoting reject. The final profile
requires an executable external link and an effective empty `-buildid`.
It rejects `-B`, `-fipso` and unsupported flag forms. The pinned linker SHA256 is
`97df419b7cf8f4bb8e76c3623b8a1d371c6818533c4be04f0a3563ff6169843d`.

The observation writer and reader reuse `deriveQueryResources`,
`finalOutputTransition` and authenticated operation metadata. Readers rerun the
complete common body and compare every serialized proof summary. No alternate
link or ELF validator, producer replay or bulk hash cache supplies acceptance.

## Select compiled artifacts

The public API is:

```go
ObserveCompiled(requestPin Pin, out string) (Pin, error)
ReadH08Artifact(evidence, selection Pin) (Artifact, error)
ReadCPU13Artifact(evidence, selection Pin) (Artifact, error)
func (a Artifact) Output() Pin
```

Pass both an independently selected evidence Pin and selection Pin. There is no
optional selector or fallback. `Artifact` keeps its state and origin private and
exposes only the selected output Pin. It conveys no runtime admission.

A successful selection names the expected role and complete producer tuple before
calling the unchanged strict successful reader. An observed selection names the
preselected request and the external post-exit Execution. Matching self-authored
hashes do not prove independent selection. The selecting controller must carry
both digests through its authenticated request. The retained profile supports
ordinary H08 only. The CPU13 reader does not make that profile a CPU13 artifact.

All new JSON requires exact case, mandatory fields, no null, unknown or duplicate
keys, and one JSON value. Every Pin is
`{"Path":"/absolute/path","SHA256":"64hex"}`.
The request has this exact structure (symbolic pin values below must be replaced
with independent selections):

```json
{
  "Schema": "nanopore-h08-observation-request/v1",
  "Role": "h08-command",
  "Retained": {
    "Producer": {
      "Receipt": {"Path":"/selected/receipt","SHA256":"RECEIPT_SHA"},
      "Plan": {"Path":"/selected/plan","SHA256":"PLAN_SHA"},
      "Controller": {"Path":"/selected/producer-controller","SHA256":"CONTROLLER_SHA"},
      "Queries": {"Path":"/selected/queries","SHA256":"QUERIES_SHA"},
      "Output": {"Path":"/selected/output","SHA256":"OUTPUT_SHA"}
    },
    "Handoff": {"Path":"/selected/handoff","SHA256":"HANDOFF_SHA"},
    "Outer": {"Path":"/selected/outer","SHA256":"OUTER_SHA"},
    "Cleanup": {"Path":"/selected/cleanup","SHA256":"CLEANUP_SHA"},
    "FailureStderr": {"Path":"/selected/stderr","SHA256":"STDERR_SHA"},
    "Profile": "saved005-final-reader-v1"
  },
  "ObserverController": {"Path":"/selected/current-controller","SHA256":"OBSERVER_SHA"},
  "ObserverRoot": "/absolute/selected/checkout",
  "AlgorithmReferences": [
    {"Path":"/usr/lib/go/src/cmd/link/internal/ld/fips140.go","SHA256":"f214648c1a2ec04ea5483812ceec13cec96aa1a1acd9f95d2882b6373ef432c5"},
    {"Path":"/usr/lib/go/src/cmd/internal/obj/fips140.go","SHA256":"25fba80147d77aa812720b5c4ec71304b1547491ff59f0221c12a3c504236137"}
  ],
  "OutputDirectory": "/absolute/selected/checkout/artifacts/tensor/fresh-observation"
}
```

The historical producer pins remain fixed by `retained.go`. AlgorithmReferences
are new authenticated inputs, separate from historical Plan.Inputs. Keep them
separate from the observer's source/tool closure. ObserverController must seal
`observation.go` and `cmd/main.go` under ObserverRoot, plus the complete selected
CGO0 source, tools, list/build receipts, snapshots and actual executable.

## Seal and run the observer

Use the [CGO0 seal instructions](tools/README.md) and
[independent launcher instructions](tools/OBSERVE.md) for exact launch mechanics.
The sealer runs the maintained `check-controller` and a rehashed missing-snapshot
rejection. Mocked glue controls do not prove a real observer execution.

Preselect and hash the request before startup. Use a fresh output directory.
The writer checks its actual cwd and executable against the selected controller,
reserves that directory, runs the full retained proof, writes `observation.json`
and emits its Pin. Failed reserved directories remain as evidence. A retry needs
a new request and output directory. The original failed receipt stays unchanged.

The actual observer argv is:

```text
EXE -mode observe-compiled -observation-request REQUEST -observation-request-sha REQUEST_SHA -out OUT -host-admitted
```

Use cwd=ObserverRoot and this complete cleared environment in this exact order:

```text
CGO_ENABLED=0 GOTOOLCHAIN=local GOWORK=off GOPROXY=off GOSUMDB=off GOENV=off GOMAXPROCS=2 GOMEMLIMIT=3GiB
```

The root launcher holds the canonical stock lock O_RDONLY with LOCK_EX|LOCK_NB
through cleanup. The observer never reacquires it. The external deadline starts
before interpreter startup. Bounds remain outer 600s, service 575s and work 550s,
with at most 2 CPU, 3 GiB, swap 0 and 64 tasks. The CLI checks host admission,
CheckRoot and Limits before reading the request.

The root independently witnesses the actual executable, argv, cwd, environment,
PID, start identity and owned cgroup. After exit 0, stdout must be exactly one
observation Pin. The root verifies PID/start absence, an empty owned group and
an absent cgroup before selection. It retains raw logs, UTC RFC3339Nano clocks,
and this command's exit 0 and raw four-property dead-unit result:

```text
systemctl --user show UNIT.service -p ActiveState -p SubState -p MainPID -p ControlGroup
```

The properties are exactly ActiveState=inactive, SubState=dead, MainPID=0 and
ControlGroup=, each once and newline-terminated. Order is immaterial.
UNIT.service matches the recorded cgroup's final component. Cleanup's raw command
has environment_overrides=[]. Readers authenticate these selected records without
live process checks or CheckController on their own executable.

The post-exit wire types below are schematic notation, not JSON evidence.
Process is `{PID:int,Start:string,Cgroup:string,Group:true}`. PID is the positive
group leader, Start is decimal /proc/PID/stat field22, and Cgroup is the raw
single cgroup-v2 line with an optional final newline.
```text
Cleanup = {
 Schema:"nanopore-h08-observer-cleanup/v1", UTC:string, Unit:string,
 Process:{PID:int,Start:string,Cgroup:string,Group:true},
 PIDAbsent:true, GroupAbsent:true, CgroupAbsent:true, RawCommand:CommandReceipt
}
Execution = {
 Schema:"nanopore-h08-observer-execution/v1",
 Request:Pin, Observation:Pin, ObserverController:Pin, Executable:Pin,
 Process:Process, Cwd:string, Command:CommandReceipt, Cleanup:Pin
}
ObservedSelection = {
 Schema:"nanopore-h08-observed-selection/v1", Request:Pin, Execution:Pin
}
SuccessfulSelection = {
 Schema:"nanopore-h08-successful-selection/v1", Role:BuildRole,
 Producer:{Receipt:Pin,Plan:Pin,Controller:Pin,Queries:Pin,Output:Pin}
}
CommandReceipt = {
 command:[string], environment_overrides:[string], started_utc:string,
 finished_utc:string, exit_code:int, stdout:string, stderr:string,
 stdout_sha256:string, stderr_sha256:string
}
Observation = {
 Schema:"nanopore-h08-compiled-observation/v1", Request:Pin, RuntimeHeld:true,
 NativePreimage:Pin, Output:Pin,
 Proofs:{Aliases:{rawToken:Pin}, Operations:[{
   Operation:Pin,Role:string,Status:string,Before:Pin,OutputSnapshot:Pin,
   Bindings:{rawToken:canonicalPath}
 }],ChecksumOffset:uint64,Checksum:[32 bytes],Ranges:[[start,end] x4]}
}
```

The root writes and hashes Cleanup, then Execution, then ObservedSelection after
real exit and cleanup. The observer creates none of these root records.
Pass the Observation Pin and independently selected Selection Pin to
`ReadH08Artifact`.

## Retained acceptance and limits

The [retained-observation evidence](../../../../docs/gpu-caller/h08-retained-observation.md)
records the accepted complete 297-operation proof, real CGO0 seal, witnessed
observer and selected-reader result, with their pins and preserved failures.
The saved executable embeds the old reader and remains unexecuted. It cannot
serve as the current runtime consumer. All six consumer call sites now require
independent evidence and selection pins; see the
[consumer contract](../../../../docs/gpu-caller/h08-selected-consumers.md).
Source-current native target builds and complete admitted runtime fixtures remain
open.

The opt-in tests require independent absolute paths and digests, with no defaults:

- `TestGenuineRetainedLinkSubset` uses H08_RETAINED_LINK_RECEIPT and
  H08_RETAINED_LINK_RECEIPT_SHA256.
- `TestGenuineRetainedFullBody` uses H08_RETAINED_FULL_RECEIPT,
  H08_RETAINED_FULL_RECEIPT_SHA256, H08_RETAINED_FULL_HANDOFF and
  H08_RETAINED_FULL_HANDOFF_SHA256.
- `TestGenuineSelectedObservation` uses H08_OBSERVATION, H08_OBSERVATION_SHA256,
  H08_OBSERVATION_SELECTION and H08_OBSERVATION_SELECTION_SHA256.
- `TestGenuineSelectedObservationRehashedNegative` also requires
  H08_OBSERVATION_NEGATIVE=output, native, or summary. Run each family in its own
  bounded attempt. Rehashed ancestors are adversarial fixtures, never witnesses.

Retain each attempt's exit, clocks, logs and cleanup. A timeout stays failed.
Ordinary `go test ./...`, maintained parity and the final independent review
remain separate integration gates. RuntimeHeld stays true. This metadata proof
does not admit native target execution, GPU graphs, CPU13, all64, padding, fixed
profiles, the full caller, biological results or performance. H08 remains open
and `approvedGPUBundles` remains empty. Dorado, stock and HG005 are unchanged.
