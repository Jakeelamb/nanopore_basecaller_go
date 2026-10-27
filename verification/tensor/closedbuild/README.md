# Closed tensor backend compilation

This controller prepares and compiles the maintained tensor backend recipe.
It has no operation that executes a native output, invokes `ldd`, initializes
a device, or runs a fixture. Compiler work requires a separate exact grant;
runtime remains held.

The maintained `verification/tensor/build` command now prints its recipe with
`-plan-only`. That path returns before creating its output directory, querying
a compiler, or running native code. Its default behavior and flags remain the
same: it still builds and executes the failure probe, builds the backend,
invokes `ldd`, and builds/generates fixtures. Use the closed controller for
separately reviewed compiler stages.

The shared `recipe` package supplies the actual flags and combined driver
argv to the maintained default path. The closed controller splits each
explicitly selected TU into a retained object and link, keeps the arithmetic
flags, changes dependency emission from `-MMD` to `-MD`, replaces library
searches with independently pinned files, and adds a link trace and map.
There is no implicit `all` profile. Available closed profiles are `backend`,
`fixtures`, and `failure-test`. The maintained CUDA source oracle recipe is
unchanged; the closed controller rejects that profile until its extracted
source block has a separate immutable preparation contract.

`-unit controller -host-admitted -out NEW` builds two CGO0 executables:
`closed` and `maintained`. It pins the finite selected Go tool family before
the first query, records actual dependency selection, snapshots every input,
uses a fresh private cache with forced rebuild and PGO disabled, and verifies
the selection afterward. The compiler phase requires the exact sealed
`closed` executable and current source/tool/snapshot pins.

`-unit plan` takes independent path/hash pairs for the maintained recipe,
D01 compiler contract, S00 link contract, retained distribution asset contract,
sealed controller, actual CUDA driver and pthread archive, plus explicit
`-profiles`. It reads
immutable contract data rather than importing mutable S00/D01 implementations.
The stable H07 dependency and link-trace parsers bring their existing HAC,
model and tensor Go dependencies into the declared controller closure.

The plan rederives every selected compiler ELF domain through `nativeproof`,
including GCC LTO and BFD plugins, and pins the loader cache. It authenticates
all selected implicit objects, linker scripts/resources, Koi and direct link
files against the existing closed bindings. The added `-lcuda` is replaced by
the independently pinned actual driver ELF; a stub path is rejected. Selected
include/library-root aliases are bound to canonical paths and checked before
every compiler command. The toolkit header contract is CUDA13.4 (`13040`).
The pinned ONT archive authenticates the selected Torch files; all emitted
Torch header dependencies must match that archive before compilation.

The 23-object prospective runtime inventory closes each `DT_NEEDED` name over
the explicitly selected files, including the actual driver. It is not a
loader search, cache, hwcaps or future lazy-load claim. Actual runtime maps and
the separately reviewed S00 audited-load guard remain required before loading
the new backend. No second H06/backend runtime guard is introduced here.

`-unit compile -plan PLAN -plan-sha SHA -out NEW` returns75 without an explicit
`-compiler-admitted` grant. With a separate grant, it takes the canonical
read-only nonblocking host lock and requires at most two CPUs, 3GiB, zero swap
and64 tasks. Every child has a180-second process-group deadline; the required
outer launch must also bound and clean the whole control group. Selected
compiler/input/root bindings are checked before and after every command.
All `-M` sets are frozen before the first compile, and actual `-MD` sets must
match exactly. Link traces admit only pinned files and retained objects.
The current ABI5 backend output must export the guide program and frame, beam
program and numbers, post numbers, and SUP symbols. Output status remains
`built_outputs_unexecuted`, with `RuntimeHeld=true`. The integrated source
requires a new controller and plan. See
[`integrated-abi5.md`](../../../docs/gpu-caller/integrated-abi5.md).

Initial source/host evidence is retained in
`artifacts/backend-closed-host-20261025-003/`. Six named test roots and vet
pass with CGO0. Its sealed two-binary controller has944 inputs and snapshots;
its backend-only plan has1,999 inputs,22 selected queries,14 compiler ELF
domains containing37 pinned compiler/plugin/cache files, and23 prospective
runtime objects. Actual maintained plan-only execution creates no output.
All four normalized maintained compiler argv profiles match the recipe after
explicit compiler/output normalization and the reviewed D02 `-lcuda` addition.
A self-rehashed `-O1` to `-O2` recipe is rejected before output, and an actual
unadmitted compile request returns75 before reading its deliberately invalid
plan. An include-root retarget with identical header bytes is also rejected
by a host test. Failed host001 vet evidence and superseded host002 remain.

The current source/host handoff does not establish C++ compatibility, new
backend loading, the thirteen CPU contracts, fixture generation with the
genuine HAC weight, CUDA/PTX compilation or execution, or GPU parity. The
recipe pins the genuine `0.conv.weight.tensor` for that later CPU gate.
The closed build emits its own `build.json`; it does not manufacture a
legacy `status=built` manifest. Existing maintained CUDA acceptance expects
the older build/fixture contract, so its eventual consumer needs a reviewed
producer/current-consumer handoff and the shared runtime guard. The historical
f197 backend remains unchanged and lacks the two new mandatory symbols.

The first admitted compiler attempt is retained as failed in
`artifacts/tensor/backend-closed-build-001/`. All25 compiler child commands
exit zero, and its5,076-entry `-M` and `-MD` sets agree. The actual link consumes
`/usr/lib/libpthread.a`, which the reused S00/H06 binding data omitted. The
closed trace guard rejects that omission. Static ELF inspection finds all42
current loader names, including both D02 exports, but this does not qualify
the failed artifact. The failed manifest and before-compile set remain
unchanged; no library has loaded. The sealed failure handoff is
`artifacts/backend-compiler-failed-handoff-20261025-001/handoff.json`.

The prospective repair requires `-pthread-archive` and its independent SHA.
The selected file must be named `libpthread.a`, contain exactly the eight
bytes `!<arch>` followed by a newline, and match SHA256
`f0a17a43c74d2fe5474fa2fd29c8f14799e777d7d75a2cc4d11c20a6e7b161c5`.
It enters the plan before any native query. A selected
`-print-file-name=libpthread.a` query must return the same canonical pinned
file. Its path binding and bytes are checked at each execution boundary.
Missing declaration, changed/self-rehashed archive bytes, and a different
selected path with identical bytes are rejected. This is a new prospective
contract; it does not rewrite the old receipt or change `-pthread`.

The repaired source/host handoff uses
`artifacts/backend-closed-host-20261025-005/`. Seven test roots and vet pass.
The fresh sealed controller has945 inputs and snapshots; the new backend
plan has2,002 inputs and23 selected queries, including the future pthread
selection check. An actual sealed-controller request omitting only that pin
is rejected before output. A Go-only regression reads the unchanged failed
receipt and trace: the old omission still fails, and the same trace matches
only a prospective set with the archive explicitly added. It does not promote
the old artifact. All2,002 current plan inputs were independently rehashed.
No native retry or actual pthread print-file query ran in this repair phase.
