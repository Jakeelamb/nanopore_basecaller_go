# Native file evidence mechanisms

`InspectELF(path) (ELFMetadata, error)` reads ELF metadata and returns its
canonical regular-file SHA256 pin. `CloseELF(seeds []Pin, searchDirs []string)
(ELFClosure, error)` verifies seed hashes before parsing and returns sorted
`Nodes`, `Inputs`, and dependency `Bindings`.

The closure includes the interpreter and recursive `DT_NEEDED` dependencies.
It expands only `$ORIGIN` and `${ORIGIN}`. RUNPATH is direct; RPATH propagates to
descendants. Explicit and embedded directories form conservative candidates:
multiple distinct files reject, even when their bytes match. Canonical symlink
aliases deduplicate. Missing paths, relative search paths, unknown tokens,
dependency cycles, inconsistent name/SONAME bindings, incompatible dependency
ABIs, and incomplete or unsupported dynamic metadata reject.

Dynamic strings come from the runtime `DT_STRTAB`/`DT_STRSZ` range, uniquely
mapped through file-backed `PT_LOAD` bytes. The linked string-table section must
match that address, offset, and size. The dynamic stream must terminate with
`DT_NULL`; subsequent entries must be zero padding. Missing or duplicate table
tags, conflicting sections, unmapped or overflowing extents, and invalid string
references reject. Sectionless dynamic inputs are unsupported.
For both metadata ranges, every other `PT_LOAD` memory extent must be disjoint;
even a one-byte partial overlap or a competing BSS range rejects. Empty and
exactly adjacent extents are valid nonoverlaps.

One closure describes one selected loader domain. Independent compiler tools or
plugin contexts can have separate closures whose pins the caller unions; this
does not weaken name or SONAME ambiguity checks within each closure.

This is prospective metadata inspection. It executes no tool, compiler, target,
loader, or device API and does not emulate loader-cache, environment, hwcaps, or
namespace precedence. Callers explicitly select tools, compiler plugins,
dlopen families, loader-cache evidence, and runtime mapping policy. Those
selections remain required alongside the ELF closure.

`sha256.inc` is a byte-for-byte copy of the existing decoder oracle's standalone
SHA256 implementation. It is source reuse, not a new cryptographic implementation.

Host checks: `CGO_ENABLED=0 go test -count=1 ./verification/nativeproof` and
`CGO_ENABLED=0 go vet ./verification/nativeproof`. Tests generate metadata-only
ELF fixtures in temporary directories; they execute no fixture or C compiler.
