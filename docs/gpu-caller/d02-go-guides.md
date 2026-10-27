# D02 Go guide scan candidate

Status: source and host candidate in `feature/d02-decoder-go-20261025`.
No genuine C++ backend compilation, PTX compilation, device execution or numerical
comparison has run. This does not close D02 or any caller acceptance gate.

Go owns the fixed-batch reverse-time loop and zero terminal row.
`internal/gpudecode.BackGuides` borrows half scores `[N,T,4*S]` and returns an
independently owned float32 `[T+1,N,S]` tensor. The source oracle stores guides as
`[N,T+1,S]`; a differential consumer must transpose indices before comparing.
The admitted geometry is checked HAC tiles (N multiple of64, S256) or SUP tiles
(N multiple of32, S1024), with bounded extents. Packed and variable chunks remain
unsupported by this candidate.

The native boundary is one `DecoderGuideFrame` operation, with typed half score
rows and float next/output rows. Scores may have a batch stride; next and output
rows are contiguous. Go selects every timestep and every row view. The native
operation neither loops over timesteps nor calls `host_back_guide_step`.

The arithmetic is transcribed from selected `back_guide_step` PTX in pinned
`decoder_lib.cu.o` SHA
`930b35afa9e352b412b58e07a43d693630280ed9f422dc324106b2b8e700758f`,
SM120 PTX SHA
`7e46f59645025bca2cf7a519c9f0334b5435942ca9006d0dec5faa6d3e63b01a`.
It keeps half conversion, symmetric clamp (zero selects FLT_MAX), FTZ adds,
ordered five-term maximum and sum, approximate EX2/LG2 and the final FTZ FMA.
Only address selection and the surrounding time loop were replaced. These are
source claims requiring independent review and differential proof; the inspected
SM120 image is not yet established as the runtime-selected Koi image.

`DecoderGuideProgram` explicitly initializes the per-device module and returns
its actual linked cubin bytes. The driver links embedded PTX targeting SM120.
The caller must authenticate and retain those bytes before using BackGuides.
Initialization does not run a decoder timestep. Driver/compiler/runtime
identities and library admission remain required; source bytes alone do not
qualify runtime JIT output.

The device owns the module, function and linked bytes. Failed initialization
retains module/linker resources until device close. Close drains all streams
before unloading the module and destroying any retained linker. Failed release
keeps the owner for retry. Starting device close disables further work even
after a failed release; retry device close to finish cleanup. Frame execution uses existing CUDAWork tensor
retention and drain handling. The ABI and Go wrapper reject wrong shape, dtype,
tags, scale bindings, owner, bounds, overlapping byte intervals and invalid
scalar options before launch.

Host attempt001 passes Go scheduling/layout tests. Attempt002 failed while
compiling a fake C fixture because its local enum was missing; the failure is
preserved. Attempt003 passes after that fixture correction. Attempt004 adds a
real loader/cgo marshalling check for cubin bytes, scalar bits, operand roles,
closed handles and writable overlap. Fake results are not GPU arithmetic.

Remaining work includes independent exact-delta review, a closed native build,
prospectively admitted runtime/JIT execution, D01 guide comparisons, effective
reverse-index/blank/clamp/stride/chunk/stale-row controls, SUP qualification,
packed scheduling, posterior scans and caller integration. Model or decoder
source inference must not be repeated merely to repair consumer evidence.

The producer-free guide comparator converts TNS to the fixed D01 NTS layout
using byte copies. Its host test covers every chunk/frame/state, changed first
and last guide bytes, contract/extent rejection, and the prospective exclusion
of separate uninitialized beam history. It has not consumed a genuine D01 GPU
capture yet.
