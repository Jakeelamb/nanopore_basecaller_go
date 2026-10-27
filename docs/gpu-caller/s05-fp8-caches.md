# S05 FP8 feed-forward cache candidate

Status: bounded unit native cache parity passed independent review. This does
not close S05's GPU operations or qualify SUP inference.

`HalfToSUPFP8` ports the E4M3FN conversion in the pinned Torch header
`torch/headeronly/util/Float8_e4m3fn.h`, SHA256
`64dcf781f48a354a30ca44379e09aecdc2f2224f407d8d69a3c4b47080745da2`.
It consumes the already-cast half bits, preserving signed zero and the source's
signed NaN/overflow encoding. The host exhaustive check compares all 65,536
half patterns with an independent nearest-even representable-value definition.
That mathematical comparison is not a native CUDA conversion observation.

`PackSUPFeedForwardFP8` follows pinned `TxModules.cpp:488-506` at C512/E4096.
It interleaves the two fc1 halves in 16-row groups, applies the source's explicit
16-lane shuffle, converts to FP8, then tiles. Fc2 converts and tiles separately.
The immutable cache snapshots have shapes `[256,32,16,16]` and `[32,128,16,16]`.
Inputs are source half parameters before remove_bits. Source remove_bits is a
no-op for these FP8 tensors; it still applies to the separate half caches.

The admitted packer domain is finite half input producing finite E4M3FN output.
Other shapes, nonfinite parameters and conversion overflow are rejected. The
scalar helper retains all raw encodings for differential work. Caller dispatch
must still use the observed E4M3 capability and default koi_use_f8=true; this
host utility grants no device or dispatch qualification.

`artifacts/s05-fp8-host-20261025-001/` retains three passing host roots:
exhaustive conversion, all-byte cache comparison against independently composed
source tensor views, and domain/shape rejection. Source arrays and immutable
snapshot ownership are checked.

The CGO0 consumer in `verification/supgpu/s05` reads the original 36 model
parameter archives using the authoritative SUP manifest. It requires their exact
hashes and known single-parameter, little-endian FloatStorage archive profile;
it never executes pickle. It casts those raw float32 parameters to half with Go,
then runs the production FP8 packer. It uses no native intermediate as input.

`artifacts/tensor/s05-native-cache-comparison-001/receipt.json`, SHA256
`208d106463688988fccd85016108f8b7c54d4d556d2cf8fad16ef55f545494ad`,
records exact equality for all 72 FC1/FC2 buffers across 18 layers before and
after source weight rounding: 113,246,208 compared bytes. Changed Go bytes and
an equal-size wrong shape are rejected. This qualifies the composed cache
pipeline on these model parameters, not every scalar conversion or a separate
intermediate half-cast equality. Twelve host tests, vet and a retained CGO0
consumer build passed with source/tool snapshots in
`artifacts/s05-cache-consumer-host-20261025-003/`.

Independent review is retained in
`artifacts/parity-resume-review-20261024/s05-fp8-native-cache-review.md`, SHA256
`9013baa65a87c0271de1715bb4af4edd51ca122da83e9e2d6857f4309fcc16e2`.
It directly compares all 36 distinct Go/native payload pairs. Executable003 owns
the numerical evidence. Review found an external capture-pin filename gap;
fixed executable004 requires the canonical imported `capture.json` identity.
Its 13 host tests and vet pass. The numerical cache computation was not repeated.

The oracle is S00 capture002, whose three source purposes have identical final
scores. Its documented per-role watchdog gap remains a limit; no source model
was rerun for this comparison. Cache upload, the remaining cache families,
primitive operations and full graph acceptance remain open.
