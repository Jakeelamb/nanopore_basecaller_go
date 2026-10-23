# Native backend bringup

The Go computations do not yet form a complete basecaller. The selected target
gives Go ownership of the model graph, weight interpretation, decoder decisions,
signal preparation, chunk membership, stitching and scheduling. A small C ABI
exposes pinned tensor allocation and native compute kernels. A whole-model
forward or decode bridge can support intermediate diagnostics, but does not
complete that target. Invoking the stock CLI remains an oracle operation.

## Current evidence

Observed on 2026-10-23 UTC, from Dorado source
`8b8fc5d36a9c0baab262a743cb175b5878e38ca6`, committed 2026-08-26.
`cmake/Torch.cmake`, `cmake/Koi.cmake` and `cmake/Pod5.cmake` specify public,
checksummed development distributions. Their exact source hashes are checked
by `verification/backend/preflight.go`. This evidence is local source and HTTP
metadata, not a completed native backend build.

| Stock CUDA 12.8 package | Archive bytes | Source-declared SHA256 |
| --- | ---: | --- |
| `torch-2.9.0.0-ont-CUDA-12.8-linux-x64-cxx11-abi.zip` | 2,016,718,479 | `ed09249f48bc3e20130307e86a729e8bc8a97aed794c149b60efcae172654a9d` |
| `libkoi-0.7.5-Linux-x86_64-cuda-12.8.tar.gz` | 69,125,663 | `bfb3bea6f1bb7f7e6e602d4899db14f422a46519ce69e2d1b645050eba307aad` |
| `lib_pod5-0.3.39-linux-x64.tar.gz` | 44,034,293 | `7fcf1d9f1da6f1bd13167782e900462fbadb0db57c666911bdba9bbdfd4e7e06` |

Torch and Koi URLs start with
`https://cdn.oxfordnanoportal.com/software/analysis/`, followed by their archive
names. POD5 is hosted at
`https://github.com/nanoporetech/pod5-file-format/releases/download/0.3.39/`,
followed by its archive name. All three returned final HTTP 200 to `curl -I -L`.
A Torch one-MiB range request returned HTTP 206 and exactly 1,048,576 bytes.
HEAD sizes do not replace payload checksums.

Koi and POD5 were downloaded, fully checksum-verified and extracted under
`external/backend/`. Their successful extraction receipts pin both archive and
extracted tree. Koi includes `include/koi.h` and `lib/libkoi.a`. POD5 includes
`include/pod5_format/c_api.h` and the four static libraries required by upstream
POD5 CMake. Expanded archive sizes are retained in the receipts. No full Torch
payload was acquired. Its partial download was preserved after slow full-file
and range attempts. Resume is supported; availability is established, while
completion time remains an external constraint.

The stock executable contains Koi symbols in its normal ELF symbol table.
`readelf --dyn-syms` exposes no matching `koi_` entries on this artifact. Treat
these compiled-in kernels as unavailable to `dlsym` through the executable.
Use the downloaded static Koi archive for a separately linked bridge.

## Toolkit selection and scope

The host has `/opt/cuda/bin/nvcc` 13.4.92, CUDA runtime headers and libcudart.
Upstream full CMake would select CUDA 13.0 Torch and Koi on this host.
That differs from the stock runtime family. The maintained tool therefore
**defaults explicitly to CUDA 12.8**. It never substitutes the host-selected
13.0 family during acquisition.

The optional 13.0 packages also returned HTTP 200. Torch is 1,281,640,079 bytes
with SHA256 `8680441dbe3b0990fa7a99a70319affe25bc51f0cd6e105c3e07d522d7355929`.
Koi is 58,019,481 bytes with SHA256
`0600a138761b17467bd3a1f8933c6a6a02e17415a9da377f22bc14357c3101ca`.
They are a separate future configuration, not stock parity evidence.

A C++-only bridge can link precompiled Torch/Koi objects without compiling new
CUDA kernels. It may not require nvcc. Consequently the observed 13.4 compiler
is reported as present but unqualified for stock-family compilation, rather
than an absolute missing prerequisite. Header compatibility, linking, ABI,
runtime library resolution and a real native operation still need proof.
Do not mix a 13.0 development package with stock 12.8 runtime libraries.

Koi's source-build override references a vendor-internal repository. The public
binary development distribution avoids needing that source repository for
initial bringup. Distribution access is not a license determination. Preserve
upstream licensing and distribution notices when producing a native backend.

## Maintained commands

```sh
go test ./verification/backend/...
go run ./verification/backend/cmd -cuda-family 12.8

go run ./verification/backend/cmd -acquire -cuda-family 12.8 -packages pod5,koi

go run ./verification/backend/cmd -acquire -cuda-family 12.8 -packages torch
```

All acquisitions use the ignored `external/backend` root by default. The tool
checks a 30 GiB free-space reserve before downloading, plans at most 12 GiB of
expanded content, verifies full archive sizes and SHA256 before extraction,
inspects expanded sizes and unsafe member paths, and checks the reserve again
before and after extraction. Each curl attempt has a 180-second limit. A failed
download retains its `.partial` and command log for a later resumed attempt.
Verified extracted trees are checked against extraction receipts before reuse.
Failed extraction stages are preserved with unique paths.
The authoritative extraction receipt is written inside the staged package
before the directory is atomically renamed into place. The sibling JSON is a
convenience copy. An interruption before that copy is written does not prevent
reuse. Existing packages produced by the earlier tool retain support for their
verified sibling receipts. A published tree with neither receipt is rejected.

Preflight returns machine-readable JSON and exits nonzero for missing or invalid
required files. It verifies the pinned CMake files and downloaded archives.
For manually supplied extracted files it records observed hashes and labels
presence as unqualified. Even a complete dependency inventory leaves
`BackendQualified` false. A backend build and native computation are separate
acceptance steps.

Local receipts are under `artifacts/backend/`. The initial preflight rejected
missing development assets. The successful smaller-package acquisition is
`acquire-verified-retry.stdout`, with exit zero. An earlier retry encountered a
retained extraction directory; the tool now uses unique stages, and that failed
receipt remains preserved. The after-preflight still exits one because Torch's
archive, two headers, CMake package and library are missing. It does not claim
backend readiness.

## Smallest next acceptance

Complete the stock-family Torch archive and verify its declared hash. Build a
C++ shared library with opaque handles, explicit dtype/layout/shape and bounded
ownership/error rules. First prove allocation, a float32 arithmetic operation
and deterministic copied output through Go. Add rejection cases for wrong dtype,
shape and stale handles. Pin compiler flags, every linked library and runtime
resolution. Then load one pinned stock model and capture per-layer or per-chunk
intermediates on HG002 development data. Require matched native outputs before
expanding coverage. GPU execution, decoding, end-to-end biological accuracy and
performance remain unqualified by this dependency work.
