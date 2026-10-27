# H08 fixed source-default geometry

`FixedHACGPUProfile()` remains N64, chunk192, T32 for small verification.
`SourceDefaultHACGPUProfile()` explicitly selects N64, chunk12288, T2048.
`PlanHACGPU` accepts only these two I8 profiles. This is a source and host
planning change. Native execution at the larger geometry remains unqualified.

The pinned Dorado source is `8b8fc5d36a9c0baab262a743cb175b5878e38ca6`.
`CRFModel.cpp:87-145` reserves the graph separately before actual execution.
`ConvStack.cpp:191-211` propagates allocated padding during reservation.
The HAC v6.0.0 model config sets `basecaller.chunksize = 12288`.
For that profile, conv1 reserves 12306 rows. Conv2 reservation therefore uses
T2051 and 2056 parent rows. Execution crops conv1 to 12288 rows before conv2,
so conv2 and all seven recurrent layers use T2048 in a `[2053,64,1024]` I8 parent.
The factorised output returns contiguous half `[64,2048,1024]` scores.

The resulting checked reservation is 437174272 bytes. Actual recurrent storage
occupies 134545408 bytes at offset302628864. Actual half projection scratch
occupies 301989888 bytes, comprising 33554432 down bytes and 268435456 score
bytes. Reservation still uses T2051 for its larger projection scratch.
All runtime input, crop, parent, padding, observation, and result dimensions
come from the admitted profile or its checked convolution and output plans.
C1024 and K128 remain model constants with unchanged archive identities.

Both profiles reject one-slot production forwarding with
`ErrHACGPUInactiveSlotsUnqualified`. `ProbeInactiveSlots` retains the same
production graph for qualification and cannot promote the production policy.
Variable chunk sizing is excluded. It needs distinct encodings, row offsets,
lookup tables, and native kernels.

Host acceptance uses `CGO_ENABLED=0` with the exact selector
`^Test(HACGPU|HACFused|HACOutput)` in `./internal/model`, followed by package vet.
The new tests check both profiles, padded reservation versus cropped use,
view extents and lifetimes, source-default recurrence host calls, input profile
identity, inactive-slot quarantine, and negative or overflowing requests.
These checks do not allocate GPU memory or run native arithmetic. They establish
no native parity, device memory qualification, end-to-end caller, biological,
or speed result. Maintained native parity and larger-geometry native oracle
comparisons remain separate coordinator-owned acceptance gates.

The isolated acceptance receipt is
`artifacts/hac-full-fixed-host-20261025-001/receipt.json`, SHA256
`69a7471b2ac785fb9d76bbf2266a3a969e725176e6ee70abeb486dbb0f316b9f`.
All 23 required test roots and 67 named tests including subtests passed with
zero skips. Package vet passed. The receipt retains 1606 source and tool pins
with immutable copies and zero drift. The readonly nonblocking host lock was
released at `2026-10-25T14:28:06.257131Z`; `release.json` confirms cgroup removal.
Effective caps were 2 CPUs, 3 GiB memory, no swap, and 64 tasks, with a
180-second runtime bound. The test process used private Go cache, temporary,
and GOPATH directories. No CGO1, genuine backend, or GPU action ran.
