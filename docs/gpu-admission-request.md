# Native GPU admission request (P01 profile capture)

Status: **request only**. This document is not admission, not a release, and not
evidence. An idle GPU or a free host lock is not admission. No native compiler,
lock operation or kernel was run for this request, and no phase below has been
released. One read-only identity query is disclosed at the end.

## What is requested

One coordinator release, scoped to the first native phase:

- **Phase:** `P01` native runtime profile capture for the `source-cuda13` family.
- **Device:** `GPU-807d7ae7-6a99-c795-12c2-23fc801d38e7` (RTX 5070 Laptop).
- **Does:** compile the single standalone probe
  `verification/gpuprofile/native/probe.cpp` against the CUDA 13.4 toolkit and
  the pinned Koi CUDA13 library, run its device/status query, acquire the
  existing nonblocking host lock, and record compiler/probe argv, environment,
  clocks, exits, stdout, stderr and the resulting `profile.json`.
- **Does not:** run model inference, basecall a read, write a BAM, measure
  accuracy or performance, or qualify HAC/SUP arithmetic. Those need their own
  releases (`H*`, `S*`, `D*`, `I07`).

## Exact command

Run from the checkout that holds the accepted ABI4 artifacts
(`artifacts/tensor/...`, ignored by Git). Do not substitute paths.

```sh
go run ./verification/gpuprofile/cmd \
  -plan provenance/gpu-target.json \
  -model dna_r10.4.1_e8.2_400bps_hac@v6.0.0 \
  -config-sha256 e550ccbedb68ed74d6ee88e2f510adf33c52a1c868bd5e3a2bfbb7dfe6154c3f \
  -batch 64 \
  -output bam \
  -kit SQK-LSK114 \
  -runtime-admitted \
  -basic artifacts/tensor/replay-abi4-cuda-001/manifest.json \
  -build artifacts/tensor/attempt-abi4-cuda-004/manifest.json \
  -koi-root /home/jake/Projects/nanopore_basecaller/software/source-build-qualification-20261023/dependencies/libkoi-0.7.5-Linux-x86_64-cuda-13.0/libkoi-0.7.5-Linux-x86_64-cuda-13.0 \
  -cuda-root /opt/cuda \
  -model-root /home/jake/Projects/nanopore_basecaller/models \
  -out artifacts/gpu-plan/gpu-profile-admission-NEW
```

`-out` must not exist. `-runtime-admitted` asserts a specific coordinator
release; it is not a permission obtained by waiting and it is not implied by
this document.

## Pins the command checks

| Input | SHA256 / value |
| --- | --- |
| Dorado source | `8b8fc5d36a9c0baab262a743cb175b5878e38ca6` |
| `provenance/gpu-target.json` | `59944d3d17456a82f12302d509a05ad5e18f4af061fe3de6169ad0864fced582` |
| accepted runtime manifest (basic) | `15ca5c4a0cda12af18a430cf7763efa2bf78590636aa501892b9106d53c9beae` |
| accepted build manifest | `f41faaed7ba863f3d16c5a5f08cb373bf588794ec73fe25d1620359863b9d0dd` |
| ABI4 backend | `d70b612b442e8449b2357d12284705bd82b30cdfa7316358a29ad1c800c9e3d7` |
| Koi CUDA13 header | `8b4115c2e92ebf1acf1db2aa76bdc9b3a22768cb595b92363f88e44ff92d3b88` |
| Koi CUDA13 library | `17cf3083c5db538f1e275b0a41ac19560b5986c143634e8a25540b9735f5805f` |
| HAC v6.0.0 config | `e550ccbedb68ed74d6ee88e2f510adf33c52a1c868bd5e3a2bfbb7dfe6154c3f` |
| CUDA toolkit | `/opt/cuda`, release 13.4 |

The target family is `source-cuda13`. Official CUDA12 is a separate family and
is not requested here.

## Host control already run

Source preparation without admission succeeds and writes
`source-prepared-runtime-unobserved`, with no compiler, lock or device access:

```sh
go run ./verification/gpuprofile/cmd \
  -plan provenance/gpu-target.json \
  -model dna_r10.4.1_e8.2_400bps_hac@v6.0.0 \
  -config-sha256 e550ccbedb68ed74d6ee88e2f510adf33c52a1c868bd5e3a2bfbb7dfe6154c3f \
  -batch 64 -output bam -out artifacts/gpu-plan/gpu-profile-prep-NEW
```

I ran this to a temporary directory and it wrote
`"status": "source-prepared-runtime-unobserved"`. The negative control is `TestCaptureRejectsAdmissionAndQualificationBeforeLock`:
capture must reject a missing admission or missing qualification **before**
acquiring the lock or starting a child.

## Evidence a release would produce

- `profile.json` with the frozen identity, device properties, numeric
  capabilities, Koi statuses, dispatch decisions and the compile/probe receipts.
- Raw compiler and probe logs, argv, environment and clocks under the attempt
  directory.
- A receipt hash that later phases (`H*`, `S*`, `D*`) can pin.

## What still needs separate authorization after this phase

- `P02`/`P03`/`P04` ABI, workspace and CUDA12-family work.
- `H01`–`H09`, `S00`–`S14`, `D01`–`D04`: model inference, stage captures and
  decoder decisions.
- `I07` caller integration and `A01`–`A04` conformance, accuracy and timing.

## Disclosure

While assembling this request I ran one read-only `nvidia-smi` identity query to
name the device. It executed no kernel, acquired no lock and is not evidence for
any card. It is disclosed here because the P01 checkpoint states that no GPU
query was run; that statement was true of the checkpoint, not of this request.
