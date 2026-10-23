# Chunk computational parity

`internal/chunk` ports the three chunk-coordinate algorithms in untouched Dorado
`dorado/read_pipeline/base/chunk.cpp`, revision
`8b8fc5d36a9c0baab262a743cb175b5878e38ca6`.

Run unit checks and the bounded property seeds with `go test ./internal/chunk/...`.
Native parity skips explicitly unless `DORADO_SOURCE` is set. Run native parity:

```sh
DORADO_SOURCE=/path/to/dorado-src go test -count=1 -v ./internal/chunk/...
```

The native test requires the exact source revision and clean tracked files. It
compiles `oracle.cpp` together with the original stock `chunk.cpp` in a temporary
directory using `g++ -std=c++20 -O2`. No source copies or replacements implement
the oracle algorithms. The test compares exact coordinates and accepted/rejected
status for 4,995 deterministic upstream-style, boundary, invalid, and seeded
random cases. Exception text is not part of the parity endpoint. Inputs at the
Go resource boundary are tested separately and are not fed to the native oracle.

`RepeatPad` copies float16 sample bits without decoding floats. It matches the
repeat-prefix padding behavior of `BasecallerNode::basecall_worker_thread` at
stock `dorado/read_pipeline/nodes/BasecallerNode.cpp:468`. Its bit-pattern unit
checks are Go behavior evidence. They are not a native ATen differential result.
For variable chunks, callers supply a target rounded up to the model's working
unit. This helper does not choose that target or implement model execution.

The Go input boundary rejects coordinates above `MaxCoordinate = 2^53 - 1`,
planner results above `MaxChunks = 1,000,000`, and padding targets above
`MaxPaddedSamples = 2^26`. It rejects empty signals and shrinking padding.
These named limits prevent overflow and unreasonable allocation. They narrow
the unrestricted native integer domain and do not change realistic read sizes.
Variable planning intentionally retains the stock double-precision `ceil`
calculation. The transformed chunk planner retains stock behavior allowing
unaligned overlap values if its other configuration checks pass.

These checks establish planner parity only. They do not establish scaling,
model, CRF decoding, stitching, GPU, end-to-end, or biological parity.

Set `CHUNK_ARTIFACT_DIR` to an absolute ignored output directory to retain the
compiled oracle, `compile.log`, `input.txt`, `native-output.txt`, and
`native-stderr.txt`. Without it these files live in the test temporary directory.
The caller should choose a fresh attempt directory to preserve prior attempts.
