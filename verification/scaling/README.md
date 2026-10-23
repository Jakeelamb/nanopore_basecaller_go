# Standardised DNA pA reference

The Go implementation owns ADC subtraction, float32 division, IEEE binary16
rounding and the fixed DNA prefix trim. It accepts a precomputed ADC open-pore
adjustment. It does not infer an expected open-pore level from a flowcell.

`oracle.go` checks the entire pinned ScalerNode.cpp SHA256 before extracting
lines 202-203 and 233-234 verbatim. The generated `source-slices.json` contains
the exact text, line ranges and SHA256 of each slice. Changing the upstream
source fails extraction before compilation. The original source checkout is
never modified.

The native reference uses the extracted metadata formulas. Its separate scalar
float32 subtraction and division followed by compiler `_Float16` conversion
model the arithmetic in tensor_utils.cpp. This reference does not execute
ATen/Torch, the SIMD dispatch, GPU inference, quantile scaling, RNA trimming or
end-to-end Dorado basecalling. Its binary output is native little-endian on the
qualified Linux x64 host.

From the repository root:

```sh
go test ./internal/signal/...
DORADO_SOURCE=/home/jake/Projects/nanopore_basecaller/software/dorado-src \
SCALING_ARTIFACT_DIR="$PWD/artifacts/scaling" \
go test ./internal/signal/... -count=1 -v
```

The differential covers all 65,536 signed ADC inputs for six parameter sets,
including positive and negative nonzero adjustments. It compares exact half
bits and exact float32 pA metadata. Prefix padding allows every ADC input to
survive the ten-sample trim. Separate tests exercise lengths on both sides of
the trim boundary, reject invalid parameters and derived overflow, reject a
mutated native source, round-trip every half bit pattern, and check every
adjacent finite positive half midpoint and both neighboring float32 values.
Negative midpoint ties are checked too. NaNs are quieted with sign and available
payload retained. The benchmark retains its complete returned result in a
package variable; no benchmark speed claim has been established.
