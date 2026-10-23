# Float CPU CRF decoder

`internal/decode.Decode` computes beam search, CRC32C sequence merging,
backtrace, moves, sequence, and quality aggregation in Go. Five scalar float
operations use native libm through cgo. Its reference is Dorado commit
`8b8fc5d36a9c0baab262a743cb175b5878e38ca6`,
[`beam_search.cpp`, lines 18-520](https://github.com/nanoporetech/dorado/blob/8b8fc5d36a9c0baab262a743cb175b5878e38ca6/dorado/basecall/decode/beam_search.cpp#L18-L520).
This unit accepts existing score, backward-guide, and posterior tensors.
Tensor generation, model inference, and CUDA decoding remain separate work.

## Input and result

```go
input := decode.Input{
	Blocks: t, States: s,
	Scores: scores, BackGuides: guides, Posts: posts,
}
result, err := decode.Decode(input, decode.Options{
	BeamWidth: 32, BeamCut: 100, BlankScore: 2,
	QShift: 0, QScale: 1,
})
```

All tensors use contiguous row-major `float32` storage. `Scores` has shape
`[T,4*S]`. `BackGuides` and `Posts` each have shape `[T+1,S]`.
The initial guide row selects seed states. Row `block+1` guides each extension
and supplies posteriors for that block. Score and posterior scaling are both
one, matching Dorado's float dispatch. No tensor stride or quantized path is
exposed.

`Result.Sequence` and `Result.Qstring` contain one byte per emitted base.
`Result.Moves` contains one binary move per input block, with the first move
forced to one. `States` contains the emitted base code for each block,
where A, C, G, and T are 0, 1, 2, and 3. The native routine also replaces its
backtraced k-mer states with those base codes during quality generation.
`FinalScore` is the selected path score after removing the final guide.
`QualData` holds the four per-base probabilities per block before accumulation
into emitted bases. Input slices remain unchanged. Results own their storage.

The supported external domain has these limits.

- `T` is positive. `S` is a DNA k-mer state count, one of 4, 16, 64, 256,
  1024, 4096, or 16384.
- Beam width is 1 through 256. Beam cut is zero or at least one.
  Zero disables the initial logarithmic cut but retains beam-width pruning.
- Tensors and options contain finite values. Posterior entries are in `[0,1]`.
  Posterior row sums are unrestricted, including zero. The algorithm clamps
  the sum of relevant posterior states before its quality transform.
- Quality scale is positive. Native zero scale can compute infinity times zero,
  followed by an undefined conversion of NaN into a quality byte.
- Each input tensor and the persistent beam has at most `2^27` elements.
  Scores, guides, and blank score have absolute value at most
  `MaxFloat32 / (16*(T+1))` to prevent accumulation overflow.

These checks reject malformed or unsafe inputs before the search allocates its
workspace. No native-equivalence claim applies to rejected inputs. The original
routine does not safely define every malformed case, including zero blocks,
zero beam width, or a positive beam cut below one.

## Preserved search behavior

The seed threshold is the beam-width-th largest initial guide. Seeds enter in
ascending state order, including the first qualifying states when guides tie.
Go sorts a copied guide row to obtain the same threshold as native
`std::nth_element`. Sorting the beam itself would change behavior.

Each block appends all four step candidates per previous beam element, followed
by the stay candidates. The sequence hash uses the reversed CRC32C polynomial
`0x82f63b78` with seed `0x12345678`, first consuming 32 state bits and then two
bits per emitted base. A 4096-entry presence filter avoids unnecessary merge
comparisons. Matching step and stay hashes merge with native log-sum-exp and
the cutoff of 17. A score tie retains the step. Hash equality remains the
native merge criterion, including its possibility of hash collisions.

Pruning preserves candidate order. The cutoff bisection uses at most ten
guesses and targets 80 through 100 percent of maximum width. Its fallback uses
the high score and takes the first candidates when maximum scores tie.
The final block moves the first highest-scoring element to index zero.
Backtrace follows retained predecessor indices and suppresses the seed event.

Quality generation sums unique left- and right-shifted k-mer posteriors,
clamps to `[0,1]`, and applies the native power of `0.4f`. Stay events
accumulate into the preceding emitted base. Phred conversion, scale, shift,
clamping to `[1,50]`, and truncation of `33.5f + qscore` follow the source.
Arithmetic retains `float32` intermediates. Go's transcendental functions
compute in `float64`, so narrowing those results does not reproduce native
float libm. `internal/decode/math_native.go` instead calls `logf`, `log1pf`,
`expf`, `powf`, and `log10f` through five scalar C wrappers. Go owns all beam
decisions, loops, tensor access, and quality accumulation. C receives scalar
float arguments and retains no Go pointers.

`CGO_ENABLED=1`, a C compiler, and native libm are required. A build with cgo
disabled exposes the same API, and `Decode` returns `ErrCGORequired`.
There is no alternative numerical implementation. The no-cgo test checks that
failure explicitly. Native parity requires the qualified compiler and libm
identity. Results from one host do not prove identity across other math
libraries, architectures, or compiler modes.

## Extracted native oracle

`verification/decode` checks the complete source SHA256 before using its bytes.
The pinned hash is
`80868413877e26a264924263aa3def5ea4fa68877e72aadbf8d5420dad1af07e`.
The authoritative constant is `decodeoracle.SourceSHA256`.
The build extracts lines 18 through 521 verbatim into
`beam_search.slice.cpp`. That contiguous slice includes all helper and
`beam_search` statements. The driver supplies standard headers and closes the
namespace opened in the source. It directly instantiates
`beam_search<float,float>` and calls the original `generate_sequence` helper.
The extracted code needs no substitute tensor or SIMD functions.

The driver uses binary32 bit patterns encoded as unsigned decimal integers.
Fixture transfer therefore preserves every input bit. The native compiler uses
`-std=c++20 -O2 -ffp-contract=off -DENABLE_NEON_IMPL=0`. This selects the source's
scalar cutoff counter and prevents fused multiply-add changes. The receipt
records compiler identity, arguments, source hash, extracted-slice hash, driver
hash, timestamps, and exit status.

With cgo enabled, the native tests have two entry points, `TestPinnedNativeDecode` and
`TestNativeSourceMutationRejected`. Without `DORADO_SOURCE`, both explicitly
skip. The commands are:

```sh
CGO_ENABLED=1 go test ./internal/decode
CGO_ENABLED=1 DORADO_SOURCE=/path/to/pinned/dorado-src \
  DECODE_ARTIFACT_DIR=/absolute/path/to/unique-attempt \
  go test -count=1 -v ./internal/decode
CGO_ENABLED=0 go test -count=1 -v ./internal/decode
```

`DECODE_ARTIFACT_DIR` is the attempt directory itself. When unset, the test uses
a temporary directory. Each attempt contains the source slice, copied driver,
compiled executable, compiler receipt and log, fixture input, Go and native
outputs, case descriptions, native stderr, and run receipt. Use a fresh directory
for each attempt because filenames within one directory are fixed. Failed native
runs preserve their output and exit receipt before the test reports failure.

## Recorded qualification

The local 2026-10-22 run passed the commands above. The native run used
`/home/jake/Projects/nanopore_basecaller/software/dorado-src` and retained
`artifacts/decode/attempt-20261022-quality-exact`.
It compared 1465 deterministic fixtures. The set includes all-score ties,
CRC merging, stay and step preferences, beam widths 1 through 256, initial-guide
ties, beam cuts 0 and 1, sparse and zero posterior rows, normalized random
posterior rows, quantized ties, and the 16384-state limit. It also includes the
independent quality-byte counterexamples and neighboring quality shifts within
four binary32 ULPs. Another 1296 fixtures sweep quality-byte thresholds at
Q1.5, Q10.5, Q25.5, and Q49.5, using nine posterior masses and scales
1, 1.02, 1.05, and 10.

Sequence, quality bytes, moves, and emitted states matched exactly for every
fixture. Final scores and every `QualData` float matched bit for bit.
All native comparisons require exact equality. No floating tolerance applies.

The earlier 151-case attempts in `attempt-20261022-01` and
`attempt-20261022-02` are superseded. They accepted six intermediate float
differences and missed a final quality-byte error. Independent review changed
fixture 130 to `QScale=1` and `QShift=0.7775435`, producing Go quality byte 34
where native returned 35 at emitted base 8. The retained red regression is
`attempt-20261022-quality-red/regression-test.log`.

The first differing transform was `powf` in block 9. The posterior mass had
bits `0x3c16a2ee`, with exponent bits `0x3ecccccd` for `0.4f`.
The old Go float64 power narrowed to `0x3e1cede5`, while native `powf`
returned `0x3e1cede6`. That one-ULP difference crossed the final byte threshold.
`TestQualityMidpointRegression` fails before the libm fix and passes after it,
including an amplified scale-10 counterexample. The old attempts and independent
counterexample payloads remain retained. They are failed qualification evidence.

Unit checks cover binary moves, output lengths, printable quality bounds,
input immutability, deterministic replay, CRC bit order, and malformed input
rejection. The native source-mutation control rejects changed bytes before
compilation. Fixtures are synthetic and contain no HG005 data.

This evidence qualifies the tested float CPU algorithm and extracted native
oracle. It does not establish every possible floating-point tie, full Torch
decoder parity, half or integer score paths, native guide or posterior tensor
generation, CUDA behavior, biological accuracy, performance, or a complete
basecaller. The original stock source remains unchanged.
