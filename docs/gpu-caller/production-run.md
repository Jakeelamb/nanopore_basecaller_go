# Production GPU run boundary

`caller.RunGPU` owns a bounded one-file POD5 operation, cross-read batching,
decoded assembly, shared postcall, output finalization and resource drain.
`cmd/basecaller --device cuda:0` calls it without an outer deferred writer.
The CPU and replay paths remain explicit existing producers.

The production bundle registry is intentionally empty. No complete ABI5/POD5
installation and independently accepted numerical-program identities have been
qualified together. `--native-bundle DIR` looks for `gpu-bundle.json`; its exact
manifest digest must identify a compiled approved release whose library members,
program identities and admitted profiles are independently pinned. Hashing the
currently loaded program does not authorize it. This is an unavailable product
startup boundary, not a numerical acceptance claim or a fake successful caller.

Behind that gate the concrete fixed path loads the actual Go-owned HAC or SUP
graph and calls Go guides, beam search, posterior scheduling and quality decoding.
Quality owns T sequence/Q storage bytes; only the prefix selected by the sum of
the moves is materialized. There is no stock decoder wrapper or CPU inference/
decoder fallback. Existing postcall mean-Q arithmetic remains shared.

The default simplex source profiles are `[12288,6144]` for both fixed modes and
variable HAC, and `[12288]` for variable SUP. Profile selection uses the literal
strict native comparison after normalization. Lengths 6143/6144/6145 select
6144/12288/12288. Each parent keeps its selected planner, queue and capacity.
`DORADO_EXTRA_CHUNK_SIZES` must be unset for this admitted default policy.

The current numerical adapters represent full-active fixed chunk12288 and
chunk6144, physical N64 HAC or N32 SUP. The [short-profile source unit](source-short-profiles.md)
adds actual graph geometry but no native qualification. Genuine partial lanes,
packed GPU graphs/indexing and the remaining legal beam CRC-collision schedule remain
unfinished. Selecting an unavailable domain fails; the caller never promotes a
short parent, calls a tiny verification graph, pads the parent universe or retries
with a different decoder. Host packing evidence, including the HAC VCS unit,
does not qualify a packed GPU graph.

The synchronous run owns one engine/device and one lazily loaded typed HAC or
SUP model. Each used source profile owns its stream, immutable forward plan and
input buffer. Full and short calls borrow the same persistent cache set; each
forward owns its workspace. SUP keeps the full RoPE owner separate from each
invocation's prefix view, and persistent QKV scale bindings survive profile
cleanup. See [shared-model ownership](shared-model-caches.md).

Stream drain occurs while model/input/transient owners remain live. After all
profiles drain, the run closes the shared model before its device and engine.
A failed drain defers cache teardown until device/engine recovery; failed owners
remain available for retry. Cleanup errors survive every later close and prevent
successful completion. These ownership contracts do not qualify numerical
profile alternation, native allocation counts or device memory use.

`GPULimits` bounds live parent count, retained normalized samples, decoded samples
per input parent and aggregate queued chunks across profiles. A long parent
submits a coordinate cursor in queue-sized windows. It need not fit every chunk
in the queue. Retained state is removed after ordered postcall/output, including
parents whose children are all filtered. The single input callback may also hold
one bounded raw parent while admission drains previous parents.

FASTQ uses existing called-read serialization. BAM consumes postcall children,
including lineage and the original binary32 scaling metadata, before retirement.
Its program identity is `nanopore_basecaller_go`; command line and policy supply
truthful PG/trim metadata. Header protocol/acquisition clocks and experiment-ID
fallback follow pinned `file_info.cpp`. The current BAM header API supports one
file run. Mixed-run BAM fails before output until ordered multi-read-group support
is implemented. GPU `--capture` is rejected rather than silently ignored.

A nil run error requires selected input EOF, every parent processed/serialized,
every kept child emitted once, no failed/abandoned/pending chunks, finalized output
and successful native/input cleanup. Counts returned with an error are prefix
diagnostics; even equal parent counts cannot override a final flush error. Short
writes with nil errors become `io.ErrShortWrite`, including BAM EOF.

Host tests use private input/invocation/output substitutes only for ownership and
failure behavior. They do not qualify numerical work. Source-only preparation
does not establish even those host tests passed. The separate handoff lists the
requested checks and remaining genuine native endpoints. Full acceptance still
requires `go test ./...`, maintained parity and genuine caller/native comparisons.
HG005 remains excluded from selection.
