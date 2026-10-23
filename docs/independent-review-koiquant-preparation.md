# SUP int8 oracle preparation review

An independent native Sol-family verifier passed the source-only preparation
unit. This qualifies source extraction, saved-input reuse, fixture preparation,
and validation. Native compilation, arithmetic, GPU execution, and the int8
cache implementation remain unfinished.

The eight reviewed files are pinned in
[`koiquant-preparation-reviewed-source.sha256`](../provenance/koiquant-preparation-reviewed-source.sha256).
The full verdict and raw receipts are retained locally under
`artifacts/koiquant/independent-source-review/`. The worker handoff is
`artifacts/koiquant/attempt-handoff-001/report.md`, SHA256
`6c6c8a98dcbf5422a95a3a2c945e6c353f53b2e56410094ba221f03ac4214dd2`.

The reviewer independently ran these commands against copied, pinned sources.

```sh
CGO_ENABLED=0 go test -json -count=1 ./verification/koiquant/...
CGO_ENABLED=0 go build -o /tmp/nanopore-go-koiquant-source-review/koiquant ./verification/koiquant/cmd
```

Eleven named tests passed without skips. The command package has no tests.
The reviewer checked both retained CPU and CUDA preparation manifests and
rejected stale source, missing pins, altered extraction, missing stages and
fixtures, invalid payloads, and false native claims. A process trace of a check
showed no child executable launch. The coordinator also ran
`go vet ./verification/koiquant/...`, exit 0, with raw output and clocks retained
in `artifacts/koiquant/root-vet-001/`.

The CPU and CUDA labels describe planned execution. They do not identify an
executed runtime. Storage contracts and matching hashes do not prove arithmetic.
The existing genuine capture has absolute source-root bindings, so independent
fresh preparation under `/tmp` was rejected. Existing preparations were checked
in place. No native producer, GPU query, or sequencing computation was repeated.

See [the preparation contract](koi-qkv-int8.md) for commands and pending gates.
