# Caller and R1 source checkpoint

This branch packages the accepted Go caller integration and its integrated model,
decoder, tensor ABI5, and verification dependencies. The caller owns model caches,
invocations, output, and cleanup. Native operations remain behind the typed tensor
boundary documented in [the port plan](port-plan.md).

The R1 correction authenticates, bounds, and parses one historical-producer metadata
buffer through `readJSON`. Its regression demonstrates rejection above 64 MiB.
The retained red attempt failed and the corrected regression, package tests, and
vet passed. This does not establish a demonstrated capture bypass.

## Accepted evidence

The accepted source tree is `72cdc40a7c231366f1f0fe05c20d64fc8eb4d158`.
Its 762-file manifest has SHA256
`d7e53dece49d5f163e7d153fdf607e0dde1732ce0147eccee30a0149938139bb`.
[The source manifest](../provenance/caller-r1-accepted-source.json) records its
members and the packaging changes. The owned caller/R1 review covered a 49-path
delta over the integrated dependency snapshot. Those dependencies are included
here because the caller cannot compile against the earlier committed baseline.
This is an integration checkpoint, not 407 newly authored caller files.

Two independent Sol XHigh processes accepted the frozen delta with no unresolved
findings. Their report hashes are
`e0c38a9a21835960177310abde9e53b157a9bc23bf8d520a0b5d77f6d20781ff` and
`ba6532ed3f57b0f549c94fa06c392392674d067f6a2784ae8637643ac267e9cb`.
The reviews use the same model family and provide no cross-model validation.

The retained final verification receipt has SHA256
`14bb2d08c100399438c4046a48634421bab342532cb97cefbbd3d3492fc6b086`.
It records these commands and outcomes:

- `go test -p 2 -count=1 -json -timeout=180s ./...` exited 0 with 755 root passes
  and 66 opt-in root skips.
- `verification/verify.sh --source "$PWD/external/dorado-source"` exited 0 with
  47 root passes and no skips in the selected CPU parity set.
- The selected saved-lineage readback exited 0 with one root pass and no skips.
  It did not execute a native producer.

The packaging audit rehashed all 638 CPU inputs, 29 artifacts, 14 native source
files, nine tool and numerical-library declarations, and 1,439 saved-lineage
inputs. It also checked the accepted source and raw test logs. These are reused
results from the unchanged declared inputs, not new executions. The CPU parity
receipt has SHA256
`09b0e094a442a7a9a86a0f36fadea812be63432bcf70bb0f9caab7c10ac1e67e`.

Raw receipts and payloads remain under ignored project artifact paths. On the
originating host, the review and verification summaries are in
`artifacts/pstack-herdr-parity-20261026/`; the packaging audit is task `t-487826`.
The shipping worktree excludes generated Python bytecode, adds cache ignore rules,
and removes one trailing blank line from a native header. It adds this note and
the accepted manifest. No Go computation or native statement changes.

## Limits

The ordinary suite skips opt-in native tests. The maintained parity result covers
its selected CPU core. Neither result establishes current native backend, GPU,
complete caller, biological, or performance parity. The production bundle approval
registry remains empty. Development and validation remain the candidate data;
HG005 results do not select this implementation. No merge is authorized.
