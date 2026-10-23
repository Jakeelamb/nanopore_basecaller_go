# Go basecaller port

Read `/home/jake/Projects/agent-scripts/AGENTS.MD` when present.

This repository ports actual caller computation from pinned Dorado source to Go.
Calling the stock executable does not qualify as a port. Keep the original
`nanopore_basecaller` workspace and stock source unchanged.

Work on feature branches. Pull requests target `main`. Do not merge without
authorization. Keep source, fixtures manifests, verification tools, and concise
evidence in Git. Keep signals, weights, native distributions, generated fixture
payloads, raw receipts, and results outside Git under ignored paths.

Use `go test ./...` and the maintained parity command. A Go unit test is not
native parity, GPU qualification, end-to-end basecalling, or biological evidence.
Every completed port unit needs a pinned native oracle, rejection controls,
and reproducible differential evidence. Preserve failed attempts. Reuse
content-addressed outputs only when every declared input pin matches.

Use development and validation for candidate work. Do not use HG005 results
for selection. Preserve the stock experiment's fixed hypotheses and controls.
Go owns the new caller. Native GPU kernels may remain behind a narrow typed
boundary, explicitly listed in the port coverage document.
