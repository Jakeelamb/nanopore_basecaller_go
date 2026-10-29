# Verify a genuine successful compiler artifact

`TestGenuineSuccessfulSelectedArtifact` authenticates saved successful compiler
evidence through `ReadH08Artifact` or `ReadCPU13Artifact`. The controller supplies
the real fixtures after compilation and independently selects the evidence Pin,
successful selection Pin, expected output Pin, and expected role.

Set all eight variables below. With none set, the test skips. Any partial
configuration, empty value, or unknown role fails. Pin paths must be clean absolute
paths. SHA256 values must be 64 lowercase hexadecimal characters.

```sh
H08_GENUINE_EVIDENCE_PATH='/absolute/selected/receipt.json' \
H08_GENUINE_EVIDENCE_SHA256='<independently-selected-receipt-sha256>' \
H08_GENUINE_SELECTION_PATH='/absolute/selected/successful-selection.json' \
H08_GENUINE_SELECTION_SHA256='<independently-selected-selection-sha256>' \
H08_GENUINE_OUTPUT_PATH='/absolute/selected/output' \
H08_GENUINE_OUTPUT_SHA256='<independently-selected-output-sha256>' \
H08_GENUINE_ROLE='h08-command' \
H08_GENUINE_CONTROLS_DIR='/absolute/retained/artifact-controls' \
go test ./verification/hacgpu/h08/cgobuild \
	-run '^TestGenuineSuccessfulSelectedArtifact$' -count=1 -v \
	> '/absolute/retained/h08-genuine-success.log' 2>&1
```

Replace every placeholder with an independent selection. Create the controls
parent directory and raw-log parent directory before invocation. Retain the
command's exit status beside its raw log. For CPU13, select its separate successful
receipt, selection, and output, set `H08_GENUINE_ROLE='cpu13-tensor-tests'`, and use
a separate raw log. The `H08_` prefix applies to both roles.

The selection must have schema `nanopore-h08-successful-selection/v1`, the exact
expected role, and all five producer Pins: `Receipt`, `Plan`, `Controller`,
`Queries`, and `Output`. The test reads the exact `successfulSelectionWire` through
`exactRead` and compares its receipt and output with the external selections.
Do not derive the expected output Pin from the selection being tested. Rehashing
self-authored metadata does not establish independent selection.

The selected public reader must authenticate the full successful proof and return
exactly the independently expected output Pin. The selected receipt must retain
`RuntimeHeld=true`. Only after that positive passes does the test create a fresh
`genuine-success-controls-*` directory under `H08_GENUINE_CONTROLS_DIR`.

The test preserves six JSON fixtures there. Each starts from the genuine selection
and changes only `Role`, `Receipt`, `Plan`, `Controller`, `Queries`, or `Output`.
The role changes to the other supported role. Each Pin mutation changes one digest
digit while preserving its path. Every fixture gets a fresh selection Pin and goes
through the selected public reader. Role and receipt controls require the
`role/receipt differs` diagnostic. The other controls require `tuple differs`.
Every rejection must expose a zero `Artifact.Output()`. These gates precede full
historical body validation, so the positive is not repeated for every mutation.

The test leaves the fixtures and `controls.log` on disk, including on failure.
The log records the original selections, each negative path and SHA256, expected
diagnostic, actual error, and returned output. Verbose test output also records the
fixture and log paths. No directory cleanup runs.

This proves compiler artifact authentication only. The test does not execute the
output, load a backend, query a compiler, acquire a lock, or call `CheckController`
on the test executable. It grants no output execution, GPU, CPU13 runtime,
basecalling, biological, or timing acceptance. A normal run that skips this test
provides no genuine compiler or native acceptance.
