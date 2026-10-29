# SUP metadata provider controls

The metadata guard must authenticate the providers actually opened by the
identity command. A prospective ELF closure and valid CSV alone are insufficient.
The CPU fixture selects provider A for both commands. One command opens A; the
other opens B by absolute path. A and B have distinct bytes and the same SONAME.
Each provider's constructor writes a separate marker.

Run all compilation and native controls in a controller-owned finite unit with
the existing stock lock, 2 CPU, 3 GiB memory, zero swap and 64 tasks. Arm the
watchdog before starting Python, Go or a native process. Retain raw commands,
clocks, exits and failed outputs. Release the lock only after the unit and its
cgroup are empty. These CPU loader controls do not qualify a GPU device or graph.

`verification/supgpu/tools/build_metadata_fixture.py` requires an explicit
canonical compiler path, independent compiler SHA256 and fresh output beneath
this checkout's `artifacts`. It compiles four files without executing them and
retains command receipts, depfiles, link maps and raw strace logs. This fixture
receipt is not the closed-build proof used for the caller backend.

Compile the Go0 graph test binary with `CGO_ENABLED=0 go test -c -o BINARY
./verification/supgpu/graph`. Run the binary from the repository root.
`TestPrepareMetadataProviderFixtureOptIn` reads independently pinned options
through `SUP_METADATA_PREPARE` and `SUP_METADATA_PREPARE_SHA256`. The closed
`nanopore-sup-metadata-prepare/v1` options select Root, fresh Output, sized Audit,
MatchedCLI, MismatchedCLI, ProviderA, ProviderB, MarkerA, MarkerB, SearchDirs and
TimeoutSeconds. Native pins have `Path`/`SHA256`; sized pins have
`path`/`sha256`/`bytes`. SearchDirs includes A's directory and canonical system
library directories; B's directory remains unselected.

The constructor emits `metadata-fixture.json`, two identity/guard-plan pairs
and an explicit zero-phase metadata contract. It validates them through the
production readers before publication. It neither compiles nor runs an ELF.

Select the emitted fixture with `SUP_METADATA_FIXTURE` and the independent
`SUP_METADATA_FIXTURE_SHA256`. `TestGenuineMetadataProviderBoundaryOptIn`
requires matched A acceptance, valid CSV, saved receipt replay and A's marker.
It then requires B's actual audited open to be denied and B's marker absent.
An earlier failure or unselected skip supplies no native qualification.

For the original regression, overlay
`graph/testdata/metadata_before_test.go.txt` as `graph/metadata_before_test.go`
in the frozen pre-correction graph. Select the same fixture and run
`TestOriginalMetadataProviderExpectedRejectionOptIn`. Its matched positive must
succeed first. The intended old failure explicitly reports that unselected B
returned valid metadata and ran its constructor. Preserve both raw receipts and
move only the two declared markers into retained before-evidence before running
the corrected control. Keep the four compiled ELF files unchanged.
