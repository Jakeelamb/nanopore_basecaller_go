package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

func main() {
	out := flag.String("attempt", "", "retained H08 host attempt")
	flag.Parse()
	h08 := []string{"TestFixedStageContractsAreProspectiveAndComplete", "TestFixedStageMaskRejectsDefinedChangesAndKeepsDeclaredExclusions", "TestFixedCacheComparatorRejectsSameShapedForeignPayload", "TestFixedConsumerAdmissionPrecedesLockAndNativeLoader", "TestFixedRetainedInputControlUsesIndependentDigest", "TestBackendSelectionKeepsOutputIdentityEvenWithEqualBytes", "TestH08CPU13OverlayOnlyWrapsMaintainedTestMain", "TestH08GuardContractRejectsRehashedPhaseAndCustodyChanges", "TestH08StatusesRequireExactNamedTestsAndNoSkips", "TestH08RuntimeCannotArmFromReceiptOrFlag", "TestH08ChildRequiresPrivateParentBeforeRequestRead", "TestH08CPUFixtureGateRejectsWrongDistributionBeforeLoading"}
	if err := runner.RequireGoTestRoots(filepath.Join(*out, "h08.stdout"), "github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu/h08", h08); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if err := runner.RequireGoTestRoots(filepath.Join(*out, "fixture-interface.stdout"), "github.com/Jakeelamb/nanopore_basecaller_go/internal/tensor", []string{"TestNativeFixtureDirectorySelection"}); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("13 required named tests passed exactly once; CLI package has no tests")
}
