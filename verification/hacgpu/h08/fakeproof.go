package h08

import (
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

// FakeProof is host-only evidence for two maintained loader tests. It is not
// evidence about the genuine backend. The outer host runner supplies its pinned
// C-only logger and all source-before-compiler/output snapshots.
type FakeProof struct {
	Schema, Status                            string
	ControllerBuild, Target, Logger, Compiler Pin
	ControllerBuildSelection                  Pin
	Command                                   runner.CommandReceipt
	Inputs                                    map[string]string
	Compiles                                  []FakeCompile
	Tests                                     map[string]string
}
type FakeCompile struct {
	Command                                      runner.CommandReceipt
	Source, SourceBefore, Output, OutputSnapshot Pin
}

// The fixed ABI5 pair compiles four initial loader rejections, 34 missing-symbol
// cases (45 symbols, skipping the first 11), and one close-failure case.
const abi5FakeLoaderCompileCount = 39

func fakeCommand(target Pin) []string {
	return []string{target.Path, "-h08-profile", "fake", "-test.v", "-test.count=1", "-test.run", "^(" + strings.Join(fakeNames, "|") + ")$"}
}

func readCommand(c runner.CommandReceipt) error {
	a, e := time.Parse(time.RFC3339Nano, c.Started)
	b, f := time.Parse(time.RFC3339Nano, c.Finished)
	if e != nil || f != nil || b.Before(a) || c.Exit != 0 {
		return fmt.Errorf("host command exit/clocks differ")
	}
	for _, p := range []Pin{{Path: c.Stdout, SHA256: c.StdoutSHA}, {Path: c.Stderr, SHA256: c.StderrSHA}} {
		if _, e = readPin(p); e != nil {
			return e
		}
	}
	return nil
}

func readFakeProof(pin, target, build, selection Pin) (r FakeProof, err error) {
	if err = pinnedJSON(pin, &r); err != nil {
		return
	}
	if r.Schema != "nanopore-h08-fake-loader-proof/v2" || r.Status != "two_fake_loader_tests_passed" || r.Target != target || r.ControllerBuild != build || r.ControllerBuildSelection != selection || !slices.Equal(r.Command.Args, fakeCommand(target)) || len(r.Compiles) != abi5FakeLoaderCompileCount {
		return r, fmt.Errorf("fake-only target/selection/custody differs")
	}
	for _, p := range []Pin{target, build, selection, r.Logger, r.Compiler} {
		if !filepath.IsAbs(p.Path) || len(p.SHA256) != 64 || r.Inputs[p.Path] != p.SHA256 {
			return r, fmt.Errorf("fake input custody omitted")
		}
	}
	if err = checkPins(r.Inputs); err != nil {
		return
	}
	if err = readCommand(r.Command); err != nil {
		return
	}
	body, e := readPin(Pin{Path: r.Command.Stdout, SHA256: r.Command.StdoutSHA})
	if e != nil {
		return r, e
	}
	actual, e := testStatuses(body, fakeNames)
	if e != nil || len(actual) != len(r.Tests) {
		return r, fmt.Errorf("fake-only raw test result differs: %v", e)
	}
	for name, value := range actual {
		if r.Tests[name] != value {
			return r, fmt.Errorf("fake result mismatch")
		}
	}
	outputs := map[string]bool{}
	for _, c := range r.Compiles {
		args := []string{r.Compiler.Path, "-shared", "-fPIC", c.Source.Path, "-o", c.Output.Path}
		if !slices.Equal(c.Command.Args, args) || c.Source.SHA256 != c.SourceBefore.SHA256 || c.Output.SHA256 != c.OutputSnapshot.SHA256 || outputs[c.Output.Path] || !filepath.IsAbs(c.Source.Path) || !filepath.IsAbs(c.Output.Path) {
			return r, fmt.Errorf("fake C-only recipe/source-before-compiler differs")
		}
		outputs[c.Output.Path] = true
		if err = readCommand(c.Command); err != nil {
			return
		}
		for _, p := range []Pin{c.SourceBefore, c.OutputSnapshot} {
			if _, err = readPin(p); err != nil {
				return
			}
		}
		b, e := readPin(c.SourceBefore)
		if e != nil {
			return r, e
		}
		for _, include := range []string{"<torch", "<ATen", "<cuda", "\"cuda", "<koi", "\"koi"} {
			if strings.Contains(string(b), include) {
				return r, fmt.Errorf("fake source contains genuine framework dependency")
			}
		}
	}
	return
}

func runFakeTests(run func() int) int {
	// The exact selector and source-bound target keep this profile confined to
	// C-only fake DSOs. A genuine backend path is never installed.
	for key, value := range map[string]string{"NANOPORE_TENSOR_BACKEND": "/fake-only/no-backend.so", "NANOPORE_TENSOR_WEIGHT": "", "NANOPORE_TENSOR_FIXTURES": "", "NANOPORE_TENSOR_RUNTIME_MANIFEST": ""} {
		if e := os.Setenv(key, value); e != nil {
			fmt.Fprintln(os.Stderr, e)
			return 1
		}
	}
	return run()
}
