package h08

import (
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu/h08/cgobuild"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

func fakeProofHostFile(t *testing.T, dir, name, body string) Pin {
	t.Helper()
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte(body), 0600); err != nil {
		t.Fatal(err)
	}
	return Pin{Path: path, SHA256: digest([]byte(body))}
}

// syntheticFakeProof exercises metadata custody only. Its selected pair is a
// failed-build fixture, and its text snapshots are never compiled or loaded.
func syntheticFakeProof(t *testing.T, compileCount int) FakeProof {
	t.Helper()
	dir := t.TempDir()
	file := func(name, body string) Pin { return fakeProofHostFile(t, dir, name, body) }
	build, selection := rejectedBuildPair(t, cgobuild.CPU13Tests)
	target := file("target.txt", "synthetic target identity; not executable")
	logger := file("logger.txt", "synthetic logger identity; not executable")
	compiler := file("compiler.txt", "synthetic compiler identity; not executable")
	names := []string{"TestNativeLoaderABIRejections", "TestNativeCloseFailureRetainsLibrary"}
	var raw strings.Builder
	for _, name := range names {
		fmt.Fprintf(&raw, "=== RUN   %s\n--- PASS: %s (0.00s)\n", name, name)
	}
	stdout, stderr := file("stdout.txt", raw.String()), file("stderr.txt", "")
	command := runner.CommandReceipt{Started: "2026-10-28T00:00:00Z", Finished: "2026-10-28T00:00:01Z", Stdout: stdout.Path, StdoutSHA: stdout.SHA256, Stderr: stderr.Path, StderrSHA: stderr.SHA256}
	r := FakeProof{Schema: "nanopore-h08-fake-loader-proof/v2", Status: "two_fake_loader_tests_passed", ControllerBuild: build, ControllerBuildSelection: selection, Target: target, Logger: logger, Compiler: compiler, Command: command, Inputs: map[string]string{}, Tests: map[string]string{}}
	r.Command.Args = []string{target.Path, "-h08-profile", "fake", "-test.v", "-test.count=1", "-test.run", "^(TestNativeLoaderABIRejections|TestNativeCloseFailureRetainsLibrary)$"}
	for _, pin := range []Pin{build, selection, target, logger, compiler} {
		r.Inputs[pin.Path] = pin.SHA256
	}
	for _, name := range names {
		r.Tests[name] = "pass"
	}
	compileLog := file("compile-log.txt", "")
	for i := 0; i < compileCount; i++ {
		sourceBefore := file(fmt.Sprintf("source-before-%d.txt", i), fmt.Sprintf("/* Synthetic custody fixture %d; never compiled. */\nint host_fixture;\n", i))
		outputSnapshot := file(fmt.Sprintf("output-snapshot-%d.txt", i), fmt.Sprintf("synthetic output snapshot %d; not a native library\n", i))
		source := Pin{Path: filepath.Join(dir, fmt.Sprintf("removed-source-%d.c", i)), SHA256: sourceBefore.SHA256}
		output := Pin{Path: filepath.Join(dir, fmt.Sprintf("removed-output-%d.so", i)), SHA256: outputSnapshot.SHA256}
		c := FakeCompile{Source: source, SourceBefore: sourceBefore, Output: output, OutputSnapshot: outputSnapshot, Command: command}
		c.Command.Args = []string{compiler.Path, "-shared", "-fPIC", source.Path, "-o", output.Path}
		c.Command.Stdout, c.Command.StdoutSHA = compileLog.Path, compileLog.SHA256
		c.Command.Stderr, c.Command.StderrSHA = compileLog.Path, compileLog.SHA256
		r.Compiles = append(r.Compiles, c)
	}
	return r
}

func TestH08ABI5FakeProofProfile(t *testing.T) {
	// Independently pinned from the selected source: 4 initial + (45-11)
	// missing-symbol + 1 close-failure command. Do not use the reader's constant.
	r := syntheticFakeProof(t, 39)
	check := func(proof FakeProof) error {
		_, err := readFakeProof(hostJSON(t, filepath.Join(t.TempDir(), "proof.json"), proof), r.Target, r.ControllerBuild, r.ControllerBuildSelection)
		return err
	}
	t.Run("complete-39", func(t *testing.T) {
		if err := check(r); err != nil {
			t.Fatal(err)
		}
	})
	for _, count := range []int{34, 38, 40} {
		t.Run(fmt.Sprintf("reject-count-%d", count), func(t *testing.T) {
			mutant := r
			mutant.Compiles = syntheticFakeProof(t, count).Compiles
			for i := range mutant.Compiles {
				mutant.Compiles[i].Command.Args[0] = r.Compiler.Path
			}
			requireSelectionError(t, check(mutant), "fake-only target/selection/custody differs")
		})
	}
	raw, err := readPin(Pin{Path: r.Command.Stdout, SHA256: r.Command.StdoutSHA})
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{
		"target", "build", "selection", "profile", "selector", "status",
		"target-membership", "logger-membership", "compiler-membership", "input-hash",
		"compiler-recipe", "c-only-recipe", "source-before-hash", "source-before-bytes", "source-before-missing",
		"output-snapshot-hash", "output-snapshot-bytes", "output-snapshot-missing", "duplicate-output", "framework-source",
		"host-exit", "compile-exit", "compile-clocks", "compile-log-hash",
		"test-skip", "test-fail", "test-missing", "test-extra", "test-duplicate", "test-repeat-run", "test-map", "test-map-extra",
	} {
		t.Run(name, func(t *testing.T) {
			mutant := r
			mutant.Inputs = maps.Clone(r.Inputs)
			mutant.Tests = maps.Clone(r.Tests)
			mutant.Command.Args = slices.Clone(r.Command.Args)
			mutant.Compiles = slices.Clone(r.Compiles)
			mutant.Compiles[0].Command.Args = slices.Clone(r.Compiles[0].Command.Args)
			c := &mutant.Compiles[0]
			file := func(label, body string) Pin { return fakeProofHostFile(t, t.TempDir(), label, body) }
			setRaw := func(body string) {
				pin := file("stdout.txt", body)
				mutant.Command.Stdout, mutant.Command.StdoutSHA = pin.Path, pin.SHA256
			}
			want := "fake-only target/selection/custody differs"
			switch name {
			case "target":
				mutant.Target = r.Logger
			case "build":
				mutant.ControllerBuild = r.Target
			case "selection":
				mutant.ControllerBuildSelection = r.Target
			case "profile":
				mutant.Command.Args[2] = "genuine"
			case "selector":
				mutant.Command.Args[6] = "TestNativeTensorOperations"
			case "status":
				mutant.Status = "failed"
			case "target-membership", "logger-membership", "compiler-membership":
				pin := map[string]Pin{"target-membership": r.Target, "logger-membership": r.Logger, "compiler-membership": r.Compiler}[name]
				delete(mutant.Inputs, pin.Path)
				want = "fake input custody omitted"
			case "input-hash":
				mutant.Inputs[file("changed-input.txt", "changed").Path] = strings.Repeat("0", 64)
				want = "retained SHA256 mismatch"
			case "compiler-recipe", "c-only-recipe", "source-before-hash", "output-snapshot-hash", "duplicate-output":
				want = "fake C-only recipe/source-before-compiler differs"
				switch name {
				case "compiler-recipe":
					c.Command.Args[0] = r.Logger.Path
				case "c-only-recipe":
					c.Command.Args[2] = "-xc++"
				case "source-before-hash":
					c.SourceBefore.SHA256 = strings.Repeat("0", 64)
				case "output-snapshot-hash":
					c.OutputSnapshot.SHA256 = strings.Repeat("0", 64)
				case "duplicate-output":
					c.Output = r.Compiles[1].Output
					c.OutputSnapshot = r.Compiles[1].OutputSnapshot
					c.Command.Args[5] = c.Output.Path
				}
			case "source-before-bytes", "output-snapshot-bytes", "compile-log-hash":
				changed := file("changed-snapshot.txt", "changed bytes")
				want = "retained SHA256 mismatch"
				switch name {
				case "source-before-bytes":
					c.SourceBefore.Path = changed.Path
				case "output-snapshot-bytes":
					c.OutputSnapshot.Path = changed.Path
				case "compile-log-hash":
					c.Command.Stdout = changed.Path
				}
			case "source-before-missing", "output-snapshot-missing":
				missing := filepath.Join(t.TempDir(), "absent-snapshot.txt")
				if name == "source-before-missing" {
					c.SourceBefore.Path = missing
				} else {
					c.OutputSnapshot.Path = missing
				}
				want = "no such file or directory"
			case "framework-source":
				c.SourceBefore = file("framework-source.txt", "#include <torch/torch.h>\n")
				c.Source.SHA256 = c.SourceBefore.SHA256
				want = "fake source contains genuine framework dependency"
			case "host-exit", "compile-exit", "compile-clocks":
				want = "host command exit/clocks differ"
				if name == "host-exit" {
					mutant.Command.Exit = 1
				} else if name == "compile-exit" {
					c.Command.Exit = 1
				} else {
					c.Command.Finished = "2026-10-27T00:00:00Z"
				}
			case "test-skip", "test-fail", "test-missing", "test-extra", "test-duplicate", "test-repeat-run":
				body := string(raw)
				switch name {
				case "test-skip":
					body = strings.Replace(body, "--- PASS:", "--- SKIP:", 1)
				case "test-fail":
					body = strings.Replace(body, "--- PASS:", "--- FAIL:", 1)
				case "test-missing":
					body = strings.Replace(body, "--- PASS: TestNativeCloseFailureRetainsLibrary (0.00s)\n", "", 1)
				case "test-extra":
					body += "=== RUN   TestUnexpected\n--- PASS: TestUnexpected (0.00s)\n"
				case "test-duplicate":
					body += "--- PASS: TestNativeLoaderABIRejections (0.00s)\n"
				case "test-repeat-run":
					body += "=== RUN   TestNativeLoaderABIRejections\n"
				}
				setRaw(body)
				want = "fake-only raw test result differs"
			case "test-map":
				mutant.Tests["TestNativeLoaderABIRejections"] = "skip"
				want = "fake result mismatch"
			case "test-map-extra":
				mutant.Tests["TestUnexpected"] = "pass"
				want = "fake-only raw test result differs"
			}
			requireSelectionError(t, check(mutant), want)
		})
	}
}
