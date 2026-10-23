// Command check retains standalone child materialization verification attempts.
package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"time"

	oracle "github.com/Jakeelamb/nanopore_basecaller_go/verification/subread"
)

func pins() (map[string]string, error) {
	out := map[string]string{}
	for _, root := range []string{"internal/subread", "verification/subread", "docs/subread.md", "go.mod"} {
		if err := filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if d.IsDir() {
				return nil
			}
			data, err := os.ReadFile(path)
			if err != nil {
				return err
			}
			out[path] = oracle.Hash(data)
			return nil
		}); err != nil {
			return nil, err
		}
	}
	return out, nil
}
func run() error {
	native := len(os.Args) > 1
	if native && (len(os.Args) != 2 || os.Args[1] != "--native") {
		return fmt.Errorf("usage: verification/subread/verify.sh [--native]")
	}
	if native && os.Getenv("DORADO_SOURCE") == "" {
		return fmt.Errorf("--native requires DORADO_SOURCE; no inference or GPU is used")
	}
	root, err := os.Getwd()
	if err != nil {
		return err
	}
	if _, err = os.Stat("internal/subread/subread.go"); err != nil {
		return fmt.Errorf("run from repository root: %w", err)
	}
	output := os.Getenv("SUBREAD_ARTIFACT_ROOT")
	if output == "" {
		output = filepath.Join(root, "artifacts/subread")
	}
	output, err = filepath.Abs(output)
	if err != nil {
		return err
	}
	if err = os.MkdirAll(output, 0700); err != nil {
		return err
	}
	attempt, err := os.MkdirTemp(output, "checks-"+time.Now().UTC().Format("20060102T150405Z")+"-")
	if err != nil {
		return err
	}
	fmt.Println(attempt)
	before, err := pins()
	if err != nil {
		return err
	}
	head, err := oracle.Command(attempt, "git-head", exec.Command("git", "rev-parse", "HEAD"), nil)
	if err != nil {
		return err
	}
	version, err := oracle.Command(attempt, "go-version", exec.Command("go", "version"), nil)
	if err != nil {
		return err
	}
	goPath, err := exec.LookPath("go")
	if err != nil {
		return err
	}
	goData, err := os.ReadFile(goPath)
	if err != nil {
		return err
	}
	self, err := os.Executable()
	if err != nil {
		return err
	}
	selfData, err := os.ReadFile(self)
	if err != nil {
		return err
	}
	if err = oracle.SaveJSON(filepath.Join(attempt, "inputs.json"), map[string]any{"source_pins": before, "git_head": strings.TrimSpace(string(head)), "go_version": strings.TrimSpace(string(version)), "go_binary_sha256": oracle.Hash(goData), "checker_sha256": oracle.Hash(selfData), "native_requested": native, "native_source": os.Getenv("DORADO_SOURCE")}); err != nil {
		return err
	}
	type check struct {
		name      string
		args, env []string
	}
	checks := []check{
		{"ordinary", []string{"test", "-count=1", "-v", "./internal/subread", "./verification/subread/..."}, nil},
		{"race", []string{"test", "-race", "-count=1", "./internal/subread", "./verification/subread/..."}, []string{"CGO_ENABLED=1"}},
		{"no-cgo", []string{"test", "-count=1", "./internal/subread", "./verification/subread/..."}, []string{"CGO_ENABLED=0"}},
		{"vet", []string{"vet", "./internal/subread", "./verification/subread/..."}, nil},
		{"mutations", []string{"test", "-count=1", "-v", "./verification/subread", "-run", "^TestBehaviorMutations$"}, []string{"SUBREAD_MUTATIONS=1"}},
	}
	if native {
		checks = append(checks, check{"native", []string{"test", "-count=1", "-v", "./verification/subread", "-run", "^TestPinnedNativeMaterialization$"}, []string{"DORADO_SOURCE=" + os.Getenv("DORADO_SOURCE")}})
	}
	failed := false
	for _, check := range checks {
		cmd := exec.Command("go", check.args...)
		cmd.Env = append(os.Environ(), "DORADO_SOURCE=", "SUBREAD_MUTATIONS=", "SUBREAD_ARTIFACT_ROOT="+attempt)
		cmd.Env = append(cmd.Env, check.env...)
		stdout, runErr := oracle.Command(attempt, check.name, cmd, nil)
		if runErr != nil {
			failed = true
			fmt.Printf("%s FAIL\n%s\n%s\n", check.name, runErr, stdout)
		} else {
			fmt.Printf("%s PASS\n", check.name)
		}
	}
	after, err := pins()
	if err != nil {
		return err
	}
	frozen := reflect.DeepEqual(before, after)
	if !frozen {
		failed = true
	}
	if err = oracle.SaveJSON(filepath.Join(attempt, "manifest.json"), map[string]any{"passed": !failed, "source_frozen": frozen, "source_pins_before": before, "source_pins_after": after, "native_requested": native, "qualification": "standalone supplied-spacer child materialization only; individual command receipts retain exit, clocks, and raw streams; no caller, inference, GPU, biological accuracy or performance claim"}); err != nil {
		return err
	}
	if failed {
		return fmt.Errorf("subread verification failed; retained at %s", attempt)
	}
	return nil
}
func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
