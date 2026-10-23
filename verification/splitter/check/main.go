// Command check preserves each simplex splitter acceptance attempt.
package main

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"time"

	oracle "github.com/Jakeelamb/nanopore_basecaller_go/verification/splitter"
)

func save(path string, v any) error {
	data, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(data, '\n'), 0600)
}
func sourcePins() (map[string]string, error) {
	pins := map[string]string{}
	for _, path := range []string{"internal/splitter", "verification/splitter", "internal/align", "internal/signal/half.go", "docs/splitter.md", "go.mod"} {
		if err := filepath.WalkDir(path, func(path string, d os.DirEntry, e error) error {
			if e != nil {
				return e
			}
			if d.IsDir() {
				return nil
			}
			data, e := os.ReadFile(path)
			if e != nil {
				return e
			}
			pins[path] = fmt.Sprintf("%x", sha256.Sum256(data))
			return nil
		}); err != nil {
			return nil, err
		}
	}
	return pins, nil
}
func run() error {
	native := len(os.Args) > 1
	if native && (len(os.Args) != 2 || os.Args[1] != "--native") {
		return fmt.Errorf("usage: verification/splitter/verify.sh [--native]")
	}
	keys := []string{"DORADO_SOURCE", "SPLITTER_TORCH", "SPLITTER_EDLIB_SOURCE", "SPLITTER_CAPTURE_MANIFESTS", "SPLITTER_MOVES_RECEIPT"}
	if native {
		for _, key := range keys {
			if os.Getenv(key) == "" {
				return fmt.Errorf("--native requires %s; no inference is run", key)
			}
		}
	}
	root, err := os.Getwd()
	if err != nil {
		return err
	}
	if _, err := os.Stat(filepath.Join(root, "internal/splitter/splitter.go")); err != nil {
		return fmt.Errorf("run from repository root: %w", err)
	}
	output := os.Getenv("SPLITTER_ARTIFACT_ROOT")
	if output == "" {
		output = filepath.Join(root, "artifacts/splitter")
	}
	output, err = filepath.Abs(output)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(output, 0700); err != nil {
		return err
	}
	attempt, err := os.MkdirTemp(output, "checks-"+time.Now().UTC().Format("20060102T150405Z")+"-")
	if err != nil {
		return err
	}
	fmt.Println(attempt)
	pins, err := sourcePins()
	if err != nil {
		return err
	}
	head, err := exec.Command("git", "rev-parse", "HEAD").CombinedOutput()
	if err != nil {
		return err
	}
	version, err := exec.Command("go", "version").CombinedOutput()
	if err != nil {
		return err
	}
	goPath, err := exec.LookPath("go")
	if err != nil {
		return err
	}
	goBytes, err := os.ReadFile(goPath)
	if err != nil {
		return err
	}
	self, err := os.Executable()
	if err != nil {
		return err
	}
	selfBytes, err := os.ReadFile(self)
	if err != nil {
		return err
	}
	if err := save(filepath.Join(attempt, "inputs.json"), map[string]any{"source_pins": pins, "git_head": strings.TrimSpace(string(head)), "go_version": strings.TrimSpace(string(version)), "go_binary_sha256": fmt.Sprintf("%x", sha256.Sum256(goBytes)), "checker_sha256": fmt.Sprintf("%x", sha256.Sum256(selfBytes)), "native_requested": native}); err != nil {
		return err
	}
	type check struct {
		name      string
		args, env []string
	}
	checks := []check{
		{"repository", []string{"test", "-count=1", "./..."}, nil},
		{"ordinary", []string{"test", "-count=1", "-v", "./internal/splitter", "./verification/splitter/..."}, nil},
		{"race", []string{"test", "-race", "-count=1", "./internal/splitter", "./verification/splitter/..."}, []string{"CGO_ENABLED=1"}},
		{"no-cgo", []string{"test", "-count=1", "./internal/splitter", "./verification/splitter/..."}, []string{"CGO_ENABLED=0"}},
		{"vet", []string{"vet", "./internal/splitter", "./verification/splitter/..."}, nil},
		{"mutations", []string{"test", "-count=1", "-v", "./verification/splitter", "-run", "TestBehaviorMutations"}, []string{"SPLITTER_MUTATIONS=1"}},
	}
	if native {
		env := []string{}
		for _, key := range keys {
			env = append(env, key+"="+os.Getenv(key))
		}
		checks = append(checks, check{"native", []string{"test", "-count=1", "-v", "./internal/splitter", "-run", "TestPinnedNativeSimplexPA"}, env})
	}
	failed := false
	var receipts []map[string]any
	for _, c := range checks {
		cmd := exec.Command("go", c.args...)
		cmd.Env = append(os.Environ(), "SPLITTER_MUTATIONS=", "SPLITTER_ARTIFACT_ROOT="+attempt)
		for _, key := range keys {
			cmd.Env = append(cmd.Env, key+"=")
		}
		cmd.Env = append(cmd.Env, c.env...)
		_, runErr := oracle.Command(attempt, c.name, cmd, nil)
		data, err := os.ReadFile(filepath.Join(attempt, c.name+".receipt.json"))
		if err != nil {
			return err
		}
		var receipt map[string]any
		if err := json.Unmarshal(data, &receipt); err != nil {
			return err
		}
		receipt["name"] = c.name
		receipt["environment_overrides"] = append([]string{"native variables cleared", "SPLITTER_MUTATIONS=", "SPLITTER_ARTIFACT_ROOT=" + attempt}, c.env...)
		receipts = append(receipts, receipt)
		fmt.Printf("%s exit=%v\n", c.name, receipt["exit_code"])
		if runErr != nil {
			failed = true
			fmt.Println(runErr)
			log, _ := os.ReadFile(filepath.Join(attempt, c.name+".stdout"))
			fmt.Print(string(log))
		}
	}
	after, err := sourcePins()
	if err != nil {
		return err
	}
	frozen := reflect.DeepEqual(pins, after)
	if !frozen {
		failed = true
	}
	if err := save(filepath.Join(attempt, "final-source-pins.json"), after); err != nil {
		return err
	}
	if err := save(filepath.Join(attempt, "summary.json"), map[string]any{"checks": receipts, "passed": !failed, "native_qualified": native && !failed, "source_freeze": frozen, "qualification": "standalone simplex DNA PA pore-adapter decisions; no child materialization, pipeline integration, inference, GPU, performance or biological claim"}); err != nil {
		return err
	}
	if failed {
		return fmt.Errorf("acceptance failed; evidence retained at %s", attempt)
	}
	return nil
}
func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
